# Databricks notebook source
# MAGIC %md
# MAGIC # Fast video segmentation V2
# MAGIC
# MAGIC The source video is decoded once into resized, sampled JPEGs. SAM3 runs on a
# MAGIC warm GPU endpoint, masks stay compact, and FFmpeg renders directly without
# MAGIC retaining the video in memory. Transcript generation is intentionally separate.

# COMMAND ----------

dbutils.widgets.text("trigger_location", "")
dbutils.widgets.text("prompt", "weimaraner")
dbutils.widgets.text("frame_stride", "30")
dbutils.widgets.text("threshold", "0.5")
dbutils.widgets.text("truncate", "true")
dbutils.widgets.text("inference_width", "960")
dbutils.widgets.text("max_sampled_frames", "1200")
dbutils.widgets.text("endpoint_name", "sam3-video-frames-v2")
dbutils.widgets.text("volume_root", "")

trigger_location = dbutils.widgets.get("trigger_location").removeprefix("dbfs:")
prompt = dbutils.widgets.get("prompt").strip()
frame_stride = int(dbutils.widgets.get("frame_stride"))
threshold = float(dbutils.widgets.get("threshold"))
truncate = dbutils.widgets.get("truncate").lower() in {"1", "true", "yes"}
inference_width = int(dbutils.widgets.get("inference_width"))
max_sampled_frames = int(dbutils.widgets.get("max_sampled_frames"))
endpoint_name = dbutils.widgets.get("endpoint_name").strip()
volume_root = dbutils.widgets.get("volume_root").rstrip("/")

if not trigger_location or not prompt or not endpoint_name:
    raise ValueError("trigger_location, prompt, and endpoint_name are required.")
if frame_stride < 1 or inference_width < 320 or max_sampled_frames < 1:
    raise ValueError(
        "frame_stride and max_sampled_frames must be positive; "
        "inference_width must be >= 320."
    )
if not 0.0 < threshold < 1.0:
    raise ValueError("threshold must be between 0 and 1.")

# COMMAND ----------

import base64
import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from databricks.sdk import WorkspaceClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("segment-video-v2")

INPUT_ROOT = f"{volume_root}/inputs"
OUTPUT_ROOT = f"{volume_root}/outputs"
MANIFEST_ROOT = f"{volume_root}/manifests"
FRAME_ROOT = f"{volume_root}/transcript_frames"
source = Path(trigger_location)

try:
    source.relative_to(INPUT_ROOT)
except ValueError as error:
    raise ValueError(f"Input must be below {INPUT_ROOT}: {source}") from error
if not source.is_file():
    raise FileNotFoundError(source)

output_path = Path(OUTPUT_ROOT) / source.name
manifest_path = Path(MANIFEST_ROOT) / f"{source.stem}.json"
transcript_frame_dir = Path(FRAME_ROOT) / source.stem
for directory in (output_path.parent, manifest_path.parent, transcript_frame_dir):
    directory.mkdir(parents=True, exist_ok=True)


def timed(stage, timings):
    class Timer:
        def __enter__(self):
            self.started = time.perf_counter()
            return self

        def __exit__(self, *_):
            timings[stage] = round(time.perf_counter() - self.started, 3)
            logger.info("%s completed in %.3fs", stage, timings[stage])

    return Timer()


def video_metadata(path):
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": frame_count / fps if frame_count else None,
    }


def extract_sampled_frames(source_path, destination, stride, max_width):
    destination.mkdir(parents=True, exist_ok=True)
    output_pattern = str(destination / "frame-%08d.jpg")
    video_filter = (
        f"select=not(mod(n\\,{stride})),"
        f"scale=min({max_width}\\,iw):-2"
    )
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-vf",
            video_filter,
            "-vsync",
            "vfr",
            "-q:v",
            "3",
            output_pattern,
        ],
        check=True,
    )
    return sorted(destination.glob("frame-*.jpg"))


def endpoint_predictions(client, records, attempts=3):
    delay = 1.0
    for attempt in range(attempts):
        try:
            response = client.serving_endpoints.query(
                name=endpoint_name,
                dataframe_records=records,
            )
            predictions = response.predictions
            if predictions is None:
                payload = response.as_dict()
                predictions = payload.get("predictions")
            if not isinstance(predictions, list):
                raise RuntimeError("SAM3 endpoint returned no prediction records.")
            return predictions
        except Exception:
            if attempt == attempts - 1:
                raise
            logger.warning("Endpoint request failed; retrying in %.1fs", delay)
            time.sleep(delay)
            delay *= 2


def decode_mask(encoded):
    contents = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    mask = cv2.imdecode(contents, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError("Could not decode endpoint mask.")
    return mask


def add_overlay(frame, masks, frame_index, fps):
    overlay = frame.copy()
    for encoded in masks:
        mask = decode_mask(encoded)
        overlay[mask > 0] = (0, 255, 0)
    composited = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    seconds = frame_index / fps
    text = f"{seconds:08.2f}s   Count: {len(masks)}"
    cv2.rectangle(composited, (12, 12), (500, 58), (0, 0, 0), -1)
    cv2.putText(
        composited,
        text,
        (24, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return composited


def start_encoder(path, width, height, fps):
    return subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def keep_representative(reservoir, item, seen, limit=24):
    if len(reservoir) < limit:
        reservoir.append(item)
        return
    replacement = random.Random(seen).randrange(seen)
    if replacement < limit:
        reservoir[replacement] = item

# COMMAND ----------

timings = {}
metadata = video_metadata(source)
estimated_sampled_frames = (
    (metadata["frame_count"] + frame_stride - 1) // frame_stride
    if metadata["frame_count"]
    else None
)
if estimated_sampled_frames and estimated_sampled_frames > max_sampled_frames:
    raise ValueError(
        f"Video would produce {estimated_sampled_frames} sampled frames, exceeding "
        f"the interactive cap of {max_sampled_frames}. Increase frame_stride or use "
        "an offline processing path."
    )
logger.info(
    "source=%s size_mb=%.2f duration=%.2fs dimensions=%dx%d fps=%.3f",
    source,
    source.stat().st_size / 1024 / 1024,
    metadata["duration_seconds"] or 0,
    metadata["width"],
    metadata["height"],
    metadata["fps"],
)

with tempfile.TemporaryDirectory(prefix="sam3-v2-") as temporary:
    temporary_root = Path(temporary)
    sampled_dir = temporary_root / "sampled"
    local_output = temporary_root / f"output{source.suffix or '.mp4'}"

    with timed("extract_sampled_frames", timings):
        sampled_frames = extract_sampled_frames(
            source,
            sampled_dir,
            frame_stride,
            inference_width,
        )
    if not sampled_frames:
        raise RuntimeError("FFmpeg extracted no frames from the source video.")
    if len(sampled_frames) > max_sampled_frames:
        raise ValueError(
            f"Extracted {len(sampled_frames)} frames, exceeding the interactive cap "
            f"of {max_sampled_frames}."
        )

    first_frame = cv2.imread(str(sampled_frames[0]))
    height, width = first_frame.shape[:2]
    output_fps = max(1.0, metadata["fps"] / frame_stride * 5.0)
    encoder = start_encoder(local_output, width, height, output_fps)
    client = WorkspaceClient()
    representative_frames = []
    matched_count = 0
    written_count = 0
    batch_size = 8

    try:
        with timed("endpoint_inference_and_render", timings):
            for offset in range(0, len(sampled_frames), batch_size):
                paths = sampled_frames[offset : offset + batch_size]
                records = []
                batch_bytes = 0
                for sequence, path in enumerate(paths, start=offset):
                    encoded_image = base64.b64encode(path.read_bytes()).decode("ascii")
                    batch_bytes += len(encoded_image)
                    records.append(
                        {
                            "frame_id": str(sequence),
                            "image_b64": encoded_image,
                            "prompt": prompt,
                            "threshold": threshold,
                            "mask_threshold": 0.5,
                        }
                    )
                if batch_bytes > 12 * 1024 * 1024:
                    raise ValueError(
                        "Encoded endpoint batch exceeds the 12 MiB safety cap. "
                        "Reduce inference_width or endpoint batch size."
                    )

                predictions = endpoint_predictions(client, records)
                by_id = {str(item["frame_id"]): item for item in predictions}
                for sequence, path in enumerate(paths, start=offset):
                    prediction = by_id[str(sequence)]
                    masks = json.loads(prediction["masks_json"])
                    frame = cv2.imread(str(path))
                    if frame is None:
                        raise RuntimeError(f"Could not read sampled frame {path}.")
                    if masks:
                        matched_count += 1
                        keep_representative(
                            representative_frames,
                            (sequence, path.read_bytes()),
                            matched_count,
                        )
                    if masks or not truncate:
                        rendered = add_overlay(
                            frame,
                            masks,
                            sequence * frame_stride,
                            metadata["fps"],
                        )
                        encoder.stdin.write(rendered.tobytes())
                        written_count += 1

            if written_count == 0:
                empty = first_frame.copy()
                cv2.putText(
                    empty,
                    f'No matches for "{prompt}"',
                    (24, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                encoder.stdin.write(empty.tobytes())
                written_count = 1
    finally:
        if encoder.stdin:
            encoder.stdin.close()
        return_code = encoder.wait()
        if return_code:
            raise RuntimeError(f"FFmpeg encoder exited with status {return_code}.")

    with timed("publish_artifacts", timings):
        shutil.copy2(local_output, output_path)
        for old_frame in transcript_frame_dir.glob("*.jpg"):
            old_frame.unlink()

        transcript_frames = []
        for sequence, contents in sorted(representative_frames):
            frame_index = sequence * frame_stride
            frame_path = transcript_frame_dir / f"frame-{frame_index:08d}.jpg"
            frame_path.write_bytes(contents)
            transcript_frames.append(
                {
                    "path": str(frame_path),
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / metadata["fps"], 3),
                }
            )

        manifest = {
            "version": 2,
            "source_path": str(source),
            "output_path": str(output_path),
            "prompt": prompt,
            "frame_stride": frame_stride,
            "threshold": threshold,
            "inference_width": inference_width,
            "max_sampled_frames": max_sampled_frames,
            "source": metadata,
            "sampled_frame_count": len(sampled_frames),
            "matched_frame_count": matched_count,
            "written_frame_count": written_count,
            "transcript_frames": transcript_frames,
            "timings_seconds": timings,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

logger.info(
    "complete output=%s output_mb=%.2f manifest=%s timings=%s",
    output_path,
    output_path.stat().st_size / 1024 / 1024,
    manifest_path,
    timings,
)
dbutils.notebook.exit(
    json.dumps(
        {
            "output_path": str(output_path),
            "manifest_path": str(manifest_path),
            "matched_frame_count": matched_count,
            "timings_seconds": timings,
        }
    )
)
