import base64
import json
import logging
import random
import subprocess
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np
from databricks.sdk import WorkspaceClient

from telemetry import elapsed_ms, emit


logger = logging.getLogger("cv-video-app.live")
# OpenCV otherwise changes its internal worker count with the App compute size.
# Frame-level parallelism below is explicit, bounded, and order-preserving.
cv2.setNumThreads(1)

MAX_ENDPOINT_BATCH_BYTES = 12 * 1024**2
INFERENCE_WIDTH = 960
ENDPOINT_BATCH_SIZE = 8
# baldie: Each worker must map to a distinct endpoint and preserve ordered writes.
MAX_ENDPOINT_CONCURRENCY = 4
RENDER_CONCURRENCY = 4
TRANSCRIPT_FRAME_LIMIT = 24
TRANSCRIPT_FRAME_WIDTH = 1280
PreviewCallback = Callable[[dict[str, Any]], None]


def ffmpeg_executable() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass(frozen=True)
class AnnotatedFrame:
    frame_index: int
    timestamp_seconds: float
    image_b64: str


@dataclass(frozen=True)
class LiveSegmentationResult:
    sampled_frame_count: int
    matched_frame_count: int
    written_frame_count: int
    timings_seconds: dict[str, float]
    transcript_frames: tuple[AnnotatedFrame, ...]

    @property
    def elapsed_seconds(self) -> int:
        if "segmentation_pipeline" in self.timings_seconds:
            return max(1, round(self.timings_seconds["segmentation_pipeline"]))
        return max(1, round(sum(self.timings_seconds.values())))


@dataclass(frozen=True)
class PredictedBatch:
    offset: int
    paths: tuple[Path, ...]
    predictions: list[dict[str, Any]]
    request_bytes: int
    inference_duration_ms: int
    endpoint_name: str


@dataclass
class ExtractionStats:
    frame_count: int = 0
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class RenderedFrame:
    sequence: int
    masks: list[str]
    image: np.ndarray
    transcript_image_b64: str | None


def video_metadata(path: Path) -> tuple[float, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if frame_count < 1:
        raise ValueError("The video has no readable frames.")
    return fps, frame_count


def extract_sampled_frames(
    source: Path,
    destination: Path,
    frame_stride: int,
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            f"select=not(mod(n\\,{frame_stride}))",
            "-vsync",
            "vfr",
            "-q:v",
            "2",
            str(destination / "frame-%08d.jpg"),
        ],
        check=True,
    )
    frames = sorted(destination.glob("frame-*.jpg"))
    if not frames:
        raise ValueError("The video has no readable frames.")
    return frames


def extract_sampled_frame_batches(
    source: Path,
    destination: Path,
    frame_stride: int,
    batch_size: int,
    stats: ExtractionStats,
) -> Iterator[tuple[int, tuple[Path, ...]]]:
    """Yield closed JPEG batches while FFmpeg continues decoding the video."""
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    process = subprocess.Popen(
        [
            ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            f"select=not(mod(n\\,{frame_stride}))",
            "-vsync",
            "vfr",
            "-q:v",
            "2",
            str(destination / "frame-%08d.jpg"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    next_file_number = 1
    offset = 0
    try:
        while True:
            return_code = process.poll()
            batch_paths = tuple(
                destination / f"frame-{number:08d}.jpg"
                for number in range(
                    next_file_number,
                    next_file_number + batch_size,
                )
            )
            sentinel = destination / (
                f"frame-{next_file_number + batch_size:08d}.jpg"
            )

            # FFmpeg writes sequentially. The next file proves that the preceding
            # batch is closed, so readers never observe a partial JPEG.
            if sentinel.exists():
                yield offset, batch_paths
                next_file_number += batch_size
                offset += batch_size
                continue

            if return_code is None:
                time.sleep(0.05)
                continue

            stderr = process.stderr.read() if process.stderr else ""
            if return_code:
                raise RuntimeError(
                    f"FFmpeg frame extraction exited with status {return_code}: "
                    f"{stderr.strip()}"
                )

            final_paths: list[Path] = []
            while True:
                path = destination / f"frame-{next_file_number:08d}.jpg"
                if not path.exists():
                    break
                final_paths.append(path)
                next_file_number += 1
            if final_paths:
                yield offset, tuple(final_paths)
                offset += len(final_paths)
            if offset == 0:
                raise ValueError("The video has no readable frames.")
            stats.frame_count = offset
            stats.duration_seconds = round(time.perf_counter() - started, 3)
            return
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait()
        if process.stderr:
            process.stderr.close()


def encode_for_inference(frame: np.ndarray) -> str:
    height, width = frame.shape[:2]
    if width > INFERENCE_WIDTH:
        scaled_height = round(height * INFERENCE_WIDTH / width)
        frame = cv2.resize(
            frame,
            (INFERENCE_WIDTH, scaled_height),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("OpenCV could not encode a frame for inference.")
    return base64.b64encode(encoded).decode("ascii")


def encode_annotated_frame(frame: np.ndarray) -> str:
    height, width = frame.shape[:2]
    if width > TRANSCRIPT_FRAME_WIDTH:
        scaled_height = round(height * TRANSCRIPT_FRAME_WIDTH / width)
        frame = cv2.resize(
            frame,
            (TRANSCRIPT_FRAME_WIDTH, scaled_height),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError("OpenCV could not encode an annotated transcript frame.")
    return base64.b64encode(encoded).decode("ascii")


def encode_preview_frame(frame: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
    if not ok:
        raise RuntimeError("OpenCV could not encode a preview frame.")
    return base64.b64encode(encoded).decode("ascii")


def keep_representative_frame(
    reservoir: list[AnnotatedFrame],
    frame: AnnotatedFrame,
    seen: int,
) -> None:
    if len(reservoir) < TRANSCRIPT_FRAME_LIMIT:
        reservoir.append(frame)
        return
    replacement = random.Random(seen).randrange(seen)
    if replacement < TRANSCRIPT_FRAME_LIMIT:
        reservoir[replacement] = frame


def endpoint_predictions(
    client: WorkspaceClient,
    endpoint_name: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    response = client.serving_endpoints.query(
        name=endpoint_name,
        dataframe_records=records,
    )
    predictions = response.predictions
    if predictions is None:
        predictions = response.as_dict().get("predictions")
    if not isinstance(predictions, list):
        raise RuntimeError("SAM3 endpoint returned no prediction records.")
    return predictions


def predict_batch(
    client: WorkspaceClient,
    endpoint_name: str,
    prompt: str,
    threshold: float,
    offset: int,
    paths: tuple[Path, ...],
) -> PredictedBatch:
    records = []
    request_bytes = 0
    for sequence, path in enumerate(paths, start=offset):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Could not decode sampled frame {sequence}.")
        encoded = encode_for_inference(frame)
        request_bytes += len(encoded)
        records.append(
            {
                "frame_id": str(sequence),
                "image_b64": encoded,
                "prompt": prompt,
                "threshold": threshold,
                "mask_threshold": 0.5,
            }
        )
    if request_bytes > MAX_ENDPOINT_BATCH_BYTES:
        raise ValueError("An endpoint batch exceeds the 12 MiB live request cap.")

    started = time.perf_counter()
    predictions = endpoint_predictions(client, endpoint_name, records)
    return PredictedBatch(
        offset=offset,
        paths=paths,
        predictions=predictions,
        request_bytes=request_bytes,
        inference_duration_ms=elapsed_ms(started),
        endpoint_name=endpoint_name,
    )


def decode_mask(encoded: str) -> np.ndarray:
    contents = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    mask = cv2.imdecode(contents, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError("Could not decode endpoint mask.")
    return mask


def add_overlay(
    frame: np.ndarray,
    masks: list[str],
    frame_index: int,
    fps: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    for encoded in masks:
        mask = decode_mask(encoded)
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        overlay[mask > 0] = (0, 255, 0)
    composited = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    draw_label(composited, f"{frame_index / fps:08.2f}s   Count: {len(masks)}")
    return composited


def render_prediction_frame(
    sequence: int,
    path: Path,
    prediction: dict[str, Any],
    frame_stride: int,
    fps: float,
) -> RenderedFrame:
    masks = json.loads(prediction["masks_json"])
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Could not decode sampled frame {sequence}.")
    rendered = add_overlay(
        frame,
        masks,
        sequence * frame_stride,
        fps,
    )
    return RenderedFrame(
        sequence=sequence,
        masks=masks,
        image=rendered,
        transcript_image_b64=encode_annotated_frame(rendered) if masks else None,
    )


def draw_label(image: np.ndarray, text: str) -> None:
    width = image.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, width / 1280.0)
    thickness = max(1, round(font_scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    margin = round(12 * font_scale)
    cv2.rectangle(
        image,
        (margin, margin),
        (margin * 2 + text_width, margin * 2 + text_height + baseline),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        text,
        (margin + margin // 2, margin + text_height + margin // 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def start_encoder(
    output: Path,
    width: int,
    height: int,
    fps: float,
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            ffmpeg_executable(),
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
            str(output),
        ],
        stdin=subprocess.PIPE,
    )


def segment_video(
    client: WorkspaceClient,
    endpoint_name: str | tuple[str, ...],
    source: Path,
    output: Path,
    prompt: str,
    frame_stride: int,
    threshold: float,
    truncate: bool,
    output_fps: float | None = 8.0,
    preview_callback: PreviewCallback | None = None,
) -> LiveSegmentationResult:
    timings: dict[str, float] = {}
    fps, source_frame_count = video_metadata(source)
    endpoint_names = (
        (endpoint_name,) if isinstance(endpoint_name, str) else endpoint_name
    )
    if not endpoint_names:
        raise ValueError("At least one SAM3 endpoint is required.")
    endpoint_concurrency = min(MAX_ENDPOINT_CONCURRENCY, len(endpoint_names))
    emit(
        logger,
        "segmentation.started",
        endpoints=endpoint_names,
        source_bytes=source.stat().st_size,
        source_frames=source_frame_count,
        source_fps=round(fps, 3),
        frame_stride=frame_stride,
        threshold=threshold,
        truncate=truncate,
        output_fps=output_fps,
        inference_width=INFERENCE_WIDTH,
    )

    with tempfile.TemporaryDirectory(prefix="sam3-live-") as temporary:
        sampled_directory = Path(temporary) / "sampled"
        expected_sampled_frames = (
            source_frame_count + frame_stride - 1
        ) // frame_stride
        extraction_stats = ExtractionStats()
        pipeline_started = time.perf_counter()
        extracted_batches = extract_sampled_frame_batches(
            source,
            sampled_directory,
            frame_stride,
            ENDPOINT_BATCH_SIZE,
            extraction_stats,
        )
        try:
            first_batch = next(extracted_batches)
        except StopIteration as error:
            raise ValueError("The video has no readable frames.") from error

        first_frame = cv2.imread(str(first_batch[1][0]))
        if first_frame is None:
            raise ValueError("The first sampled frame could not be decoded.")
        height, width = first_frame.shape[:2]
        encoded_fps = (
            output_fps if output_fps is not None else max(0.1, fps / frame_stride)
        )
        encoder = start_encoder(
            output,
            width,
            height,
            encoded_fps,
        )
        emit(
            logger,
            "segmentation.encoder_started",
            output_width=width,
            output_height=height,
            output_fps=round(encoded_fps, 3),
            playback_mode="review" if output_fps is not None else "source_timing",
            batch_count=(expected_sampled_frames + ENDPOINT_BATCH_SIZE - 1)
            // ENDPOINT_BATCH_SIZE,
            endpoint_concurrency=endpoint_concurrency,
            render_concurrency=RENDER_CONCURRENCY,
        )
        matched_count = 0
        written_count = 0
        sampled_count = 0
        transcript_frames: list[AnnotatedFrame] = []
        processing_started = time.perf_counter()
        try:
            batches = enumerate(chain((first_batch,), extracted_batches))
            with ThreadPoolExecutor(
                max_workers=endpoint_concurrency
            ) as endpoint_executor, ThreadPoolExecutor(
                max_workers=RENDER_CONCURRENCY
            ) as render_executor:
                pending = deque()

                def submit_next_batch() -> bool:
                    try:
                        batch_index, (offset, paths) = next(batches)
                    except StopIteration:
                        return False
                    endpoint = endpoint_names[batch_index % len(endpoint_names)]
                    emit(
                        logger,
                        "sam3.batch_started",
                        batch_index=batch_index,
                        endpoint=endpoint,
                        first_sampled_frame=offset,
                        frame_count=len(paths),
                    )
                    pending.append(
                        (
                            batch_index,
                            len(paths),
                            endpoint_executor.submit(
                                predict_batch,
                                client,
                                endpoint,
                                prompt,
                                threshold,
                                offset,
                                paths,
                            ),
                        )
                    )
                    return True

                for _ in range(endpoint_concurrency):
                    if not submit_next_batch():
                        break

                while pending:
                    batch_index, frame_count, future = pending.popleft()
                    try:
                        batch = future.result()
                    except Exception as error:
                        emit(
                            logger,
                            "sam3.batch_failed",
                            level=logging.ERROR,
                            batch_index=batch_index,
                            frame_count=frame_count,
                            error_type=type(error).__name__,
                        )
                        raise

                    submit_next_batch()

                    render_started = time.perf_counter()
                    by_id = {
                        str(item["frame_id"]): item for item in batch.predictions
                    }
                    batch_match_count = 0
                    render_futures = [
                        render_executor.submit(
                            render_prediction_frame,
                            sequence,
                            path,
                            by_id[str(sequence)],
                            frame_stride,
                            fps,
                        )
                        for sequence, path in enumerate(
                            batch.paths,
                            start=batch.offset,
                        )
                    ]
                    rendered_batch = [future.result() for future in render_futures]
                    for rendered_frame in rendered_batch:
                        sequence = rendered_frame.sequence
                        masks = rendered_frame.masks
                        rendered = rendered_frame.image
                        if masks:
                            matched_count += 1
                            batch_match_count += 1
                            frame_index = sequence * frame_stride
                            keep_representative_frame(
                                transcript_frames,
                                AnnotatedFrame(
                                    frame_index=frame_index,
                                    timestamp_seconds=round(frame_index / fps, 3),
                                    image_b64=(
                                        rendered_frame.transcript_image_b64 or ""
                                    ),
                                ),
                                matched_count,
                            )
                        if masks or not truncate:
                            if encoder.stdin is None:
                                raise RuntimeError(
                                    "FFmpeg encoder input is unavailable."
                                )
                            encoder.stdin.write(rendered.tobytes())
                            written_count += 1
                    sampled_count += len(rendered_batch)
                    if preview_callback and rendered_batch:
                        latest = rendered_batch[-1]
                        processing_elapsed = time.perf_counter() - pipeline_started
                        estimated_remaining = (
                            processing_elapsed
                            / sampled_count
                            * max(0, expected_sampled_frames - sampled_count)
                        )
                        preview_callback(
                            {
                                "type": "preview",
                                "preview_b64": encode_preview_frame(latest.image),
                                "processed_frame_count": sampled_count,
                                "sampled_frame_count": expected_sampled_frames,
                                "matched_frame_count": matched_count,
                                "current_detection_count": len(latest.masks),
                                "elapsed_seconds": round(processing_elapsed),
                                "estimated_remaining_seconds": round(
                                    estimated_remaining
                                ),
                            }
                        )
                    for path in batch.paths:
                        path.unlink(missing_ok=True)

                    emit(
                        logger,
                        "sam3.batch_completed",
                        batch_index=batch_index,
                        endpoint=batch.endpoint_name,
                        frame_count=len(batch.paths),
                        prediction_count=len(batch.predictions),
                        matched_frames=batch_match_count,
                        request_bytes=batch.request_bytes,
                        inference_duration_ms=batch.inference_duration_ms,
                        render_duration_ms=elapsed_ms(render_started),
                    )

            timings["extract_sampled_frames"] = extraction_stats.duration_seconds
            emit(
                logger,
                "segmentation.frames_extracted",
                sampled_frames=extraction_stats.frame_count,
                duration_ms=round(extraction_stats.duration_seconds * 1000),
            )
            if written_count == 0:
                draw_label(first_frame, f'No matches for "{prompt}"')
                if encoder.stdin is None:
                    raise RuntimeError("FFmpeg encoder input is unavailable.")
                encoder.stdin.write(first_frame.tobytes())
                written_count = 1
        finally:
            if encoder.stdin:
                encoder.stdin.close()
            return_code = encoder.wait()
            extracted_batches.close()
            if return_code:
                raise RuntimeError(f"FFmpeg encoder exited with status {return_code}.")
        timings["streaming_inference_and_render"] = round(
            time.perf_counter() - processing_started,
            3,
        )
        timings["segmentation_pipeline"] = round(
            time.perf_counter() - pipeline_started,
            3,
        )

    emit(
        logger,
        "segmentation.completed",
        sampled_frames=sampled_count,
        matched_frames=matched_count,
        written_frames=written_count,
        transcript_frames=len(transcript_frames),
        output_bytes=output.stat().st_size,
        timings_seconds=timings,
    )
    return LiveSegmentationResult(
        sampled_frame_count=sampled_count,
        matched_frame_count=matched_count,
        written_frame_count=written_count,
        timings_seconds=timings,
        transcript_frames=tuple(
            sorted(transcript_frames, key=lambda frame: frame.frame_index)
        ),
    )
