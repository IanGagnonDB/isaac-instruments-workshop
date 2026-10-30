import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import cv2

import live_segment


def make_test_video(path: Path, *, frames: int = 31, fps: int = 10) -> None:
    subprocess.run(
        [
            live_segment.ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=96x64:rate={fps}",
            "-frames:v",
            str(frames),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def test_streaming_extraction_matches_blocking_extraction(tmp_path):
    source = tmp_path / "source.mp4"
    make_test_video(source)
    blocking_directory = tmp_path / "blocking"
    streaming_directory = tmp_path / "streaming"

    blocking = live_segment.extract_sampled_frames(
        source,
        blocking_directory,
        frame_stride=3,
    )
    stats = live_segment.ExtractionStats()
    batches = list(
        live_segment.extract_sampled_frame_batches(
            source,
            streaming_directory,
            frame_stride=3,
            batch_size=4,
            stats=stats,
        )
    )
    streaming = [path for _, paths in batches for path in paths]

    assert [path.name for path in streaming] == [path.name for path in blocking]
    assert [path.read_bytes() for path in streaming] == [
        path.read_bytes() for path in blocking
    ]
    assert [offset for offset, _ in batches] == [0, 4, 8]
    assert stats.frame_count == 11
    assert stats.duration_seconds > 0


def test_segment_video_preserves_order_and_reports_batch_progress(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    make_test_video(source, frames=31)
    preview_events = []

    def fake_query(*, name, dataframe_records):
        assert name == "sam3"
        return SimpleNamespace(
            predictions=[
                {
                    "frame_id": record["frame_id"],
                    "masks_json": "[]",
                    "scores_json": "[]",
                }
                for record in dataframe_records
            ]
        )

    client = SimpleNamespace(
        serving_endpoints=SimpleNamespace(query=fake_query),
    )
    monkeypatch.setattr(live_segment, "ENDPOINT_BATCH_SIZE", 4)

    result = live_segment.segment_video(
        client,
        "sam3",
        source,
        output,
        prompt="person",
        frame_stride=3,
        threshold=0.5,
        truncate=False,
        preview_callback=preview_events.append,
    )

    capture = cv2.VideoCapture(str(output))
    try:
        output_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    assert result.sampled_frame_count == 11
    assert result.written_frame_count == 11
    assert output_frames == 11
    assert [event["processed_frame_count"] for event in preview_events] == [4, 8, 11]
    assert all(event["sampled_frame_count"] == 11 for event in preview_events)
    assert result.timings_seconds["segmentation_pipeline"] > 0


def test_segment_video_shards_batches_and_stitches_in_source_order(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    make_test_video(source, frames=31)
    batch_endpoints = {}

    def fake_query(*, name, dataframe_records):
        first_frame_id = int(dataframe_records[0]["frame_id"])
        batch_endpoints[first_frame_id] = name
        # Force completion out of order; writes must still remain source ordered.
        time.sleep({"sam-1": 0.03, "sam-2": 0.02, "sam-3": 0.01}[name])
        return SimpleNamespace(
            predictions=[
                {
                    "frame_id": record["frame_id"],
                    "masks_json": "[]",
                    "scores_json": "[]",
                }
                for record in dataframe_records
            ]
        )

    client = SimpleNamespace(
        serving_endpoints=SimpleNamespace(query=fake_query),
    )
    monkeypatch.setattr(live_segment, "ENDPOINT_BATCH_SIZE", 4)

    result = live_segment.segment_video(
        client,
        ("sam-1", "sam-2", "sam-3"),
        source,
        output,
        prompt="person",
        frame_stride=3,
        threshold=0.5,
        truncate=False,
    )

    capture = cv2.VideoCapture(str(output))
    try:
        output_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    assert batch_endpoints == {0: "sam-1", 4: "sam-2", 8: "sam-3"}
    assert result.sampled_frame_count == 11
    assert result.written_frame_count == 11
    assert output_frames == 11
