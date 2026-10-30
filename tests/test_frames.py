import cv2
import numpy as np
import pandas as pd

from conftest import FPS, SECONDS
from video_cls.frames import COLUMNS, clip_duration_s, extract_clip, extract_frames


def _decode(jpeg):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def test_one_row_per_sampled_frame(clip):
    rows = extract_clip("c1", clip, n_frames=10)
    assert [r[1] for r in rows] == list(range(10))
    assert all(r[0] == "c1" for r in rows)
    np.testing.assert_allclose([r[2] for r in rows], np.linspace(0, SECONDS, 10, endpoint=False))


def test_seek_lands_on_the_requested_timestamp(clip):
    # Frame n is grey 8 * n, so mean brightness identifies which frame the seek returned.
    for _, _, ts, jpeg in extract_clip("c1", clip, n_frames=5):
        frame_no = _decode(jpeg).mean() / 8
        assert abs(frame_no - ts * FPS) <= 1.5, (ts, frame_no)


def test_frames_are_downscaled_before_encoding(clip):
    rows = extract_clip("c1", clip, n_frames=2, max_width=160)
    assert _decode(rows[0][3]).shape[:2] == (120, 160)
    rows = extract_clip("c1", clip, n_frames=2, max_width=None)
    assert _decode(rows[0][3]).shape[:2] == (240, 320)


def test_extract_frames_mapinpandas_contract(clip, tmp_path):
    batches = [pd.DataFrame({"clip_id": ["a", "b"], "clip_path": [clip, clip]}),
               pd.DataFrame({"clip_id": ["missing"], "clip_path": [str(tmp_path / "nope.mp4")]})]
    out = list(extract_frames(iter(batches), n_frames=4))
    assert len(out) == 2                                 # one output frame per input batch
    assert list(out[0].columns) == COLUMNS
    assert out[0].groupby("clip_id").size().to_dict() == {"a": 4, "b": 4}
    assert out[1].empty and list(out[1].columns) == COLUMNS   # unreadable clip: no rows, no crash


def test_clip_duration(clip, tmp_path):
    assert abs(clip_duration_s(clip) - SECONDS) < 0.2
    assert clip_duration_s(str(tmp_path / "nope.mp4")) == 0.0
