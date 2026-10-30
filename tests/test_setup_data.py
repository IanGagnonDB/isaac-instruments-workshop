import json

import numpy as np

from conftest import write_clip
from video_cls.setup_data import (SPEED_HZ, find_sources, make_event, replicate_clips, speed_trace,
                                  write_events)


def test_replicate_round_robin_and_idempotent(tmp_path):
    src = tmp_path / "src" / "1003"
    src.mkdir(parents=True)
    a, b = write_clip(src / "a.mp4"), write_clip(src / "b.mp4", seconds=2)
    sources = find_sources(tmp_path / "src")
    clips = replicate_clips(sources, tmp_path / "clips", 5)
    assert [c["clip_id"] for c in clips] == [f"clip_{i:06d}" for i in range(1, 6)]
    assert [c["source_file"] for c in clips] == ["a.mp4", "b.mp4", "a.mp4", "b.mp4", "a.mp4"]
    assert (tmp_path / "clips" / "clip_000002.mp4").read_bytes() == open(b, "rb").read()
    mtime = (tmp_path / "clips" / "clip_000001.mp4").stat().st_mtime_ns
    replicate_clips(sources, tmp_path / "clips", 5)
    assert (tmp_path / "clips" / "clip_000001.mp4").stat().st_mtime_ns == mtime   # not recopied


def test_speed_trace_profiles():
    rng = np.random.default_rng(1)
    for profile, (lo, hi) in {"full_stop": (0, 0), "rolling": (4, 16), "no_stop": (19, 29)}.items():
        v = speed_trace(30.0, profile, rng)
        assert len(v) == 30 * SPEED_HZ
        assert lo <= min(v) <= hi, (profile, min(v))
        assert max(v) >= 29


def test_events_are_deterministic_and_complete(tmp_path):
    clip = {"clip_id": "clip_000007", "clip_path": "/Volumes/c/s/v/clips/clip_000007.mp4", "source_file": "a.mp4"}
    first, second = make_event(clip, 29.3, 6), make_event(clip, 29.3, 6)
    assert first == second
    assert set(first) >= {"event_id", "clip_id", "clip_path", "vehicle_no", "driver_no", "event_type",
                          "start_ts", "speed_kmh"}
    assert first["event_type"] == "rolling_stop_candidate"
    assert len(first["speed_kmh"]) == 59                # ceil(29.3 s x 2 Hz)


def test_profile_mix_is_realistic(tmp_path):
    clips = [{"clip_id": f"clip_{i:06d}", "clip_path": "x", "source_file": "a.mp4"} for i in range(400)]
    mix = write_events(clips, {"a.mp4": 30.0}, tmp_path / "json")
    assert len(list((tmp_path / "json").glob("*.json"))) == 400
    assert all(80 <= n <= 200 for n in mix.values()), mix
    event = json.loads((tmp_path / "json" / "clip_000000.json").read_text())
    assert event["clip_id"] == "clip_000000"
