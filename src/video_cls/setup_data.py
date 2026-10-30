"""Sample data: replicate source clips to N clip_ids and write one synthetic event JSON per clip."""
import concurrent.futures
import hashlib
import json
import math
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

SPEED_HZ = 2
EVENT_TYPE = "rolling_stop_candidate"
# Share of clips per speed profile: full stop, rolling stop (5-15 km/h), no real slowdown.
PROFILE_MIX = (("full_stop", 0.35), ("rolling", 0.40), ("no_stop", 0.25))
EVENT_NAMESPACE = uuid.UUID("6f1c9a52-3f0e-4f43-9a55-2a4f1d1f7c10")


def find_sources(folder):
    """All MP4s under folder, sorted by relative path so replication is deterministic."""
    folder = Path(folder)
    sources = sorted((p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4"),
                     key=lambda p: p.relative_to(folder).as_posix())
    if not sources:
        raise FileNotFoundError(f"No .mp4 files under {folder}")
    return sources


def replicate_clips(sources, clips_dir, n_clips, workers=32):
    """Copy sources round-robin to clips_dir/clip_000001.mp4 ... Byte copies, no re-encoding.

    Existing files with the right size are kept, so reruns are cheap. Returns a list of
    {clip_id, clip_path, source_file} dicts in clip_id order.
    """
    if n_clips < 1:
        raise ValueError("n_clips must be positive")
    clips_dir = Path(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    plan = [(f"clip_{i + 1:06d}", sources[i % len(sources)]) for i in range(n_clips)]

    def copy_one(item):
        clip_id, source = item
        dest = clips_dir / f"{clip_id}.mp4"
        if not (dest.is_file() and dest.stat().st_size == source.stat().st_size):
            shutil.copyfile(source, dest)
        return {"clip_id": clip_id, "clip_path": str(dest), "source_file": source.name}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(copy_one, plan))


def _rng(clip_id):
    return np.random.default_rng(int(hashlib.sha256(clip_id.encode()).hexdigest()[:16], 16))


def pick_profile(rng):
    names, weights = zip(*PROFILE_MIX)
    return names[rng.choice(len(names), p=np.array(weights) / sum(weights))]


def speed_trace(duration_s, profile, rng):
    """km/h at SPEED_HZ: cruise, decelerate to a minimum, optionally hold, accelerate back."""
    n = max(1, math.ceil(duration_s * SPEED_HZ))
    t = np.arange(n) / SPEED_HZ
    cruise = rng.uniform(30, 50)
    v_min = {"full_stop": 0.0, "rolling": rng.uniform(5, 15), "no_stop": rng.uniform(20, 28)}[profile]
    t_brake = rng.uniform(0.15, 0.45) * duration_s
    t_low = t_brake + rng.uniform(3, 6)
    t_go = t_low + (rng.uniform(1, 3) if profile == "full_stop" else rng.uniform(0, 1))
    t_back = t_go + rng.uniform(5, 8)
    v = np.interp(t, [0, t_brake, t_low, t_go, t_back, max(t_back, duration_s) + 1],
                  [cruise, cruise, v_min, v_min, cruise, cruise])
    v = v + rng.normal(0, 0.4, n) * (v > 1)              # sensor noise while moving
    if profile == "full_stop":
        v[(t >= t_low) & (t <= t_go)] = 0.0
    return np.round(np.clip(v, 0, None), 1).tolist()


def make_event(clip, duration_s, index, base_ts=datetime(2026, 1, 5, 7, 0, tzinfo=timezone.utc)):
    rng = _rng(clip["clip_id"])
    profile = pick_profile(rng)
    start = base_ts + timedelta(minutes=7 * index, seconds=int(rng.integers(0, 300)))
    return {
        "event_id": str(uuid.uuid5(EVENT_NAMESPACE, clip["clip_id"])),
        "clip_id": clip["clip_id"],
        "clip_path": clip["clip_path"],
        "vehicle_no": f"V-{int(rng.integers(1, 51)):04d}",
        "driver_no": f"D-{int(rng.integers(1, 121)):04d}",
        "event_type": EVENT_TYPE,
        "start_ts": start.isoformat(),
        "speed_profile": profile,                        # synthetic ground truth, for inspection only
        "speed_kmh": speed_trace(duration_s, profile, rng),
    }


def write_events(clips, durations, json_dir, workers=32):
    """Write json_dir/<clip_id>.json for each clip. durations maps source_file -> seconds."""
    json_dir = Path(json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)

    def write_one(item):
        index, clip = item
        event = make_event(clip, durations[clip["source_file"]], index)
        (json_dir / f"{clip['clip_id']}.json").write_text(json.dumps(event))
        return event["speed_profile"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        profiles = list(pool.map(write_one, enumerate(clips)))
    return {name: profiles.count(name) for name, _ in PROFILE_MIX}
