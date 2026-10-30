# Databricks notebook source
# MAGIC %md
# MAGIC # Generate sample edge events
# MAGIC
# MAGIC Writes seven event JSON files to `<volume>/events/`, one per sample clip, shaped like
# MAGIC what an edge device uploads with its clip: event ID, event type, vehicle and driver,
# MAGIC on-device object detections, a speed trace and the device's severity estimate.
# MAGIC
# MAGIC **The metadata is simulated for the workshop.** The clips come from the Nexar dataset.
# MAGIC Each event also carries Nexar's own label (`nexar_label`) so reviewers can compare.
# MAGIC Trigger times follow Nexar's `time_of_event` where one exists, otherwise what is
# MAGIC visible in the clip. Run after `fetch_sample_clips`.

# COMMAND ----------

dbutils.widgets.text("volume_root", "")
volume_root = dbutils.widgets.get("volume_root").rstrip("/")
if not volume_root:
    raise ValueError("Set the volume_root parameter (the bundle passes it).")

inputs_dir = f"{volume_root}/inputs"
events_dir = f"{volume_root}/events"
nexar_dir = f"{volume_root}/datasets/nexar"

# COMMAND ----------

import csv
import json
from pathlib import Path


def speed_trace(keypoints, duration_s=40):
    """Linear interpolation of (time_s, km/h) keypoints, sampled at 1 Hz."""
    trace = []
    for second in range(duration_s + 1):
        for (t0, v0), (t1, v1) in zip(keypoints, keypoints[1:]):
            if t0 <= second <= t1:
                v = v0 + (v1 - v0) * (second - t0) / (t1 - t0)
                trace.append({"t_s": second, "speed_kmh": round(v, 1)})
                break
    return trace


def detection(label, confidence, first_seen_s, last_seen_s):
    return {
        "label": label,
        "confidence": confidence,
        "first_seen_s": first_seen_s,
        "last_seen_s": last_seen_s,
    }


EVENTS = [
    {
        "event_id": "EVT-0101",
        "clip": "nexar_negative_01041.mp4",
        "event_type": "stop_sign_violation",
        "event_label": "Rolling stop at stop sign",
        "vehicle_id": "TRK-0412",
        "driver_id": "DRV-1187",
        "recorded_at": "2026-10-21T14:32:08Z",
        "location": {"lat": 45.5231, "lon": -73.6012},
        "trigger_s": 23.5,
        "edge_severity": "medium",
        "detections": [
            detection("vehicle", 0.95, 0.0, 40.0),
            detection("stop_sign", 0.88, 16.5, 24.5),
            detection("crosswalk", 0.81, 18.0, 25.0),
        ],
        "speed": [(0, 34), (16, 33), (21, 18), (23, 11), (24, 12), (27, 24), (40, 32)],
        "suggested_prompts": ["stop sign", "crosswalk"],
    },
    {
        "event_id": "EVT-0102",
        "clip": "nexar_positive_00004.mp4",
        "event_type": "door_opening_hazard",
        "event_label": "Parked car door opened into the lane",
        "vehicle_id": "TRK-0388",
        "driver_id": "DRV-0954",
        "recorded_at": "2026-10-22T16:05:41Z",
        "location": {"lat": 45.4972, "lon": -73.5791},
        "trigger_s": 35.0,
        "edge_severity": "high",
        "detections": [
            detection("vehicle", 0.97, 0.0, 40.0),
            detection("open_door", 0.71, 34.0, 38.0),
            detection("person", 0.66, 35.0, 38.0),
        ],
        "speed": [(0, 27), (18, 25), (20, 16), (24, 24), (33, 24), (35, 12), (37, 6), (40, 8)],
        "suggested_prompts": ["open car door", "person"],
    },
    {
        "event_id": "EVT-0103",
        "clip": "nexar_positive_00007.mp4",
        "event_type": "following_too_close",
        "event_label": "Following too close, hard stop behind lead vehicle",
        "vehicle_id": "TRK-0517",
        "driver_id": "DRV-1302",
        "recorded_at": "2026-10-23T12:48:19Z",
        "location": {"lat": 45.5088, "lon": -73.5617},
        "trigger_s": 21.2,
        "edge_severity": "medium",
        "detections": [
            detection("vehicle", 0.98, 0.0, 40.0),
            detection("brake_lights", 0.77, 20.5, 23.0),
            detection("person", 0.84, 21.0, 24.0),
        ],
        "speed": [(0, 22), (17, 28), (20, 26), (21, 14), (22, 4), (25, 2), (30, 15), (40, 20)],
        "extra_telemetry": {"min_headway_s": 0.6},
        "suggested_prompts": ["car", "brake lights", "person"],
    },
    {
        "event_id": "EVT-0104",
        "clip": "nexar_positive_00003.mp4",
        "event_type": "harsh_braking",
        "event_label": "Harsh braking at night in rain",
        "vehicle_id": "TRK-0233",
        "driver_id": "DRV-0711",
        "recorded_at": "2026-10-20T22:17:55Z",
        "location": {"lat": 45.4631, "lon": -73.7442},
        "trigger_s": 19.2,
        "edge_severity": "high",
        "detections": [
            detection("vehicle", 0.92, 0.0, 40.0),
            detection("brake_lights", 0.89, 16.0, 21.0),
        ],
        "speed": [(0, 58), (16, 56), (18, 40), (19, 22), (20, 14), (24, 12), (40, 30)],
        "extra_telemetry": {"max_decel_g": 0.52},
        "suggested_prompts": ["brake lights", "car"],
    },
    {
        "event_id": "EVT-0105",
        "clip": "nexar_positive_00008.mp4",
        "event_type": "pedestrian_conflict",
        "event_label": "Pedestrian crossing in front at night intersection",
        "vehicle_id": "TRK-0412",
        "driver_id": "DRV-1187",
        "recorded_at": "2026-10-21T23:41:02Z",
        "location": {"lat": 45.5155, "lon": -73.5553},
        "trigger_s": 19.3,
        "edge_severity": "high",
        "detections": [
            detection("person", 0.69, 17.0, 19.5),
            detection("crosswalk", 0.74, 15.0, 22.0),
        ],
        "speed": [(0, 20), (15, 22), (17, 17), (19, 9), (21, 12), (40, 26)],
        "suggested_prompts": ["person", "crosswalk"],
    },
    {
        "event_id": "EVT-0106",
        "clip": "nexar_positive_00000.mp4",
        "event_type": "unsafe_cut_in",
        "event_label": "Van cut in from the left lane",
        "vehicle_id": "TRK-0599",
        "driver_id": "DRV-1450",
        "recorded_at": "2026-10-19T21:09:37Z",
        "location": {"lat": 45.5017, "lon": -73.5673},
        "trigger_s": 20.8,
        "edge_severity": "medium",
        "detections": [
            detection("van", 0.93, 0.0, 22.0),
            detection("vehicle", 0.96, 0.0, 40.0),
        ],
        "speed": [(0, 26), (18, 28), (20, 19), (21, 14), (23, 16), (40, 24)],
        "suggested_prompts": ["van", "truck"],
    },
    {
        "event_id": "EVT-0107",
        "clip": "nexar_negative_01043.mp4",
        "event_type": "harsh_acceleration",
        "event_label": "Harsh acceleration (edge trigger)",
        "vehicle_id": "TRK-0233",
        "driver_id": "DRV-0711",
        "recorded_at": "2026-10-24T10:22:14Z",
        "location": {"lat": 45.4479, "lon": -73.8068},
        "trigger_s": 12.0,
        "edge_severity": "low",
        "detections": [detection("vehicle", 0.90, 0.0, 40.0)],
        "speed": [(0, 48), (11, 50), (12, 54), (13, 53), (40, 55)],
        "suggested_prompts": ["vehicle"],
    },
]

# COMMAND ----------


def nexar_labels() -> dict[str, dict[str, str]]:
    labels = {}
    for kind in ("positive", "negative"):
        path = Path(nexar_dir) / f"train_{kind}_metadata.csv"
        if not path.is_file():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                labels[f"nexar_{kind}_{row['file_name']}"] = {**row, "kind": kind}
    return labels


def as_float(value: str | None) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


labels = nexar_labels()
Path(events_dir).mkdir(parents=True, exist_ok=True)

written = 0
for spec in EVENTS:
    if not (Path(inputs_dir) / spec["clip"]).is_file():
        print(f"Skipping {spec['event_id']}: {spec['clip']} is not in {inputs_dir}")
        continue
    label = labels.get(spec["clip"], {})
    event = {
        "event_id": spec["event_id"],
        "simulated": True,
        "source_note": (
            "Simulated edge metadata for the workshop. Clip from the Nexar Dashcam "
            "Collision Prediction dataset (Nexar Open Data License)."
        ),
        "clip": spec["clip"],
        "device_id": f"EDGE-{spec['vehicle_id'][-4:]}",
        "vehicle_id": spec["vehicle_id"],
        "driver_id": spec["driver_id"],
        "recorded_at": spec["recorded_at"],
        "location": spec["location"],
        "event_type": spec["event_type"],
        "event_label": spec["event_label"],
        "trigger": {"time_s": spec["trigger_s"], "clip_duration_s": 40},
        "edge_detections": spec["detections"],
        "edge_severity": spec["edge_severity"],
        "telemetry": {
            "speed_kmh": speed_trace(spec["speed"]),
            **spec.get("extra_telemetry", {}),
        },
        "suggested_prompts": spec["suggested_prompts"],
        "nexar_label": {
            "collision_or_near_miss": label.get("kind") == "positive" if label else None,
            "time_of_event_s": as_float(label.get("time_of_event")),
            "time_of_alert_s": as_float(label.get("time_of_alert")),
            "scene": label.get("scene"),
            "weather": label.get("weather"),
            "light_conditions": label.get("light_conditions"),
        },
    }
    Path(events_dir, f"{spec['event_id']}.json").write_text(json.dumps(event, indent=2))
    written += 1
    print(f"Wrote {spec['event_id']} ({spec['event_type']}) for {spec['clip']}")

print(f"Wrote {written} events to {events_dir}")
