# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup: schema, volume, sample clips, synthetic telemetry, `events` table
# MAGIC
# MAGIC 1. Create the schema and volume if missing (the catalog must already exist).
# MAGIC 2. Source clips: copy from `source_clips_path` (for example an existing D2-City `raw/` folder),
# MAGIC    or download D2-City into the volume when that parameter is empty.
# MAGIC 3. Replicate the sources to `n_clips` clip_ids under `clips/`. Byte copies, no re-encoding.
# MAGIC 4. Write one synthetic event JSON per clip under `json/`: speed at 2 Hz for the clip duration,
# MAGIC    with a mix of full stops, rolling stops (5 to 15 km/h) and no real slowdown.
# MAGIC 5. Load the JSON into the `events` Delta table with Auto Loader (`availableNow`).
# MAGIC
# MAGIC To use your own clips later, point `source_clips_path` at their folder, or write your own
# MAGIC event JSON (same fields) into `json/` and skip replication with `n_clips = 0`.

# COMMAND ----------
import os
import sys

try:
    import video_cls                      # installed as a wheel by the bundle jobs
except ImportError:                       # interactive run from the bundle folder
    sys.path.insert(0, os.path.abspath("../src"))
    import video_cls
    from pyspark import cloudpickle
    cloudpickle.register_pickle_by_value(video_cls)

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("volume", "video")
dbutils.widgets.text("source_clips_path", "")
dbutils.widgets.text("n_clips", "1000")
dbutils.widgets.dropdown("reset", "false", ["false", "true"])

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
source_clips_path = dbutils.widgets.get("source_clips_path").strip()
n_clips = int(dbutils.widgets.get("n_clips"))
reset = dbutils.widgets.get("reset") == "true"
if not catalog or not schema:
    raise ValueError("Set the catalog and schema parameters (bundle variables).")

root = f"/Volumes/{catalog}/{schema}/{volume}"
events_table = f"{catalog}.{schema}.events"
checkpoint = f"{root}/_checkpoints/events"
print(f"Volume root: {root}\nEvents table: {events_table}")

# COMMAND ----------
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
spark.sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{schema}`.`{volume}`")

if reset:
    spark.sql(f"DROP TABLE IF EXISTS {events_table}")
    for folder in ("json", "_checkpoints/events"):
        dbutils.fs.rm(f"{root}/{folder}", recurse=True)
    print("Reset: dropped events table, JSON files and Auto Loader checkpoint (clips are kept).")

for folder in ("clips", "json", "models", "labels/yes", "labels/no"):
    os.makedirs(f"{root}/{folder}", exist_ok=True)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Source clips
# MAGIC D2-City (public dashcam footage, ~30 s 1080p clips) is the default. If `source_clips_path` is empty
# MAGIC the downloader stages it once under `d2_city/raw/` and later runs reuse it.

# COMMAND ----------
from video_cls import d2_city
from video_cls.setup_data import find_sources

if not source_clips_path:
    d2_root = f"{root}/d2_city"
    raw = f"{d2_root}/raw"
    has_raw = os.path.isdir(raw) and any(f.endswith(".mp4") for _, _, fs in os.walk(raw) for f in fs)
    if not has_raw:
        print("Downloading D2-City (one ZIP, a few minutes in Canada Central)...")
        d2_city.run_workshop(root=d2_root, copies=1)
    source_clips_path = raw

sources = find_sources(source_clips_path)
print(f"{len(sources)} source clips under {source_clips_path}")

# COMMAND ----------
import concurrent.futures
import time

from video_cls.frames import clip_duration_s
from video_cls.setup_data import replicate_clips, write_events

with concurrent.futures.ThreadPoolExecutor(32) as pool:
    durations = dict(zip((s.name for s in sources), pool.map(lambda s: clip_duration_s(str(s)), sources)))
unreadable = [name for name, d in durations.items() if d <= 0]
if unreadable:
    raise RuntimeError(f"OpenCV could not read {len(unreadable)} source clips, e.g. {unreadable[:3]}")
print(f"Source durations: {min(durations.values()):.1f}-{max(durations.values()):.1f} s")

if n_clips > 0:
    started = time.monotonic()
    clips = replicate_clips(sources, f"{root}/clips", n_clips)
    print(f"{len(clips)} clips in {root}/clips ({time.monotonic() - started:.0f} s)")
    mix = write_events(clips, durations, f"{root}/json")
    print(f"Wrote {len(clips)} event JSON files. Speed profile mix: {mix}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## `events` table with Auto Loader
# MAGIC `trigger(availableNow=True)` processes every new JSON file once and stops. Reruns only pick up
# MAGIC new files; use `reset = true` after changing `n_clips`.

# COMMAND ----------
EVENT_SCHEMA = """event_id string, clip_id string, clip_path string, vehicle_no string, driver_no string,
event_type string, start_ts timestamp, speed_profile string, speed_kmh array<double>"""

query = (spark.readStream.format("cloudFiles")
         .option("cloudFiles.format", "json")
         .option("multiLine", "true")
         .schema(EVENT_SCHEMA)
         .load(f"{root}/json")
         .writeStream
         .option("checkpointLocation", checkpoint)
         .trigger(availableNow=True)
         .toTable(events_table))
query.awaitTermination()

summary = spark.sql(f"""
  SELECT COUNT(*) AS events, COUNT(DISTINCT clip_id) AS clips,
         ROUND(AVG(size(speed_kmh)) / 2, 1) AS avg_duration_s,
         SUM(CASE WHEN array_min(speed_kmh) = 0 THEN 1 ELSE 0 END) AS full_stops,
         SUM(CASE WHEN array_min(speed_kmh) BETWEEN 5 AND 15 THEN 1 ELSE 0 END) AS rolling_5_15
  FROM {events_table}""")
display(summary)
