# Databricks notebook source
# MAGIC %md
# MAGIC # Load edge events into Delta
# MAGIC
# MAGIC Creates the two review tables if needed, loads every event JSON in `<volume>/events/`
# MAGIC into `edge_events`, and grants the app's service principal access.
# MAGIC
# MAGIC - `edge_events`: one row per edge event, with its review status. New events are
# MAGIC   inserted as `pending`; existing rows (and their review status) are never overwritten.
# MAGIC - `event_reviews`: append-only history of every grade the app submits.
# MAGIC
# MAGIC Safe to rerun. The `ingest_events` job runs this notebook when new files land.

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("volume_root", "")
dbutils.widgets.text("app_name", "")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume_root = dbutils.widgets.get("volume_root").rstrip("/")
app_name = dbutils.widgets.get("app_name").strip()
if not catalog or not schema or not volume_root:
    raise ValueError("Set catalog, schema and volume_root (the bundle passes them).")

events_table = f"`{catalog}`.`{schema}`.edge_events"
reviews_table = f"`{catalog}`.`{schema}`.event_reviews"
events_dir = f"{volume_root}/events"

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {events_table} (
  event_id STRING NOT NULL,
  clip STRING,
  event_type STRING,
  event_label STRING,
  vehicle_id STRING,
  driver_id STRING,
  device_id STRING,
  recorded_at TIMESTAMP,
  edge_severity STRING,
  trigger_s DOUBLE,
  simulated BOOLEAN,
  payload STRING COMMENT 'Full edge event JSON as uploaded',
  ingested_at TIMESTAMP,
  review_status STRING COMMENT 'pending or reviewed',
  reviewed_severity STRING,
  edge_classification_correct BOOLEAN,
  corrected_event_type STRING,
  reviewed_by STRING,
  reviewed_at TIMESTAMP
) COMMENT 'Edge events awaiting or after cloud review'
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {reviews_table} (
  review_id STRING NOT NULL,
  event_id STRING NOT NULL,
  edge_event_type STRING,
  edge_severity STRING,
  severity STRING,
  edge_classification_correct BOOLEAN,
  corrected_event_type STRING,
  notes STRING,
  sam3_prompt STRING,
  sam3_matched_frames INT,
  sam3_sampled_frames INT,
  reviewer STRING,
  submitted_at TIMESTAMP
) COMMENT 'Append-only history of reviewer grades'
""")

# COMMAND ----------

import json
from datetime import datetime, timezone
from pathlib import Path

rows = []
for path in sorted(Path(events_dir).glob("*.json")):
    try:
        event = json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"Skipping {path.name}: not valid JSON")
        continue
    if not event.get("event_id") or not event.get("clip"):
        print(f"Skipping {path.name}: event_id and clip are required")
        continue
    rows.append(
        {
            "event_id": str(event["event_id"]),
            "clip": str(event["clip"]),
            "event_type": event.get("event_type"),
            "event_label": event.get("event_label"),
            "vehicle_id": event.get("vehicle_id"),
            "driver_id": event.get("driver_id"),
            "device_id": event.get("device_id"),
            "recorded_at": event.get("recorded_at"),
            "edge_severity": event.get("edge_severity"),
            "trigger_s": float((event.get("trigger") or {}).get("time_s") or 0) or None,
            "simulated": bool(event.get("simulated", False)),
            "payload": json.dumps(event),
        }
    )

print(f"Read {len(rows)} event files from {events_dir}")

if rows:
    schema_ddl = (
        "event_id STRING, clip STRING, event_type STRING, event_label STRING, "
        "vehicle_id STRING, driver_id STRING, device_id STRING, recorded_at STRING, "
        "edge_severity STRING, trigger_s DOUBLE, simulated BOOLEAN, payload STRING"
    )
    spark.createDataFrame(rows, schema_ddl).createOrReplaceTempView("incoming_events")
    result = spark.sql(f"""
        MERGE INTO {events_table} AS target
        USING incoming_events AS source
        ON target.event_id = source.event_id
        WHEN NOT MATCHED THEN INSERT (
          event_id, clip, event_type, event_label, vehicle_id, driver_id, device_id,
          recorded_at, edge_severity, trigger_s, simulated, payload, ingested_at, review_status
        ) VALUES (
          source.event_id, source.clip, source.event_type, source.event_label,
          source.vehicle_id, source.driver_id, source.device_id,
          to_timestamp(source.recorded_at), source.edge_severity, source.trigger_s,
          source.simulated, source.payload, current_timestamp(), 'pending'
        )
    """)
    display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC Grant the app's service principal read/write on both tables. The app reads the queue
# MAGIC and inserts grades through the bundle's SQL warehouse.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

if app_name:
    principal = WorkspaceClient().apps.get(app_name).service_principal_client_id
    for statement in (
        f"GRANT USE CATALOG ON CATALOG `{catalog}` TO `{principal}`",
        f"GRANT USE SCHEMA ON SCHEMA `{catalog}`.`{schema}` TO `{principal}`",
        f"GRANT SELECT, MODIFY ON TABLE {events_table} TO `{principal}`",
        f"GRANT SELECT, MODIFY ON TABLE {reviews_table} TO `{principal}`",
    ):
        try:
            spark.sql(statement)
            print(f"OK: {statement}")
        except Exception as error:  # USE CATALOG may need a catalog owner; the app may already have it.
            print(f"Skipped ({type(error).__name__}): {statement}")
else:
    print("app_name not set; skipped grants.")

print(spark.table(events_table).groupBy("review_status").count().collect())
