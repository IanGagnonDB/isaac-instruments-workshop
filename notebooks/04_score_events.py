# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Rolling-stop scoring
# MAGIC
# MAGIC > **Demonstration rule, not a validated severity model.**
# MAGIC
# MAGIC * `sign_visible`: any frame of the clip whose top-1 label is `sign_class` with score above
# MAGIC   `score_threshold`. **ImageNet has no stop-sign class, so this is a proxy**: 920 "traffic light"
# MAGIC   (common at intersections) by default. 919 "street sign" is the closer class, but on D2-City
# MAGIC   footage it is almost never top-1, so every event scores `none`. With `sign_source = finetuned`, the Step 8
# MAGIC   `stop_sign_visible` model is used instead and `sign_class` is ignored.
# MAGIC * `min_speed_kmh`: minimum of the 2 Hz speed trace within `window_s` of the frames where the sign is
# MAGIC   visible. For clips without a visible sign it is the whole-clip minimum, for context only.
# MAGIC * Severity: `none` if no sign, `ok` < 2 km/h, `low` < 8, `medium` < 15, `high` otherwise.
# MAGIC
# MAGIC The scoring job runs this on **serverless** (no cluster start-up). The default path only joins
# MAGIC existing benchmark predictions in SQL. The fallback paths (no predictions yet, or
# MAGIC `sign_source = finetuned`) run a UDF that loads the model from `/Volumes`; that is guaranteed on
# MAGIC dedicated classic clusters, so run the notebook there if serverless rejects it.
# MAGIC
# MAGIC The next cell installs the pinned extras: serverless has no cluster libraries; on classic they
# MAGIC are already installed and pip leaves them as they are.

# COMMAND ----------
# MAGIC %pip install -q onnx==1.21.0 onnxruntime==1.31.0 opencv-python-headless==4.11.0.86

# COMMAND ----------
import os
import sys

# Import video_cls from this Git folder; executors receive it by value (no wheel or install).
sys.path.insert(0, os.path.abspath("../src"))
import video_cls
from pyspark import cloudpickle
cloudpickle.register_pickle_by_value(video_cls)

from video_cls import TRAFFIC_LIGHT_CLASS

PARAMS = [("catalog", ""), ("schema", ""), ("volume", "video"), ("model", "resnet18"),
          ("variant", "fp32"), ("sign_source", "proxy"), ("sign_class", str(TRAFFIC_LIGHT_CLASS)),
          ("score_threshold", "0.3"), ("window_s", "3"), ("ok_below_kmh", "2"), ("low_below_kmh", "8"), ("medium_below_kmh", "15")]
for name, default in PARAMS:
    dbutils.widgets.text(name, default)
p = {name: dbutils.widgets.get(name) for name, _ in PARAMS}

catalog, schema = p["catalog"], p["schema"]
video_cls.check_location(catalog, schema)
models_dir = f"/Volumes/{catalog}/{schema}/{p['volume']}/models"
if p["sign_source"] not in ("proxy", "finetuned"):
    raise ValueError("sign_source must be proxy or finetuned")
thresholds = {"ok": float(p["ok_below_kmh"]), "low": float(p["low_below_kmh"]),
              "medium": float(p["medium_below_kmh"])}

# COMMAND ----------
# MAGIC %md
# MAGIC ## Frame predictions to score
# MAGIC Reuses a `frame_predictions_<label>` table from 03 when it has the chosen model/variant. Every
# MAGIC benchmark cluster scores the same frames, so the first one found is used. Otherwise classifies a
# MAGIC `frames_<label>` table here, extracting frames for all events first if 03 has not run.
# MAGIC
# MAGIC Only events with frame predictions are scored. The benchmark uses the first `max_clips` clips,
# MAGIC so a benchmark-fed run scores that subset; coverage is printed below.

# COMMAND ----------
import functools

from pyspark.sql import functions as F

from video_cls.frames import FRAME_SCHEMA, extract_frames
from video_cls.infer import make_classifier
from video_cls.models import artifact_path
from video_cls.tables import find_table

frames_table = find_table(spark, catalog, schema, "frames")
predictions_table = find_table(spark, catalog, schema, "frame_predictions")
try:
    n_partitions = 3 * spark.sparkContext.defaultParallelism
except Exception:                                         # serverless: no sparkContext
    n_partitions = 64


def ensure_frames():
    global frames_table
    if frames_table is None:
        frames_table = f"{catalog}.{schema}.frames_scoring"
        (spark.table(f"{catalog}.{schema}.events").select("clip_id", "clip_path")
         .repartition(n_partitions)
         .mapInPandas(functools.partial(extract_frames, n_frames=10), FRAME_SCHEMA)
         .write.mode("overwrite").saveAsTable(frames_table))
    return spark.table(frames_table)


def classify_frames(model_path):
    return (ensure_frames().repartition(n_partitions)
            .withColumn("pred", make_classifier(model_path)("jpeg"))
            .select("clip_id", "frame_idx", "ts_s", "pred.*"))


if p["sign_source"] == "finetuned":
    model_name = "stopsign_resnet18_" + p["variant"]
    model_path = artifact_path(models_dir, "stopsign_resnet18", p["variant"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} missing: run 05_finetune_optional first, or use sign_source=proxy")
    preds = classify_frames(model_path)
    sign_label = 1                                        # class index 1 = stop_sign_visible "yes"
else:
    model_name = f"{p['model']}_{p['variant']}"
    preds = None
    if predictions_table:
        preds = spark.table(predictions_table).where(
            (F.col("model") == p["model"]) & (F.col("variant") == p["variant"]))
        if preds.isEmpty():
            preds = None
    if preds is None:
        preds = classify_frames(artifact_path(models_dir, p["model"], p["variant"]))
    sign_label = int(p["sign_class"])
    model_name += f"_class{sign_label}"                  # record which ImageNet proxy was used

preds.select("clip_id", "ts_s", "label", "score").createOrReplaceTempView("scoring_predictions")
# Score only events whose clip has frame predictions (the benchmark may cover a subset of clips).
spark.sql(f"""SELECT * FROM {catalog}.{schema}.events
              WHERE clip_id IN (SELECT clip_id FROM scoring_predictions)""").createOrReplaceTempView("scoring_events")
print(f"Scoring with {model_name} (sign label {sign_label}, threshold {p['score_threshold']}); "
      f"predictions from {predictions_table if p['sign_source'] == 'proxy' and predictions_table else 'this run'}")

# COMMAND ----------
from video_cls.scoring import scoring_sql

scores = spark.sql(scoring_sql(
    events="scoring_events", predictions="scoring_predictions",
    sign_label=sign_label, score_threshold=float(p["score_threshold"]), window_s=float(p["window_s"]),
    speed_hz=2, thresholds=thresholds, model_name=model_name))
scores.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.event_scores")

event_scores = spark.table(f"{catalog}.{schema}.event_scores")
n_events = spark.table(f"{catalog}.{schema}.events").count()
print(f"event_scores: {event_scores.count()} of {n_events} events scored (events with frame predictions)")
display(event_scores.groupBy("severity").count().orderBy("severity"))

# COMMAND ----------
# MAGIC %md
# MAGIC Severity vs the synthetic speed profile written by 00_setup. Sign visibility comes from the image
# MAGIC model and the speed trace is synthetic, so the two are independent by construction.

# COMMAND ----------
display(event_scores.join(spark.table(f"{catalog}.{schema}.events").select("clip_id", "speed_profile"), "clip_id")
        .groupBy("speed_profile").pivot("severity").count().orderBy("speed_profile"))
