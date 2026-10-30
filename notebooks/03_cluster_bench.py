# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Cluster pipeline and benchmark
# MAGIC
# MAGIC ```
# MAGIC events -> repartition(N) -> mapInPandas(extract_frames) -> frames_<label>        [timed: extract]
# MAGIC frames_<label> -> repartition(N) -> pandas UDF classify (model loaded once per task)
# MAGIC        -> frame_predictions_<label> (one slice per model/variant)               [timed: inference]
# MAGIC ```
# MAGIC Frames are written to a Delta table between the stages so extract and inference are timed
# MAGIC separately, and extraction runs once per cluster for all `runs`. One row per run is appended to
# MAGIC `bench_cluster`. Tables are suffixed with `run_label` so the benchmark clusters can run in
# MAGIC parallel without overwriting each other; `bench_cluster` only receives appends.
# MAGIC
# MAGIC Parallelism rules applied here: repartition the input and again after extraction, one
# MAGIC ONNX/PyTorch thread per Spark task (`ort_threads` = `spark.task.cpus`), Arrow batch size set
# MAGIC explicitly, seek instead of full decode, Photon off (job cluster setting).

# COMMAND ----------
import os
import sys

# Import video_cls from this Git folder; executors receive it by value (no wheel or install).
sys.path.insert(0, os.path.abspath("../src"))
import video_cls
from pyspark import cloudpickle
cloudpickle.register_pickle_by_value(video_cls)

PARAMS = [("catalog", ""), ("schema", ""), ("volume", "video"), ("num_workers", "1"),
          ("runs", "resnet18:int8"), ("n_partitions", "0"), ("batch_size", "64"),
          ("ort_threads", "1"), ("n_frames", "10"), ("max_clips", "0"), ("run_label", "")]
for name, default in PARAMS:
    dbutils.widgets.text(name, default)
p = {name: dbutils.widgets.get(name) for name, _ in PARAMS}

catalog, schema = p["catalog"], p["schema"]
video_cls.check_location(catalog, schema)
models_dir = f"/Volumes/{catalog}/{schema}/{p['volume']}/models"
num_workers = int(p["num_workers"])
batch_size = int(p["batch_size"])
ort_threads = int(p["ort_threads"])
n_frames = int(p["n_frames"])
max_clips = int(p["max_clips"])

# COMMAND ----------
from video_cls.bench import cluster_shape, parse_runs, wait_for_workers
from video_cls.tables import BENCH_CLUSTER_SCHEMA, ensure_table, label_suffix

runs = parse_runs(p["runs"])
suffix = label_suffix(p["run_label"])
registered = wait_for_workers(spark, num_workers)
shape = cluster_shape(spark)
n_partitions = int(p["n_partitions"]) or 3 * shape["task_slots"]

# Images per forward pass. The default (10,000) can run executors out of memory with image rows.
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", str(batch_size))
if ort_threads != shape["task_cpus"]:
    print(f"WARNING: ort_threads={ort_threads} but spark.task.cpus={shape['task_cpus']}: cores will be "
          "over- or under-subscribed.")
print({"workers": registered, **shape, "n_partitions": n_partitions, "batch_size": batch_size,
       "ort_threads": ort_threads, "runs": runs})

# COMMAND ----------
# MAGIC %md
# MAGIC ## Stage 1: frame extraction

# COMMAND ----------
import functools

from video_cls.bench import Stopwatch
from video_cls.frames import FRAME_SCHEMA, extract_frames

frames_table = f"{catalog}.{schema}.frames_{suffix}"
clips = spark.table(f"{catalog}.{schema}.events").select("clip_id", "clip_path")
if max_clips:
    clips = clips.orderBy("clip_id").limit(max_clips)
n_clips = clips.count()

with Stopwatch() as extract:
    (clips.repartition(n_partitions)                                  # spread clips across every core
          .mapInPandas(functools.partial(extract_frames, n_frames=n_frames), FRAME_SCHEMA)
          .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(frames_table))
n_frames_total = spark.table(frames_table).count()
print(f"Extracted {n_frames_total} frames from {n_clips} clips in {extract.seconds:.1f} s "
      f"({n_frames_total / extract.seconds:.1f} frames/s)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Stage 2: inference, one pass per model/variant

# COMMAND ----------
from datetime import datetime, timezone

from pyspark.sql import functions as F

from video_cls.infer import make_classifier
from video_cls.models import artifact_path

predictions_table = f"{catalog}.{schema}.frame_predictions_{suffix}"
run_ts = datetime.now(timezone.utc)
rows = []
for model, variant in runs:
    classify = make_classifier(artifact_path(models_dir, model, variant), threads=ort_threads)
    scored = (spark.table(frames_table)
              .repartition(n_partitions)                              # rebalance: clips vary in length
              .withColumn("pred", classify("jpeg"))
              .select("clip_id", "frame_idx", "ts_s", "pred.*")
              .withColumn("model", F.lit(model)).withColumn("variant", F.lit(variant)))
    writer = scored.write.mode("overwrite")
    if spark.catalog.tableExists(predictions_table):
        writer = writer.option("replaceWhere", f"model = '{model}' AND variant = '{variant}'")
    with Stopwatch() as infer:
        writer.saveAsTable(predictions_table)
    rows.append({
        "run_ts": run_ts, "run_label": p["run_label"], "num_workers": registered,
        "total_cores": shape["total_cores"], "task_cpus": shape["task_cpus"], "node_type": shape["node_type"],
        "model": model, "variant": variant, "backend": "pytorch" if variant == "torch" else "onnxruntime",
        "n_partitions": n_partitions, "batch_size": batch_size, "ort_threads": ort_threads,
        "clips": n_clips, "frames": n_frames_total,
        "extract_s": extract.seconds, "inference_s": infer.seconds,
        "inference_frames_per_s": n_frames_total / infer.seconds,
        "end_to_end_frames_per_s": n_frames_total / (extract.seconds + infer.seconds),
    })
    print(f"{model} {variant}: {infer.seconds:.1f} s, {rows[-1]['inference_frames_per_s']:.1f} frames/s")

# COMMAND ----------
from video_cls.tables import bench_cluster_df

bench_table = f"{catalog}.{schema}.bench_cluster"
ensure_table(spark, bench_table, BENCH_CLUSTER_SCHEMA)
result = bench_cluster_df(spark, rows)
result.write.mode("append").saveAsTable(bench_table)               # blind append: safe in parallel
display(result)
