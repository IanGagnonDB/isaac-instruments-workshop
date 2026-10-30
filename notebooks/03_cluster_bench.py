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
# MAGIC
# MAGIC **Core sweep on one cluster (`task_caps`, e.g. `4,8,16`).** For small vCPU quotas: instead of one
# MAGIC cluster per size, one cluster runs the pipeline once per cap *k* with exactly *k* partitions, so at
# MAGIC most *k* tasks (= *k* cores) work at a time. Each pass reads its own `max_clips` clips (the next
# MAGIC slice by `clip_id`), so no pass benefits from files an earlier pass cached. This scales *cores*,
# MAGIC not machines: tasks of a small cap may share one worker's network and storage bandwidth.
# MAGIC Rows are labelled `<run_label>_cap<k>` with `total_cores = k`.

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
          ("ort_threads", "1"), ("n_frames", "10"), ("max_clips", "0"), ("run_label", ""),
          ("task_caps", "")]
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
task_caps = [int(c) for c in p["task_caps"].split(",") if c.strip()]   # empty = one pass, all slots

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
if task_caps:
    if not max_clips:
        raise ValueError("task_caps needs max_clips > 0: each pass extracts its own max_clips clips")
    too_big = [k for k in task_caps if k > shape["task_slots"]]
    if too_big:
        raise ValueError(f"task_caps {too_big} exceed this cluster's {shape['task_slots']} task slots")
print({"workers": registered, **shape, "n_partitions": n_partitions, "batch_size": batch_size,
       "ort_threads": ort_threads, "runs": runs, "task_caps": task_caps})

# COMMAND ----------
# MAGIC %md
# MAGIC ## Stage 1 (frame extraction) and stage 2 (inference, one pass per model/variant)
# MAGIC Once per pass: one pass using every task slot, or one per `task_caps` entry.

# COMMAND ----------
import functools
from datetime import datetime, timezone

from pyspark.sql import Window
from pyspark.sql import functions as F

from video_cls.bench import Stopwatch, plan_passes
from video_cls.frames import FRAME_SCHEMA, extract_frames
from video_cls.infer import make_classifier
from video_cls.models import artifact_path

frames_table = f"{catalog}.{schema}.frames_{suffix}"
predictions_table = f"{catalog}.{schema}.frame_predictions_{suffix}"
events = (spark.table(f"{catalog}.{schema}.events").select("clip_id", "clip_path")
          .withColumn("clip_rank", F.row_number().over(Window.orderBy("clip_id"))))
passes = plan_passes(p["run_label"], task_caps, n_partitions, shape["total_cores"], max_clips)
needed = passes[-1]["offset"] + passes[-1]["limit"]
if needed and events.count() < needed:
    raise ValueError(f"{len(passes)} passes x {max_clips} clips need {needed} events; "
                     "lower max_clips or run setup with more n_clips")


def extract(clips, partitions):
    """Clips -> frames_table with exactly `partitions` tasks. Returns (seconds, frames)."""
    with Stopwatch() as watch:
        (clips.repartition(partitions)                               # spread clips across the cores
              .mapInPandas(functools.partial(extract_frames, n_frames=n_frames), FRAME_SCHEMA)
              .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(frames_table))
    return watch.seconds, spark.table(frames_table).count()


def infer(model, variant, partitions):
    """frames_table -> predictions_table slice for one model/variant. Returns seconds."""
    classify = make_classifier(artifact_path(models_dir, model, variant), threads=ort_threads)
    scored = (spark.table(frames_table)
              .repartition(partitions)                                # rebalance: clips vary in length
              .withColumn("pred", classify("jpeg"))
              .select("clip_id", "frame_idx", "ts_s", "pred.*")
              .withColumn("model", F.lit(model)).withColumn("variant", F.lit(variant)))
    writer = scored.write.mode("overwrite")
    if spark.catalog.tableExists(predictions_table):
        writer = writer.option("replaceWhere", f"model = '{model}' AND variant = '{variant}'")
    with Stopwatch() as watch:
        writer.saveAsTable(predictions_table)
    return watch.seconds


run_ts = datetime.now(timezone.utc)
rows = []
for spec in passes:
    clips = events
    if spec["limit"]:
        clips = events.where(F.col("clip_rank").between(spec["offset"] + 1, spec["offset"] + spec["limit"]))
    clips = clips.drop("clip_rank")
    n_clips = clips.count()
    extract_s, n_frames_total = extract(clips, spec["partitions"])
    print(f"[{spec['label']}] extracted {n_frames_total} frames from {n_clips} clips with "
          f"{spec['partitions']} partitions in {extract_s:.1f} s ({n_frames_total / extract_s:.1f} frames/s)")
    for model, variant in runs:
        infer_s = infer(model, variant, spec["partitions"])
        rows.append({
            "run_ts": run_ts, "run_label": spec["label"], "num_workers": registered,
            "total_cores": spec["cores"], "task_cpus": shape["task_cpus"], "node_type": shape["node_type"],
            "model": model, "variant": variant, "backend": "pytorch" if variant == "torch" else "onnxruntime",
            "n_partitions": spec["partitions"], "batch_size": batch_size, "ort_threads": ort_threads,
            "clips": n_clips, "frames": n_frames_total,
            "extract_s": extract_s, "inference_s": infer_s,
            "inference_frames_per_s": n_frames_total / infer_s,
            "end_to_end_frames_per_s": n_frames_total / (extract_s + infer_s),
        })
        print(f"[{spec['label']}] {model} {variant}: {infer_s:.1f} s, "
              f"{rows[-1]['inference_frames_per_s']:.1f} frames/s")

# COMMAND ----------
from video_cls.tables import bench_cluster_df

bench_table = f"{catalog}.{schema}.bench_cluster"
ensure_table(spark, bench_table, BENCH_CLUSTER_SCHEMA)
result = bench_cluster_df(spark, rows)
result.write.mode("append").saveAsTable(bench_table)               # blind append: safe in parallel
display(result)
