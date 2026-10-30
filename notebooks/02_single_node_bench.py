# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Single-node benchmark: PyTorch vs ONNX fp32 vs ONNX int8
# MAGIC
# MAGIC Driver only, no Spark tasks. This isolates model speed from Spark and decode overhead.
# MAGIC For each model x backend x batch size: warm up, then time `n_frames` preprocessed frames.
# MAGIC Results are appended to `bench_single_node`.

# COMMAND ----------
import os
import sys

try:
    import video_cls
except ImportError:
    sys.path.insert(0, os.path.abspath("../src"))
    import video_cls
    from pyspark import cloudpickle
    cloudpickle.register_pickle_by_value(video_cls)

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("volume", "video")
dbutils.widgets.text("models", "resnet18,resnet50")
dbutils.widgets.text("batch_sizes", "1,16,64")
dbutils.widgets.text("n_frames", "500")
dbutils.widgets.text("threads", "0")       # 0 = all cores on this node

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
model_names = [m.strip() for m in dbutils.widgets.get("models").split(",") if m.strip()]
batch_sizes = [int(b) for b in dbutils.widgets.get("batch_sizes").split(",")]
n_frames = int(dbutils.widgets.get("n_frames"))
models_dir = f"/Volumes/{catalog}/{schema}/{volume}/models"

# COMMAND ----------
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from video_cls.bench import cpu_count, time_predictor
from video_cls.models import artifact_path, load_predictor

threads = int(dbutils.widgets.get("threads")) or cpu_count()
frames = np.load(f"{models_dir}/calibration.npy")
node_type = spark.conf.get("spark.databricks.clusterUsageTags.clusterNodeType", "unknown")
run_ts = datetime.now(timezone.utc)

rows = []
for name in model_names:
    for variant in ("torch", "fp32", "int8"):
        predictor = load_predictor(artifact_path(models_dir, name, variant), threads=threads)
        for batch_size in batch_sizes:
            seconds, timed = time_predictor(predictor, frames, batch_size, n_frames)
            rows.append({"run_ts": run_ts, "node_type": node_type, "model": name,
                         "backend": "pytorch" if variant == "torch" else "onnxruntime",
                         "variant": variant, "batch_size": batch_size, "threads": threads,
                         "frames": timed, "seconds": seconds,
                         "ms_per_frame": 1000 * seconds / timed, "frames_per_s": timed / seconds})
            print(f"{name:9s} {variant:5s} bs={batch_size:3d} threads={threads}: "
                  f"{rows[-1]['ms_per_frame']:7.2f} ms/frame  {rows[-1]['frames_per_s']:8.1f} frames/s")
        del predictor

# COMMAND ----------
result = spark.createDataFrame(pd.DataFrame(rows))
result.write.mode("append").option("mergeSchema", "true").saveAsTable(f"{catalog}.{schema}.bench_single_node")
display(result.orderBy("model", "batch_size", "ms_per_frame"))
