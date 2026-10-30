# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Export ResNet-18/50 to ONNX fp32 and int8
# MAGIC
# MAGIC * Weights: torchvision ImageNet defaults (BSD). If `download.pytorch.org` is blocked, the loader
# MAGIC   falls back to Hugging Face `microsoft/resnet-18` / `microsoft/resnet-50` (Apache 2.0) and records it.
# MAGIC * ONNX opset 17 with a dynamic batch axis.
# MAGIC * int8: ONNX Runtime **static** quantization (QDQ, per-channel weights), calibrated on ~100 frames
# MAGIC   extracted from the sample clips. Static is the recommended mode for CNNs.
# MAGIC * Sanity check: top-1 agreement with PyTorch on the calibration frames.
# MAGIC
# MAGIC Limitation: ImageNet-1k has no "stop sign" class. Class 919 ("street sign") is used later as a proxy.

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
dbutils.widgets.text("n_calibration", "100")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
model_names = [m.strip() for m in dbutils.widgets.get("models").split(",") if m.strip()]
n_calibration = int(dbutils.widgets.get("n_calibration"))
root = f"/Volumes/{catalog}/{schema}/{volume}"
models_dir = f"{root}/models"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Calibration frames from the sample clips
# MAGIC 10 frames from each of the first clips. clip_ids are assigned round-robin over the source clips,
# MAGIC so consecutive clip_ids are different videos. Saved for reuse by 02 and 05.

# COMMAND ----------
import math

import numpy as np

from video_cls.frames import extract_clip
from video_cls.preprocess import preprocess_batch

n_clips = math.ceil(n_calibration / 10)
clips = (spark.table(f"{catalog}.{schema}.events").select("clip_id", "clip_path")
         .orderBy("clip_id").limit(n_clips).collect())
jpegs = [row[3] for c in clips for row in extract_clip(c.clip_id, c.clip_path, n_frames=10)][:n_calibration]
calibration = preprocess_batch(jpegs)
np.save(f"{models_dir}/calibration.npy", calibration)
print(f"Calibration frames: {calibration.shape} from {len(clips)} clips")

# COMMAND ----------
import shutil
import tempfile
from datetime import datetime, timezone

from video_cls.models import (artifact_path, export_onnx, load_pretrained, quantize_int8,
                              save_checkpoint, load_predictor, top1_agreement)

# Build in local scratch, then copy whole files to the volume.
scratch = tempfile.mkdtemp(dir="/local_disk0/tmp" if os.path.isdir("/local_disk0/tmp") else None)
results = []
for name in model_names:
    model, source = load_pretrained(name)
    local = {v: artifact_path(scratch, name, v) for v in ("torch", "fp32", "int8")}
    save_checkpoint(model, source, name, local["torch"])
    export_onnx(model, local["fp32"])
    quantize_int8(local["fp32"], local["int8"], calibration)

    reference = load_predictor(local["torch"], threads=os.cpu_count())(calibration)
    for variant, path in local.items():
        logits = reference if variant == "torch" else load_predictor(path, threads=os.cpu_count())(calibration)
        dest = artifact_path(models_dir, name, variant)
        shutil.copyfile(path, dest)
        results.append({"model": name, "weights_source": source, "variant": variant, "path": dest,
                        "size_mb": round(os.path.getsize(path) / 1e6, 1),
                        "top1_agreement_vs_torch": top1_agreement(reference, logits),
                        "calibration_frames": len(calibration),
                        "exported_at": datetime.now(timezone.utc)})
    print(f"{name}: weights from {source}; exported torch / fp32 / int8")
shutil.rmtree(scratch, ignore_errors=True)

# COMMAND ----------
import pandas as pd

exports = spark.createDataFrame(pd.DataFrame(results))
exports.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.model_exports")
display(exports.select("model", "weights_source", "variant", "size_mb", "top1_agreement_vs_torch"))

fp32_low = [r for r in results if r["variant"] == "fp32" and r["top1_agreement_vs_torch"] < 0.99]
if fp32_low:
    raise AssertionError(f"ONNX fp32 should match PyTorch top-1 on ~100% of frames: {fp32_low}")
for r in results:
    if r["variant"] == "int8":
        print(f"{r['model']} int8 top-1 agreement with PyTorch: {r['top1_agreement_vs_torch']:.1%}")
