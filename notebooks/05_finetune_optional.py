# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Optional: fine-tune `stop_sign_visible` yes/no
# MAGIC
# MAGIC > **Not runnable with the current dataset.** D2-City has no stop-sign labels, so `labels/yes` and
# MAGIC > `labels/no` start empty and this notebook exits with "Skipped". It needs a labelled set first:
# MAGIC > ISAAC's own frames sorted by hand, or another dataset with stop-sign annotations.
# MAGIC
# MAGIC Freezes the ResNet-18 backbone from 01 and trains a new 2-class head on CPU from labelled frames:
# MAGIC
# MAGIC ```
# MAGIC /Volumes/<catalog>/<schema>/<volume>/labels/yes/*.jpg   stop sign visible
# MAGIC /Volumes/<catalog>/<schema>/<volume>/labels/no/*.jpg    no stop sign
# MAGIC ```
# MAGIC Need frames to label? Run with `export_unlabeled = 300`: it writes that many frames from a `frames_<label>`
# MAGIC table to `labels/unlabeled/` and stops. Move each file into `yes/` or `no/`, then rerun with 0.
# MAGIC
# MAGIC The result is exported to ONNX fp32 + int8 exactly like 01 (`models/stopsign_resnet18*.onnx`).
# MAGIC Use it in 04 with `sign_source = finetuned`.

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

PARAMS = [("catalog", ""), ("schema", ""), ("volume", "video"), ("epochs", "8"), ("export_unlabeled", "0")]
for name, default in PARAMS:
    dbutils.widgets.text(name, default)
p = {name: dbutils.widgets.get(name) for name, _ in PARAMS}

catalog, schema = p["catalog"], p["schema"]
root = f"/Volumes/{catalog}/{schema}/{p['volume']}"
models_dir, labels_dir = f"{root}/models", f"{root}/labels"

# COMMAND ----------
export_unlabeled = int(p["export_unlabeled"])
if export_unlabeled:
    out = f"{labels_dir}/unlabeled"
    os.makedirs(out, exist_ok=True)
    from video_cls.tables import find_table
    frames_table = find_table(spark, catalog, schema, "frames")
    if frames_table is None:
        raise RuntimeError("No frames table yet: run the benchmark (03) or scoring (04) job first.")
    rows = (spark.table(frames_table).select("clip_id", "frame_idx", "jpeg")
            .orderBy("clip_id", "frame_idx").limit(export_unlabeled).collect())
    for r in rows:
        with open(f"{out}/{r.clip_id}_f{r.frame_idx:02d}.jpg", "wb") as handle:
            handle.write(r.jpeg)
    dbutils.notebook.exit(f"Wrote {len(rows)} frames to {out}. Sort them into labels/yes and labels/no.")

# COMMAND ----------
from video_cls.finetune import load_labelled, split

x, y, names = load_labelled(labels_dir)
counts = {"no": int((y == 0).sum()), "yes": int((y == 1).sum())}
print(f"Labelled frames: {counts}")
if min(counts.values()) < 5:
    dbutils.notebook.exit(f"Skipped: need at least 5 labelled frames per class in {labels_dir}, found {counts}.")
train_idx, val_idx = split(len(y))

# COMMAND ----------
import shutil
import tempfile

from video_cls.finetune import accuracy, train_head
from video_cls.models import (artifact_path, export_onnx, load_checkpoint, load_predictor,
                              quantize_int8, save_checkpoint, top1_agreement)

model = load_checkpoint(artifact_path(models_dir, "resnet18", "torch"))
source = "huggingface" if not hasattr(model, "fc") else "torchvision"
losses = train_head(model, x[train_idx], y[train_idx], epochs=int(p["epochs"]))
print("Training loss per epoch:", [round(l, 4) for l in losses])
print(f"Validation accuracy: {accuracy(model, x[val_idx], y[val_idx]):.1%} on {len(val_idx)} frames")

scratch = tempfile.mkdtemp(dir="/local_disk0/tmp" if os.path.isdir("/local_disk0/tmp") else None)
local = {v: artifact_path(scratch, "stopsign_resnet18", v) for v in ("torch", "fp32", "int8")}
save_checkpoint(model, source, "resnet18", local["torch"])
export_onnx(model, local["fp32"])
quantize_int8(local["fp32"], local["int8"], x[train_idx][:100])
reference = load_predictor(local["torch"], threads=os.cpu_count())(x)
for variant, path in local.items():
    if variant != "torch":
        agreement = top1_agreement(reference, load_predictor(path, threads=os.cpu_count())(x))
        print(f"stopsign_resnet18 {variant}: top-1 agreement with PyTorch {agreement:.1%}")
    shutil.copyfile(path, artifact_path(models_dir, "stopsign_resnet18", variant))
shutil.rmtree(scratch, ignore_errors=True)
print("Saved models/stopsign_resnet18{.pt,.onnx,_int8.onnx}. Rerun 04 with sign_source=finetuned.")
