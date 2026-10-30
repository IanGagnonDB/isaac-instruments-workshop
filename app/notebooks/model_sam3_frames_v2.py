# Databricks notebook source
# MAGIC %md
# MAGIC # SAM3 frame-batch model V2
# MAGIC
# MAGIC This preserves the original `facebook/sam3` model and prompt semantics while
# MAGIC changing the serving boundary: callers send resized JPEG frame batches and
# MAGIC receive compact PNG masks. The checkpoint is packaged with the MLflow model,
# MAGIC so a serving replica does not need a Hugging Face token at startup.

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "cv")
dbutils.widgets.text("model_name", "transformers-sam3-frames-v2")
dbutils.widgets.text("alias", "performance-v2")
dbutils.widgets.text("checkpoint_root", "")
dbutils.widgets.text("secret_scope", "sam3-video-app")
dbutils.widgets.text("secret_key", "hf_token")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
model_alias = dbutils.widgets.get("alias")
checkpoint_root = dbutils.widgets.get("checkpoint_root").rstrip("/")
secret_scope = dbutils.widgets.get("secret_scope")
secret_key = dbutils.widgets.get("secret_key")
if not catalog or not checkpoint_root:
    raise ValueError("Set the catalog and checkpoint_root parameters (the bundle passes both).")
registered_model_name = f"{catalog}.{schema}.{model_name}"

# COMMAND ----------

import base64
import json
from io import BytesIO

import cv2
import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
from PIL import Image


class SAM3FrameBatchV2(mlflow.pyfunc.PythonModel):
    """Batch SAM3 inference with a compact, serving-friendly contract."""

    def load_context(self, context):
        from transformers import Sam3Model, Sam3Processor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        checkpoint = context.artifacts["sam3_snapshot"]
        self.model = Sam3Model.from_pretrained(
            checkpoint,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(
            checkpoint,
            local_files_only=True,
        )

    @staticmethod
    def _decode_jpeg(encoded: str) -> Image.Image:
        return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")

    @staticmethod
    def _encode_mask(mask: torch.Tensor) -> str:
        binary = (mask.detach().to("cpu").numpy() > 0).astype(np.uint8) * 255
        ok, encoded = cv2.imencode(".png", binary)
        if not ok:
            raise RuntimeError("OpenCV could not encode a segmentation mask.")
        return base64.b64encode(encoded).decode("ascii")

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        if model_input.empty:
            return pd.DataFrame(columns=["frame_id", "masks_json", "scores_json"])

        prompts = model_input["prompt"].astype(str).unique()
        thresholds = model_input["threshold"].astype(float).unique()
        mask_thresholds = model_input["mask_threshold"].astype(float).unique()
        if len(prompts) != 1 or len(thresholds) != 1 or len(mask_thresholds) != 1:
            raise ValueError("Each request batch must use one prompt and threshold pair.")

        images = [self._decode_jpeg(value) for value in model_input["image_b64"]]
        inputs = self.processor(
            images=images,
            text=[prompts[0]] * len(images),
            return_tensors="pt",
        ).to(self.device)
        for key, value in inputs.items():
            if value.dtype == torch.float32:
                inputs[key] = value.to(self.model.dtype)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        processed = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=float(thresholds[0]),
            mask_threshold=float(mask_thresholds[0]),
            target_sizes=inputs["original_sizes"].tolist(),
        )

        records = []
        for frame_id, result in zip(model_input["frame_id"], processed, strict=True):
            records.append(
                {
                    "frame_id": str(frame_id),
                    "masks_json": json.dumps(
                        [self._encode_mask(mask) for mask in result["masks"]],
                        separators=(",", ":"),
                    ),
                    "scores_json": json.dumps(
                        result["scores"].detach().to("cpu").tolist(),
                        separators=(",", ":"),
                    ),
                }
            )
        return pd.DataFrame.from_records(records)


signature = ModelSignature(
    inputs=Schema(
        [
            ColSpec("string", "frame_id"),
            ColSpec("string", "image_b64"),
            ColSpec("string", "prompt"),
            ColSpec("double", "threshold"),
            ColSpec("double", "mask_threshold"),
        ]
    ),
    outputs=Schema(
        [
            ColSpec("string", "frame_id"),
            ColSpec("string", "masks_json"),
            ColSpec("string", "scores_json"),
        ]
    ),
)

# COMMAND ----------

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import model_info, snapshot_download

hf_token = dbutils.secrets.get(secret_scope, secret_key)
repository = "facebook/sam3"
revision = model_info(repository, token=hf_token).sha
durable_checkpoint = Path(checkpoint_root) / "facebook-sam3" / revision
checkpoint_manifest = durable_checkpoint / "_checkpoint.json"

if checkpoint_manifest.is_file():
    snapshot_path = str(durable_checkpoint)
    print(f"Reusing durable checkpoint {snapshot_path}")
else:
    checkpoint_directory = tempfile.mkdtemp(prefix="sam3-v2-checkpoint-")
    local_snapshot = snapshot_download(
        repo_id=repository,
        revision=revision,
        token=hf_token,
        local_dir=checkpoint_directory,
    )
    durable_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_snapshot, durable_checkpoint, dirs_exist_ok=True)
    checkpoint_manifest.write_text(
        json.dumps({"repository": repository, "revision": revision}, indent=2),
        encoding="utf-8",
    )
    snapshot_path = str(durable_checkpoint)
    print(f"Saved full durable checkpoint to {snapshot_path}")

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run(run_name="sam3-frame-batch-v2") as run:
    model_info = mlflow.pyfunc.log_model(
        name="model",
        python_model=SAM3FrameBatchV2(),
        artifacts={"sam3_snapshot": snapshot_path},
        signature=signature,
        pip_requirements=[
            "mlflow",
            "transformers==5.4.0",
            "torch",
            "torchvision",
            "opencv-python-headless==4.12.0.88",
            "pillow",
            "pandas",
            "accelerate==1.12.0",
        ],
        registered_model_name=registered_model_name,
    )

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions(f"name='{registered_model_name}'")
version = max(versions, key=lambda item: int(item.version))
client.set_registered_model_alias(registered_model_name, model_alias, version.version)

print(f"Registered {registered_model_name} version {version.version} as @{model_alias}")
print("Deploy this version to an always-on GPU endpoint before running the V2 job.")
