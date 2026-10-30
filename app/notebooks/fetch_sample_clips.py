# Databricks notebook source
# MAGIC %md
# MAGIC # Fetch sample dashcam clips (Nexar)
# MAGIC
# MAGIC Copies a small sample of the Nexar Dashcam Collision Prediction dataset
# MAGIC (`nexar-ai/nexar_collision_prediction` on Hugging Face) into the app's `inputs`
# MAGIC folder, and saves the dataset licence next to it.
# MAGIC
# MAGIC Before running: accept the dataset terms on Hugging Face with the same account as
# MAGIC the token in the secret scope. Nexar Open Data License: attribute Nexar, and do not
# MAGIC sell, sublicense, or redistribute the dataset for profit.

# COMMAND ----------

dbutils.widgets.text("volume_root", "")
dbutils.widgets.text("per_class", "15")
dbutils.widgets.text("secret_scope", "sam3-video-app")
dbutils.widgets.text("secret_key", "hf_token")

volume_root = dbutils.widgets.get("volume_root").rstrip("/")
per_class = int(dbutils.widgets.get("per_class"))
secret_scope = dbutils.widgets.get("secret_scope")
secret_key = dbutils.widgets.get("secret_key")
if not volume_root:
    raise ValueError("Set the volume_root parameter (the bundle passes it).")

inputs_dir = f"{volume_root}/inputs"
license_dir = f"{volume_root}/datasets/nexar"

# COMMAND ----------

import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

repo = "nexar-ai/nexar_collision_prediction"
token = dbutils.secrets.get(secret_scope, secret_key)
files = HfApi().list_repo_files(repo, repo_type="dataset", token=token)

# Half collision or near-miss clips, half normal driving.
selected = []
for label in ("positive", "negative"):
    clips = sorted(f for f in files if f.startswith(f"train/{label}/") and f.endswith(".mp4"))
    selected += clips[:per_class]

Path(inputs_dir).mkdir(parents=True, exist_ok=True)
Path(license_dir).mkdir(parents=True, exist_ok=True)

for name in ["LICENSE", "README.md"]:
    shutil.copy(hf_hub_download(repo, name, repo_type="dataset", token=token), f"{license_dir}/{name}")

for path in selected:
    local = hf_hub_download(repo, path, repo_type="dataset", token=token)
    label = path.split("/")[1]
    target = f"{inputs_dir}/nexar_{label}_{Path(path).name}"
    shutil.copy(local, target)
    print(target)

print(f"Copied {len(selected)} clips to {inputs_dir}")
