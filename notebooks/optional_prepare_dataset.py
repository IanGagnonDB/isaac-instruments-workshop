# Databricks notebook source
# MAGIC %md
# MAGIC # Optional · Prepare shared data and models
# MAGIC One designated participant runs this once when assets are missing. Complete data/models are reused.
# MAGIC Select **Serverless → ML base environment → version 6**. Then **Run All**.
# MAGIC The install cell adds notebook dependencies; the ML base supplies Torch, torchvision, transformers,
# MAGIC NumPy and pandas. The next cell restarts Python before imports. No manual dependency-panel editing.

# COMMAND ----------
# MAGIC %pip install "opencv-python-headless==4.11.0.86"

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_configuration

# COMMAND ----------

import json
import uuid
from pathlib import Path
from utils.workshop_support import prepare_selected_dataset, selected_dataset_status
from utils.inference import stage_checkpoint, load_engine, run_events, read_manifest, REVISIONS

try:
    dbutils.widgets.get('dataset_action')
except Exception:
    dbutils.widgets.dropdown('dataset_action', 'Reuse existing data',
                            ['Reuse existing data', 'Download D2-City', 'Use synthetic fallback'])
print('Current state:', selected_dataset_status(WORKSHOP['volume_root'], WORKSHOP['dataset']))
print('Action:', dbutils.widgets.get('dataset_action'))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Choose an action above, then rerun
# MAGIC - **Reuse existing data**: never starts a video download. Both participants can use this after preparation.
# MAGIC - **Download D2-City**: default real-data preparation; full source ZIP download, one copy per source video.
# MAGIC - **Use synthetic fallback**: first set DATASET=`synthetic` in 00_configuration. Generates three tiny videos
# MAGIC   locally; never deletes or replaces real data. This is for pipeline mechanics, not prediction quality.
# MAGIC
# MAGIC Real preparation failures stop with instructions. The fallback replaces videos only: pretrained model
# MAGIC weights still need Hugging Face access if missing. No third-party videos are bundled in Git.

# COMMAND ----------
probe = Path(WORKSHOP['volume_root']) / f'.preflight-{uuid.uuid4().hex}.txt'
try:
    probe.write_text('workshop')
    assert probe.read_text() == 'workshop'
finally:
    if probe.exists():
        probe.unlink()
status = prepare_selected_dataset(WORKSHOP['volume_root'], WORKSHOP['dataset'],
                                  dbutils.widgets.get('dataset_action'))
print(json.dumps(status, indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Stage the selected model and test one frame
# MAGIC Existing model files are reused. Downloads happen here in the notebook process, never inside a UDF.
# MAGIC If a preparation session was interrupted, verify it has stopped before removing its stale lock directory.

# COMMAND ----------
model_path = stage_checkpoint(WORKSHOP['models_root'], WORKSHOP['model_id'])
engine, _ = load_engine(model_path)
smoke = run_events(read_manifest(WORKSHOP['root'], 1), engine, WORKSHOP['model_id'],
                   REVISIONS[WORKSHOP['model_id']], 1, {'default': 1, 'snapshot': 1})
assert smoke['events'][0]['predictions'], 'No predictions produced'
print(f"PASS: one decoded frame and CPU inference. Dataset={smoke['dataset_kind']}. Open 01 next.")
