# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Configure once
# MAGIC Clone as a Git Folder. Select **Serverless → ML base environment → version 6** in each numbered notebook.
# MAGIC Edit the Volume path below. Both participants share data and model files; outputs and registered models
# MAGIC are isolated by signed-in user. Use one designated preparer for shared data/models.
# MAGIC
# MAGIC Real D2-City footage is the default. Open **optional_prepare_dataset** once if data or weights are missing.
# MAGIC If a real download fails, it stops. To explicitly use generated videos, change DATASET to `synthetic` here
# MAGIC and choose **Use synthetic fallback** in that preparation notebook. Predictions on generated shapes have
# MAGIC no semantic meaning. Keep this shared configuration unchanged while another participant is running.

# COMMAND ----------
# REQUIRED: choose an existing UC Volume accessible to both participants.
VOLUME_ROOT = '/Volumes/CHOOSE_CATALOG/CHOOSE_SCHEMA/CHOOSE_VOLUME'
DATASET = 'd2_city'  # Explicit fallback: 'synthetic'. Never selected automatically.

# Small CPU starter; scale the workload after the first successful run.
EVENT_LIMIT = 4
FRAMES_PER_EVENT = 3
INFERENCE_BATCH_SIZE = 2
MODEL_ID = 'microsoft/resnet-18'  # Lighter CPU starter. Optional extension: 'microsoft/resnet-50'.

# Only notebook 04 needs a SQL warehouse. Leave blank until reaching that exercise.
TRACE_WAREHOUSE_ID = ''

# COMMAND ----------
from pathlib import Path
import sys

REPOSITORY = Path.cwd()
if not (REPOSITORY / 'utils/workshop_support.py').is_file():
    REPOSITORY = REPOSITORY.parent
if not (REPOSITORY / 'utils/workshop_support.py').is_file():
    raise RuntimeError('Open notebooks directly inside the cloned Git Folder.')
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
from utils.workshop_support import build_config

WORKSHOP = build_config({
    'volume_root': VOLUME_ROOT, 'dataset': DATASET, 'event_limit': EVENT_LIMIT,
    'frames_per_event': FRAMES_PER_EVENT, 'batch_size': INFERENCE_BATCH_SIZE,
    'model_id': MODEL_ID, 'trace_warehouse_id': TRACE_WAREHOUSE_ID,
}, spark.sql('SELECT current_user()').first()[0])
print(f"Dataset: {WORKSHOP['dataset']} | participant: {WORKSHOP['participant']} | CPU model: {MODEL_ID}")
