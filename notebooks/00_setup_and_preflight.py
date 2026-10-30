# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Start here — two people, one workspace
# MAGIC 1. One person clones this repository as a **Git Folder** and grants the other person folder access.
# MAGIC 2. Edit `workshop_config.json` together: choose an existing catalog, schema, Volume and SQL warehouse ID.
# MAGIC    Create them in Catalog Explorer / SQL Warehouses first if needed and permitted. Both people need
# MAGIC    USE CATALOG/SCHEMA, READ/WRITE VOLUME, CREATE MODEL, CREATE TABLE and CAN USE on the warehouse.
# MAGIC 3. CPU starter: select Serverless environment version **4** (Python 3.12). In **each notebook 00–04**,
# MAGIC    use the Environment panel to add the dependencies in `requirements-cpu.txt`. Each person must select
# MAGIC    that environment for their session. Wait for installation before running. No Ray is required for CPU.
# MAGIC    For classic compute use a compatible Python 3.12 runtime and install those dependencies on the cluster.
# MAGIC    For `classic_single_node` / `ray_on_spark`, also install a compatible Ray package or select a DBR ML
# MAGIC    runtime that includes Ray. Ray is intentionally excluded from the Serverless CPU starter dependencies.
# MAGIC    GPU: use the GPU runtime's compatible Torch/torchvision instead; do not install the CPU requirements.
# MAGIC 4. Both people run the checks below. **Designate one person** for optional shared data/model preparation.
# MAGIC    All setup widgets default to `false`; enable only the preparation needed, then rerun this notebook.
# MAGIC    Keep the same Git Folder revision throughout the session; changes affect both participants.
# MAGIC
# MAGIC This notebook does not promise GPU or Ray availability. Serverless CPU supports a partial workshop:
# MAGIC Sequential/Spark diagnostics, lifecycle and tracing; a matched three-way comparison needs compatible compute.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
import importlib.metadata
import json
import uuid
from databricks.sdk import WorkspaceClient
from workshop_support import dataset_status, prepare_shared_dataset

for name in ('prepare_dataset', 'prepare_models'):
    try:
        dbutils.widgets.get(name)
    except Exception:
        dbutils.widgets.dropdown(name, 'false', ['false', 'true'])

required = ['torch', 'torchvision', 'transformers', 'opencv-python-headless', 'mlflow',
            'numpy', 'pillow', 'pandas', 'pyarrow', 'huggingface-hub', 'matplotlib', 'psutil']
missing = []
for package in required:
    try:
        importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        missing.append(package)
if missing:
    raise RuntimeError(f'Missing dependencies: {missing}. Configure this notebook Environment using requirements-cpu.txt.')
if WORKSHOP['compute_mode'] in ('classic_single_node', 'ray_on_spark'):
    try:
        import ray
    except ImportError as exc:
        raise RuntimeError('This compute mode needs Ray. Select a compatible DBR ML runtime or install Ray on the cluster.') from exc
workspace = WorkspaceClient()
workspace.schemas.get(WORKSHOP['trace_schema'])
workspace.volumes.read(f"{WORKSHOP['catalog']}.{WORKSHOP['schema']}.{WORKSHOP['volume']}")
workspace.warehouses.get(WORKSHOP['trace_warehouse_id'])
# Read and write the actual Volume, rather than assuming grants from metadata visibility.
probe = Path(WORKSHOP['root']) / f".preflight-{WORKSHOP['participant']}-{uuid.uuid4().hex}.txt"
try:
    probe.write_text('workshop preflight')
    assert probe.read_text() == 'workshop preflight'
finally:
    if probe.exists():
        probe.unlink()
print(json.dumps(WORKSHOP, indent=2))
print('Resource visibility and Volume read/write passed. Model registration and trace table privileges are exercised in 03/04.')
print('Dataset status:', dataset_status(WORKSHOP['root']))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Optional — prepare the shared dataset once, only if missing
# MAGIC Set `prepare_dataset=true` only for the designated preparer. Complete data is validated and reused.
# MAGIC The initial download is the full source archive; `copies=1` avoids multiplying it, not downloading it.
# MAGIC Scoring is limited to two events by default. Larger duplication is a later benchmark exercise.
# MAGIC Incomplete data is never deleted or overwritten: inspect it or select a fresh Volume.
# MAGIC A shared lock directory blocks overlapping preparation attempts; still use one designated preparer.
# MAGIC After an interrupted session, confirm no preparation is running before manually removing a stale
# MAGIC `.dataset-preparation.lock` directory. Directory locks depend on the backing storage; they are not a transaction.

# COMMAND ----------
if dbutils.widgets.get('prepare_dataset') == 'true':
    print(prepare_shared_dataset(WORKSHOP['root'], copies=1, workers=4))
else:
    print('Dataset preparation skipped. Existing valid data is required before scoring.')

# COMMAND ----------
# MAGIC %run ./01_classify_video $library_only="true"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Optional — prepare the shared pinned model snapshots once
# MAGIC Set `prepare_models=true` for the designated preparer if model files are missing.
# MAGIC Model downloads require outbound Hugging Face access; weights remain in the shared Volume.

# COMMAND ----------
if dbutils.widgets.get('prepare_models') == 'true':
    from huggingface_hub import snapshot_download
    lock = Path(ROOT) / '.model-preparation.lock'
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError('Model preparation is already running, or its lock is stale. Check with the preparer.') from None
    try:
        for model_id, revision in REVISIONS.items():
            snapshot_download(repo_id=model_id, revision=revision,
                              local_dir=str(Path(ROOT) / 'workshop_models' / model_id.split('/')[-1] / revision),
                              allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors'])
    finally:
        lock.rmdir()
else:
    print('Model preparation skipped; checking staged snapshots next.')

# COMMAND ----------
# MAGIC %md
# MAGIC ## Readiness and one-event smoke test — both participants run this
# MAGIC This checks one decoded frame and actual model inference on the configured device. Notebook 02 probes
# MAGIC Ray/Spark separately; metadata visibility does not prove CREATE MODEL/TABLE permissions or trace readback.

# COMMAND ----------
status = dataset_status(ROOT)
if status['state'] != 'ready':
    raise RuntimeError(f'Dataset not ready: {status}. Ask the designated preparer to run the optional dataset cell.')
checkpoints = prepare_checkpoints(ROOT)
engine, _ = load_engine(checkpoints['microsoft/resnet-50'], DEVICE)
smoke = run_events(read_manifest(ROOT, 1), engine, 'microsoft/resnet-50',
                   REVISIONS['microsoft/resnet-50'], 1, {'default': 1, 'snapshot': 1})
assert smoke['events'][0]['predictions'], 'Smoke test produced no predictions'
print('PASS: dataset, staged snapshots, video decode and one-frame inference.')
print(f'Participant: {WORKSHOP["participant"]}; registered model prefix: {MODEL_PREFIX}')
print(f'Compute mode: {WORKSHOP["compute_mode"]}; device: {DEVICE}')
if WORKSHOP['compute_mode'] == 'serverless_cpu':
    print('PARTIAL PATH: Ray and the full matched three-way benchmark are unavailable in this configuration.')
print('Open 01 next. No dataset download or bundle deployment is needed in 01–04.')
