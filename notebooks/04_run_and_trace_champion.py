# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Run the champion and inspect success and failure
# MAGIC Resolve `champion` **once**, load its numeric version, and write that exact version beside every prediction.
# MAGIC This is batch inference. No serving endpoint, training or app is created.
# MAGIC
# MAGIC Use `mlflow==3.17.2` (UC trace storage requires >=3.11.1), plus the environment from 01.
# MAGIC Set `trace_schema` to an explicitly selected existing `catalog.schema`, or arrange creation of a new schema
# MAGIC before running. The destination and SQL warehouse are explicitly selected
# MAGIC in workshop_config.json; no personal workspace defaults are used. Provisioning needs USE CATALOG/SCHEMA, CREATE TABLE and warehouse
# MAGIC usage rights. Traces are stored in governed UC Delta tables; there is no silent storage fallback.
# MAGIC UC trace location binding is permanent for an experiment. A separate trace experiment isolates this choice.

# COMMAND ----------
# MAGIC %run ./01_classify_video $library_only="true"

# COMMAND ----------
from contextlib import contextmanager
from packaging.version import Version
import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
from mlflow.entities.trace_location import UnityCatalog
from databricks.sdk import WorkspaceClient

if Version(mlflow.__version__) < Version('3.11.1'):
    raise RuntimeError('UC trace storage requires MLflow >=3.11.1; use the specified notebook environment')
TRACE_SCHEMA = WORKSHOP['trace_schema']
TRACE_EXPERIMENT = parameter('trace_experiment', EXPERIMENT + '_traces')
if len(TRACE_SCHEMA.split('.')) != 2 or not all(TRACE_SCHEMA.split('.')):
    raise ValueError('Set trace_schema to your explicitly chosen catalog.schema before provisioning trace tables')
trace_catalog, trace_schema = TRACE_SCHEMA.split('.')
mlflow.set_tracking_uri('databricks')
mlflow.set_registry_uri('databricks-uc')
mlflow.tracing.enable()
client = MlflowClient()
workspace = WorkspaceClient()
workspace.current_user.me()  # Verify current workspace credentials before exporting traces.
workspace.schemas.get(TRACE_SCHEMA)  # Verify the selected schema exists; do not create an arbitrary replacement.
warehouse_id = WORKSHOP['trace_warehouse_id']
workspace.warehouses.get(warehouse_id)
os.environ['MLFLOW_TRACING_SQL_WAREHOUSE_ID'] = warehouse_id
existing = mlflow.get_experiment_by_name(TRACE_EXPERIMENT)
if existing and existing.trace_location:
    location = existing.trace_location
    if not isinstance(location, UnityCatalog) or (location.catalog_name, location.schema_name) != (trace_catalog, trace_schema):
        raise RuntimeError('Trace experiment is already bound elsewhere; use a new trace_experiment name')
experiment = mlflow.set_experiment(
    experiment_name=TRACE_EXPERIMENT,
    trace_location=UnityCatalog(catalog_name=trace_catalog, schema_name=trace_schema),
)
assert isinstance(mlflow.get_experiment(experiment.experiment_id).trace_location, UnityCatalog)
print(f'Trace storage: {TRACE_SCHEMA}; experiment {experiment.experiment_id}; warehouse {warehouse_id}')

# COMMAND ----------
name = MODEL_PREFIX + '_resnet50'
selected = client.get_model_version_by_alias(name, 'champion')
resolved_version = str(selected.version)
model_uri = f'models:/{name}/{resolved_version}'
# All artifacts come from this version's run, never from the latest alias or a mutable local processor.
provenance_file = mlflow.artifacts.download_artifacts(run_id=selected.run_id, artifact_path='provenance.json')
provenance = json.loads(Path(provenance_file).read_text())
processor_directory = mlflow.artifacts.download_artifacts(run_id=selected.run_id, artifact_path='processor')
processor = AutoImageProcessor.from_pretrained(processor_directory, local_files_only=True, use_fast=False)
model = mlflow.pytorch.load_model(model_uri, map_location=DEVICE).eval().to(DEVICE)
torch.set_num_threads(1)
cv2.setNumThreads(1)
engine = (processor, model, DEVICE)
warm_engine(engine, BATCH_SIZE)
manifest = read_manifest(ROOT, LIMIT)
result = run_events(manifest, engine, provenance['model_id'], provenance['revision'], BATCH_SIZE,
                    provenance['sampling'], model_version=resolved_version)
for event in result['events']:
    event['registered_model'] = name
save_predictions(f'{OUTPUT}/04_champion_predictions.jsonl', result['events'])
display(pd.DataFrame([{'instance_id': e['instance_id'], 'event_id': e['event_id'],
                       'model_version': e['model_version'], 'frames': len(e['predictions'])} for e in result['events']]))
display(pd.DataFrame([result['stages']]))

# COMMAND ----------
# MAGIC %md
# MAGIC ## A small manual trace demonstration
# MAGIC Process video event → Inspect video / Sample frames / Decode video / Preprocess images /
# MAGIC Run model inference / Aggregate predictions. Only compact IDs, counts, configuration and summaries are logged.
# MAGIC MP4s, decoded images and tensors remain outside trace payloads. Some spans repeat for frame batches.
# MAGIC A deliberately missing file produces an ERROR trace; its expected exception is caught **outside** the root span.
# MAGIC No cross-worker trace propagation is assumed. This demonstration runs in this notebook process only.
# MAGIC Traces explain execution paths; trace durations are not precise GPU compute measurements.

# COMMAND ----------
def trace_event(event):
    context = {'event_id': event['event_id'], 'instance_id': event['instance_id'],
               'batch_size': BATCH_SIZE, 'registered_model': name, 'model_version': resolved_version,
               'revision': provenance['revision']}
    @contextmanager
    def stage_span(stage_name):
        started = time.perf_counter()
        with mlflow.start_span(name=stage_name, span_type='TOOL') as span:
            span.set_inputs({**context, 'stage': stage_name})
            try:
                yield
            except Exception as exc:
                span.set_outputs({'completed': False, 'error_type': type(exc).__name__})
                raise
            else:
                span.set_outputs({'completed': True, 'elapsed_seconds': time.perf_counter() - started})
    with mlflow.start_span(name='Process video event', span_type='CHAIN') as root_span:
        root_span.set_inputs(context)
        root_span.set_attributes(context)
        # root_span and child contexts record exceptions automatically if decoding/inference fails.
        traced = run_events([event], engine, provenance['model_id'], provenance['revision'], BATCH_SIZE,
                            provenance['sampling'], model_version=resolved_version, span=stage_span)
        frames = traced['events'][0]['predictions']
        summary = {'event_id': event['event_id'], 'frame_count': len(frames), 'model_version': resolved_version,
                   'prediction_summary': [{'frame_id': f['frame_id'], 'label': f['label'],
                                            'probability': f['probability']} for f in frames[:3]]}
        root_span.set_outputs(summary)
        root_span.set_attribute('frame_count', len(frames))
        return traced


trace_start_ms = int(time.time() * 1000)
success = trace_event(manifest[0])
success_trace_id = mlflow.get_last_active_trace_id()
# This object exists only in memory; no staged videos are changed or deleted.
bad_event = {**manifest[0], 'instance_id': 'deliberate_failure', 'event_id': 'deliberate_failure',
             'path': str(Path(ROOT) / '__deliberately_missing_video__.mp4')}
try:
    trace_event(bad_event)
except ValueError as exc:
    print(f'Expected demonstration failure: {exc}')
else:
    raise AssertionError('Deliberately missing input unexpectedly succeeded')
failure_trace_id = mlflow.get_last_active_trace_id()
assert success_trace_id and failure_trace_id and success_trace_id != failure_trace_id
print(f'Success trace ID: {success_trace_id}\nFailure trace ID: {failure_trace_id}', flush=True)
write_json(f'{OUTPUT}/04_trace_ids.json', {'success': success_trace_id, 'failure': failure_trace_id,
                                         'experiment_id': experiment.experiment_id, 'trace_schema': TRACE_SCHEMA,
                                         'model_name': name, 'model_version': resolved_version,
                                         'start_time_ms': trace_start_ms})

# COMMAND ----------
# MAGIC %md
# MAGIC ## Verify stored traces and open the trace viewer
# MAGIC This cell has a 60-second hard deadline for one flush and two direct reads. A timeout or access error leaves
# MAGIC the trace IDs above available for inspection and fails validation visibly. Do not repeatedly poll a busy warehouse.
# MAGIC Retention/access follow the selected schema's governance policies; confirm those with the owner before rehearsal.

# COMMAND ----------
# Linux Databricks Python runs notebook cells on the main thread; SIGALRM bounds SDK retry/backoff as well as reads.
import signal

class TraceReadbackDeadline(BaseException):
    # Bypass network retry handlers so they cannot swallow the verification deadline.
    pass

def readback_timeout(signum, frame):
    raise TraceReadbackDeadline('Trace readback exceeded 60 seconds; inspect the saved trace IDs and warehouse access')

previous_handler = signal.signal(signal.SIGALRM, readback_timeout)
signal.alarm(60)
try:
    mlflow.flush_trace_async_logging()
    success_trace = mlflow.get_trace(success_trace_id)
    failure_trace = mlflow.get_trace(failure_trace_id)
    assert success_trace is not None and failure_trace is not None
    assert success_trace.data.spans and failure_trace.data.spans
    assert str(success_trace.info.status).split('.')[-1] == 'OK'
    assert str(failure_trace.info.status).split('.')[-1] == 'ERROR'
    expected = {'Process video event', 'Decode video', 'Sample frames', 'Preprocess images',
                'Run model inference', 'Aggregate predictions'}
    assert expected <= {s.name for s in success_trace.data.spans}
    display(pd.DataFrame([{'name': s.name, 'span_type': str(s.span_type), 'status': str(s.status)}
                          for s in success_trace.data.spans]))
finally:
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous_handler)
for label, trace_id in [('success', success_trace_id), ('failure', failure_trace_id)]:
    url = f'{workspace.config.host.rstrip("/")}/ml/experiments/{experiment.experiment_id}/traces/{trace_id.rsplit("/", 1)[-1]}'
    displayHTML(f'<a href="{url}" target="_blank">Open verified {label} trace</a>')
print(f'Predictions, resolved model version and trace IDs saved in {OUTPUT}')

# COMMAND ----------
# MAGIC %md
# MAGIC Preparation evidence: the champion-loading and scoring cell was validated separately on Azure Serverless CPU,
# MAGIC resolving ResNet-50 version 1 and persisting predictions for 2 events / 20 frames. Local MLflow readback
# MAGIC verified a successful trace with 20 spans and a failed-input trace with 2 spans.
# MAGIC Azure UC trace export and readback also passed in `iangagnon_azure_one_env.isaac_workshop`, using champion
# MAGIC version 1. The successful Azure trace contains 18 spans with the default batch size 8; the deliberate failure
# MAGIC was verified as ERROR. Saved IDs: `/Volumes/iangagnon_azure_one_env/isaac_workshop/videos/workshop_outputs/eb8f7b68995d/04_trace_ids.json`.
# MAGIC [Verified success trace](https://adb-7405609708766806.6.azuredatabricks.net/ml/experiments/3449939268991008/traces/05ce7939a8f8b11d00629fbeef0e55c0)
# MAGIC · [Verified deliberate failure](https://adb-7405609708766806.6.azuredatabricks.net/ml/experiments/3449939268991008/traces/6bd3cb1ac617b7c42f6bd12148cb839d).
# MAGIC GPU trace execution remains unvalidated because this workspace has no eligible GPU region.
