# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Reproduce, promote and roll back a pretrained model
# MAGIC Log the two pinned checkpoints as separate Unity Catalog registered models, with the same tensor contract:
# MAGIC float32 `[N,3,224,224]` → float32 `[N,1000]` logits. The saved processor defines RGB/resize/crop/normalization.
# MAGIC No training, evaluation dataset, fine-tuning or retraining job is created.
# MAGIC
# MAGIC The ResNet-50 alias exercise registers two versions of the **same checkpoint**, changes `champion`, verifies
# MAGIC inference through the alias, then restores its previous target. This demonstrates mechanics, not improved
# MAGIC accuracy. Architectures are registered separately; do not infer driving-severity accuracy from ImageNet outputs.
# MAGIC `model_prefix` must name a workshop-owned catalog.schema.prefix. Re-running adds versions intentionally.

# COMMAND ----------
# MAGIC %run ./01_classify_video $library_only="true"

# COMMAND ----------
import tempfile
import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

mlflow.set_tracking_uri('databricks')
mlflow.set_registry_uri('databricks-uc')
mlflow.set_experiment(EXPERIMENT)
client = MlflowClient()
checkpoints = prepare_checkpoints(ROOT)
registered_names = {model_id: MODEL_PREFIX + '_' + model_id.split('/')[-1].replace('-', '') for model_id in REVISIONS}
signature = ModelSignature(
    inputs=Schema([TensorSpec(np.dtype('float32'), (-1, 3, 224, 224))]),
    outputs=Schema([TensorSpec(np.dtype('float32'), (-1, 1000))]),
)


class ImageNetLogits(torch.nn.Module):
    """A stable MLflow PyTorch/pyfunc contract around Hugging Face's structured output."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def champion_version(name):
    # Only treat a missing alias as absent; permission/network errors must stop the demonstration.
    from mlflow.exceptions import MlflowException
    try:
        return client.get_model_version_by_alias(name, 'champion').version
    except MlflowException as exc:
        if exc.error_code == 'RESOURCE_DOES_NOT_EXIST':
            return None
        raise


def register_checkpoint(model_id, role):
    engine, load_seconds = load_engine(checkpoints[model_id], 'cpu')
    processor, model, _ = engine
    wrapped = ImageNetLogits(model).eval()
    # Real sampled images exercise the saved contract; a numeric zero tensor alone would miss preprocessing issues.
    sample_event = read_manifest(ROOT, 1)[0]
    _, images, _ = sampled_frames(sample_event, {'default': 1})
    example = processor(images=images[:1], return_tensors='pt')['pixel_values'].numpy().astype('float32')
    with torch.inference_mode():
        expected = wrapped(torch.from_numpy(example)).numpy()
    assert expected.shape == (1, 1000) and np.isfinite(expected).all()
    runtime = versions()
    requirements = [f'{p}=={importlib.metadata.version(p).split("+")[0]}' for p in
                    ['torch', 'torchvision', 'transformers', 'numpy', 'pillow', 'cloudpickle']]
    with mlflow.start_run(run_name=f'{model_id.split("/")[-1]}-{role}-{RUN_ID}') as run:
        mlflow.log_params({'model_id': model_id, 'checkpoint_revision': REVISIONS[model_id],
                           'precision': 'fp32', 'role': role, 'training': False, 'load_seconds': load_seconds})
        mlflow.set_tags({'workshop': 'isaac', 'task': 'imagenet_frame_classification', 'accuracy_claim': 'none'})
        mlflow.log_dict(runtime, 'dependencies.json')
        mlflow.log_dict({'model_id': model_id, 'revision': REVISIONS[model_id], 'sampling': SAMPLING,
                         'input_contract': 'float32 NCHW RGB, processor normalization',
                         'output_contract': 'float32 ImageNet-1k logits; not severity'}, 'provenance.json')
        with tempfile.TemporaryDirectory() as directory:
            processor.save_pretrained(directory)
            mlflow.log_artifacts(directory, artifact_path='processor')
        info = mlflow.pytorch.log_model(
            pytorch_model=wrapped, name='model', signature=signature, input_example=example,
            serialization_format='pickle',
            pip_requirements=requirements, registered_model_name=registered_names[model_id],
            metadata={'model_id': model_id, 'checkpoint_revision': REVISIONS[model_id]},
        )
        version = str(info.registered_model_version)
        if not version or version == 'None':
            raise RuntimeError('Registration returned no version')
        client.set_model_version_tag(registered_names[model_id], version, 'checkpoint_revision', REVISIONS[model_id])
        # Verify the version's packaged inference contract, not just the in-memory model.
        loaded = mlflow.pyfunc.load_model(f'models:/{registered_names[model_id]}/{version}')
        actual = loaded.predict(example)
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
        return {'name': registered_names[model_id], 'version': version, 'run_id': run.info.run_id,
                'example': example, 'expected': expected}

# COMMAND ----------
registered = {}
original_aliases = {}
for model_id in REVISIONS:
    # Query an existing model first; a first-ever model registration has no alias yet.
    from mlflow.exceptions import MlflowException
    try:
        client.get_registered_model(registered_names[model_id])
        original_aliases[model_id] = champion_version(registered_names[model_id])
    except MlflowException as exc:
        if exc.error_code != 'RESOURCE_DOES_NOT_EXIST':
            raise
        original_aliases[model_id] = None
    registered[model_id] = register_checkpoint(model_id, 'baseline')
    if original_aliases[model_id] is None:
        client.set_registered_model_alias(registered_names[model_id], 'champion', registered[model_id]['version'])

primary = 'microsoft/resnet-50'
name = registered_names[primary]
previous = champion_version(name)
candidate = register_checkpoint(primary, 'illustrative_candidate_same_checkpoint')
try:
    client.set_registered_model_alias(name, 'champion', candidate['version'])
    resolved = client.get_model_version_by_alias(name, 'champion')
    assert str(resolved.version) == candidate['version']
    # Pin the version after resolving the alias to prevent an alias race during loading.
    selected = mlflow.pyfunc.load_model(f'models:/{name}/{resolved.version}')
    np.testing.assert_allclose(selected.predict(candidate['example']), candidate['expected'], atol=1e-4, rtol=1e-4)
finally:
    client.set_registered_model_alias(name, 'champion', previous)
assert champion_version(name) == previous, 'Rollback did not restore the previous champion'
restored = mlflow.pyfunc.load_model(f'models:/{name}/{previous}')
restored_logits = restored.predict(candidate['example'])
assert restored_logits.shape == (1, 1000) and np.isfinite(restored_logits).all()
# Compare predictions when rollback targets the version created in this run. Existing champions may use other weights.
if previous == registered[primary]['version']:
    np.testing.assert_allclose(restored_logits, registered[primary]['expected'], atol=1e-4, rtol=1e-4)
summary = [{'model_id': model_id, 'registered_name': registered_names[model_id],
            'new_version': registered[model_id]['version'], 'champion': champion_version(registered_names[model_id]),
            'revision': REVISIONS[model_id]} for model_id in REVISIONS]
write_json(f'{OUTPUT}/03_registry.json', {'models': summary, 'candidate_version': candidate['version'],
                                        'rollback_version': previous, 'dependencies': versions()})
display(pd.DataFrame(summary))
print(f'Alias promotion and rollback verified; evidence: {OUTPUT}/03_registry.json')

# COMMAND ----------
# MAGIC %md
# MAGIC ## Where ISAAC's lifecycle fits
# MAGIC ISAAC substitutes its proprietary model and processor while retaining the documented input/output contract.
# MAGIC A real promotion needs a held-out, representative labeled driving dataset, task-specific quality thresholds,
# MAGIC latency/resource budgets, and approval criteria. Training and experiment tracking would happen upstream;
# MAGIC data drift and quality monitoring can trigger evaluation and, if warranted, retraining. An alias is a reversible
# MAGIC routing decision, not evidence of quality. Architecture comparisons here measure compute, not driving accuracy.
# MAGIC
# MAGIC MLflow's ONNX flavor is another integration route for ISAAC's ONNX artifacts. It is a talk track only here:
# MAGIC no ONNX export/runtime example has been validated, and no ONNX execution is claimed. Online serving and
# MAGIC real-time enrichment are optional extensions outside this workshop.

# COMMAND ----------
# MAGIC %md
# MAGIC Azure rehearsal on 2026-10-29 verified both registered architectures, packaged pyfunc predictions,
# MAGIC ResNet-50 promotion from version 1 to 2, and rollback to version 1. Evidence was saved at
# MAGIC `/Volumes/iangagnon_azure_one_env/isaac_workshop/videos/workshop_outputs/af37d32d9824/03_registry.json`.
