# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Register, promote and roll back
# MAGIC Register two versions of the same pinned checkpoint and verify alias rollback. Re-running adds versions intentionally.
# MAGIC Select **Serverless → ML base environment → version 6**. Then **Run All**.
# MAGIC The install cell adds notebook dependencies; the ML base supplies Torch, torchvision, transformers,
# MAGIC NumPy and pandas. The next cell restarts Python before imports. No manual dependency-panel editing.

# COMMAND ----------
# MAGIC %pip install "opencv-python-headless==4.11.0.86" "mlflow==3.17.2"

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_configuration

# COMMAND ----------
from utils import inference as inf
from utils.session import create_session

ctx = create_session(WORKSHOP)
print(f'Dataset: {ctx.dataset}; output: {ctx.output}')

import os
import json
import time
import importlib.metadata
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from transformers import AutoImageProcessor
from utils.inference import (load_engine, sampled_frames, read_manifest, run_events, warm_engine,
                             versions, write_json, save_predictions, REVISIONS)

ROOT, OUTPUT, DEVICE = ctx.root, ctx.output, 'cpu'
BATCH_SIZE, LIMIT, SAMPLING = ctx.batch_size, WORKSHOP['event_limit'], ctx.sampling
EXPERIMENT, MODEL_PREFIX = WORKSHOP['experiment'], WORKSHOP['model_prefix']
RUN_ID = Path(ctx.output).name


def parameter(name, default):
    try:
        return dbutils.widgets.get(name)
    except Exception:
        dbutils.widgets.text(name, str(default))
        return str(default)

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
checkpoints = {ctx.model_id: ctx.model_path}
registered_names = {model_id: MODEL_PREFIX + '_' + model_id.split('/')[-1].replace('-', '') for model_id in checkpoints}
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
        mlflow.set_tags({'workshop': 'isaac', 'dataset_kind': WORKSHOP['dataset'], 'task': 'imagenet_frame_classification', 'accuracy_claim': 'none'})
        mlflow.log_dict(runtime, 'dependencies.json')
        mlflow.log_dict({'model_id': model_id, 'revision': REVISIONS[model_id], 'sampling': SAMPLING, 'dataset': ctx.dataset,
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
for model_id in checkpoints:
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

primary = ctx.model_id
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
            'revision': REVISIONS[model_id]} for model_id in checkpoints]
write_json(f'{OUTPUT}/03_registry.json', {'models': summary, 'candidate_version': candidate['version'],
                                        'rollback_version': previous, 'dependencies': versions(), 'dataset': ctx.dataset})
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
