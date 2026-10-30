# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · From an event video to frame predictions
# MAGIC Clone the repository as a Git Folder. Start with 00_setup_and_preflight, then run 01 → 02 → 03 → 04.
# MAGIC No training or app is involved. ImageNet labels describe objects; they do **not** measure driving severity.
# MAGIC
# MAGIC **Environment:** Python 3.12, PyTorch + compatible torchvision (use the GPU runtime's CUDA build),
# MAGIC `transformers==4.57.1`, `opencv-python-headless==4.11.0.86`, `mlflow==3.17.2`, pandas, pyarrow,
# MAGIC matplotlib, psutil, huggingface_hub; Ray is needed only for 02. On Serverless CPU, tested job dependencies
# MAGIC are `torch==2.7.1`, `torchvision==0.22.1` plus the packages above. Configure dependencies in the notebook
# MAGIC Environment panel or job environment **before** running; do not replace GPU PyTorch with a CPU wheel.
# MAGIC CPU rehearsal pins: `numpy==2.1.3`, `pillow==11.1.0`, `pandas==2.2.3`, `pyarrow==19.0.1`,
# MAGIC `huggingface-hub==0.36.2`, `matplotlib==3.10.7`, `psutil==7.1.2`; Serverless environment client `4`.
# MAGIC Use these with the pinned CPU Torch/torchvision and model/MLflow packages above.
# MAGIC Pin the environment after rehearsal. Start a fresh Python session so STRICT CPU math is configured before
# MAGIC Torch/oneDNN initializes; restart Python after changing precision settings. Model weights stay in the Volume, outside Git.
# MAGIC
# MAGIC **Data contract:** the existing `download_d2_city.py` has already populated `raw/` and `benchmark/`.
# MAGIC Read `benchmark/manifest.csv`: `event_id,file_name,source_file,copy_index`. Do not rerun the downloader.
# MAGIC `event_id` from the manifest remains the unique workload `instance_id`; if a source JSON sidecar supplies
# MAGIC an event ID, preserve it as `event_id` as well. Copies share a source event but have distinct instance IDs.
# MAGIC Sidecars are JSON objects at `raw/<source_file with .json suffix>`; supported fields are `event_id` and
# MAGIC `event_type`. Adapt `read_manifest` for ISAAC's actual schema. Missing sidecars use manifest IDs and `default`.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# Shared definitions are imported by the other notebooks using %run with library_only=true.
import builtins
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
# Serverless may inherit BF16 oneDNN math even when tensors are float32. Set this BEFORE importing Torch.
os.environ['DNNL_DEFAULT_FPMATH_MODE'] = 'STRICT'
os.environ['ONEDNN_DEFAULT_FPMATH_MODE'] = 'STRICT'
import torch
torch.set_float32_matmul_precision('highest')
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

REVISIONS = {
    'microsoft/resnet-50': '34c2154c194f829b11125337b98c8f5f9965ff19',
    'microsoft/resnet-18': '65a5785d9156231087c481e0c7dd33a5ff6f7e3e',
}
SAMPLING = {'default': 10, 'snapshot': 1}


def parameter(name, default):
    try:
        return dbutils.widgets.get(name)
    except Exception:
        dbutils.widgets.text(name, str(default))
        return str(default)


ROOT = WORKSHOP['root']
DEVICE = WORKSHOP['device']  # Explicit: cpu or cuda; never silently fall back.
BATCH_SIZE = int(WORKSHOP['batch_size'])
LIMIT = int(WORKSHOP['event_limit'])
LIBRARY_ONLY = parameter('library_only', 'false').lower() == 'true'
EXPERIMENT = WORKSHOP['experiment']
MODEL_PREFIX = WORKSHOP['model_prefix']
if DEVICE not in ('cpu', 'cuda') or BATCH_SIZE < 1 or LIMIT < 1:
    raise ValueError('device must be cpu/cuda; batch_size and event_limit must be positive')
RUN_ID = uuid.uuid4().hex[:12]
OUTPUT = str(Path(ROOT) / 'workshop_outputs' / WORKSHOP['participant'] / RUN_ID)


def versions():
    packages = ['torch', 'torchvision', 'transformers', 'opencv-python-headless', 'numpy', 'pillow',
                'mlflow', 'pandas', 'pyarrow', 'ray', 'huggingface_hub']
    result = {'python': platform.python_version()}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def hardware(device):
    import psutil
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable in this process; choose CPU explicitly')
    return {
        'host': socket.gethostname(), 'cpu': platform.processor(), 'logical_cpus': os.cpu_count(),
        'memory_bytes': psutil.virtual_memory().total, 'process_rss_bytes': psutil.Process().memory_info().rss,
        'device': device,
        'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
        'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory if device == 'cuda' else 0,
        'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
        'gpu_reserved_bytes': torch.cuda.memory_reserved() if device == 'cuda' else 0,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch_threads': torch.get_num_threads(), 'pid': os.getpid(),
        'dependencies': versions(), 'torch_default_dtype': str(torch.get_default_dtype()),
        'matmul_precision': torch.get_float32_matmul_precision(),
        'mkldnn_enabled': torch.backends.mkldnn.enabled,
        'dnnl_fpmath_mode': os.environ.get('DNNL_DEFAULT_FPMATH_MODE'),
        'onednn_fpmath_mode': os.environ.get('ONEDNN_DEFAULT_FPMATH_MODE'),
    }


def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def read_manifest(root, limit):
    root = Path(root)
    with (root / 'benchmark/manifest.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {'event_id', 'file_name', 'source_file', 'copy_index'} <= set(rows[0]):
        raise ValueError('Expected the manifest emitted by download_d2_city.prepare_benchmark')
    events = []
    for row in rows[:limit]:
        source = (root / 'raw' / row['source_file']).resolve()
        video = (root / 'benchmark' / row['file_name']).resolve()
        if not source.is_relative_to((root / 'raw').resolve()) or not video.is_relative_to((root / 'benchmark').resolve()):
            raise ValueError('Manifest paths must stay within raw/ and benchmark/')
        sidecar = source.with_suffix('.json')
        metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        if not isinstance(metadata, dict):
            raise ValueError(f'Expected a JSON object in {sidecar}')
        events.append({
            'instance_id': row['event_id'], 'event_id': str(metadata.get('event_id', row['event_id'])),
            'event_type': str(metadata.get('event_type', 'default')), 'path': str(video),
            'source_file': row['source_file'], 'copy_index': int(row['copy_index']),
        })
        if not video.is_file():
            raise FileNotFoundError(video)
    if len({e['instance_id'] for e in events}) != len(events):
        raise ValueError('Duplicate manifest instance IDs')
    return events


def manifest_digest(events):
    # Includes sidecar-derived metadata; all execution paths receive these exact serialized events.
    return hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest()


def prepare_checkpoints(root):
    # Preparation happens once in 00; scoring never starts a network download.
    paths = {}
    for model_id, revision in REVISIONS.items():
        destination = Path(root) / 'workshop_models' / model_id.split('/')[-1] / revision
        for filename in ('config.json', 'preprocessor_config.json', 'model.safetensors'):
            path = destination / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f'Missing model file {path}. Run optional model preparation in notebook 00.')
        paths[model_id] = str(destination)
    return paths


def load_engine(model_path, device, cache=True):
    # Import locally: cloudpickle otherwise captures PyTorch's non-pickleable backend proxy modules.
    import torch
    os.environ['DNNL_DEFAULT_FPMATH_MODE'] = 'STRICT'
    os.environ['ONEDNN_DEFAULT_FPMATH_MODE'] = 'STRICT'
    torch.set_float32_matmul_precision('highest')
    # A process-local cache survives Arrow batches/UDF invocations and Ray actor calls.
    if not hasattr(builtins, '_isaac_engines'):
        builtins._isaac_engines = {}
    key = (str(model_path), device)
    if cache and key in builtins._isaac_engines:
        return builtins._isaac_engines[key], 0.0
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable inside inference worker')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    started = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True).eval().to(device)
    synchronize(device)
    engine = (processor, model, device)
    if cache:
        builtins._isaac_engines[key] = engine
    return engine, time.perf_counter() - started


def clear_engines():
    if hasattr(builtins, '_isaac_engines'):
        builtins._isaac_engines.clear()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def warm_engine(engine, batch_size):
    _, model, device = engine
    with torch.inference_mode():
        model(torch.zeros((batch_size, 3, 224, 224), device=device))
    synchronize(device)


def sampled_frames(event, sampling, span=nullcontext):
    # Seek directly to selected frame indices: memory is bounded by the sampled frames, not the clip.
    # ISAAC replaces sampling rules here. Roughly 10 samples from a 30-second event is the default.
    cap = cv2.VideoCapture(event['path'])
    try:
        with span('Inspect video'):
            if not cap.isOpened():
                raise ValueError(f"Cannot open video for instance {event['instance_id']}")
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if count < 1 or not np.isfinite(fps) or fps <= 0:
                raise ValueError('Invalid frame count or FPS')
        with span('Sample frames'):
            requested = int(sampling.get(event['event_type'], sampling['default']))
            if requested < 1:
                raise ValueError('Sampling count must be positive')
            indices = np.unique(np.linspace(0, count - 1, min(requested, count), dtype=int)).tolist()
        with span('Decode video'):
            images = []
            for index in indices:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                    raise ValueError(f'Video decoder cannot seek to frame {index}')
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f"Decode failed for {event['instance_id']} at frame {index}")
                images.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        return indices, images, fps
    finally:
        cap.release()


def run_events(events, engine, model_id, revision, batch_size, sampling=None, model_version=None, span=nullcontext):
    # Shared across Sequential, Ray, Spark and champion scoring. Batch frames ACROSS events.
    sampling = sampling or SAMPLING
    processor, model, device = engine
    stage = dict(decode_seconds=0.0, preprocess_seconds=0.0, inference_seconds=0.0, aggregate_seconds=0.0)
    predictions = {e['instance_id']: [] for e in events}
    pending, identities = [], []

    def flush():
        if not pending:
            return
        started = time.perf_counter()
        with span('Run model inference'), torch.inference_mode():
            tensor = torch.stack(pending).to(device)
            output = model(pixel_values=tensor)
            logits = output if isinstance(output, torch.Tensor) else output.logits
            if logits.shape != (len(pending), 1000) or not torch.isfinite(logits).all():
                raise ValueError('Invalid ImageNet logits')
            probabilities = logits.float().softmax(-1)
            confidence, label = probabilities.max(-1)
            confidence, label = confidence.cpu().tolist(), label.cpu().tolist()
            synchronize(device)
        stage['inference_seconds'] += time.perf_counter() - started
        started = time.perf_counter()
        with span('Aggregate predictions'):
            for (event, index, fps), class_id, probability in zip(identities, label, confidence):
                predictions[event['instance_id']].append({
                    'frame_id': f"{event['instance_id']}:{index:06d}", 'frame_index': index,
                    'timestamp_seconds': index / fps, 'class_id': class_id,
                    'label': model.config.id2label[class_id], 'probability': probability,
                })
        stage['aggregate_seconds'] += time.perf_counter() - started
        pending.clear()
        identities.clear()

    for event in events:
        started = time.perf_counter()
        indices, images, fps = sampled_frames(event, sampling, span)
        stage['decode_seconds'] += time.perf_counter() - started
        # Preprocess one image at a time to bound tensor memory and keep the identical recipe in all paths.
        for index, image in zip(indices, images):
            started = time.perf_counter()
            with span('Preprocess images'):
                tensor = processor(images=image, return_tensors='pt')['pixel_values'][0]
                if tensor.shape != (3, 224, 224) or not torch.isfinite(tensor).all():
                    raise ValueError('Invalid preprocessed frame')
            stage['preprocess_seconds'] += time.perf_counter() - started
            pending.append(tensor)
            identities.append((event, index, fps))
            if len(pending) == batch_size:
                flush()
    flush()
    return {
        'events': [{**event, 'model_id': model_id, 'revision': revision, 'model_version': model_version,
                    'predictions': predictions[event['instance_id']]} for event in events],
        'stages': stage, 'hardware': hardware(device),
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def save_predictions(path, events):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        for event in events:
            handle.write(json.dumps(event) + '\n')
    # Confirm that the completed write preserves event/frame identities.
    restored = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(e['instance_id'], len(e['predictions'])) for e in restored] == [
        (e['instance_id'], len(e['predictions'])) for e in events]

# COMMAND ----------
# MAGIC %md
# MAGIC ## Run one event
# MAGIC Model snapshots and videos must already be staged by notebook 00. Scoring never downloads them.
# MAGIC Results are JSONL event records with nested frame predictions, IDs, source path, copy index and checkpoint revision.
# MAGIC `device=cuda` is an assertion: a missing GPU fails clearly rather than changing the experiment.

# COMMAND ----------
if not LIBRARY_ONLY:
    print(json.dumps({'hardware': hardware(DEVICE), 'dependencies': versions()}, indent=2))
    checkpoints = prepare_checkpoints(ROOT)
    manifest = read_manifest(ROOT, LIMIT)
    model_id = 'microsoft/resnet-50'
    engine, load_seconds = load_engine(checkpoints[model_id], DEVICE)
    warm_engine(engine, BATCH_SIZE)
    result = run_events(manifest[:1], engine, model_id, REVISIONS[model_id], BATCH_SIZE)
    save_predictions(f'{OUTPUT}/01_predictions.jsonl', result['events'])
    write_json(f'{OUTPUT}/01_provenance.json', {
        'manifest_sha256': manifest_digest(manifest), 'dependencies': versions(),
        'sampling': SAMPLING, 'load_seconds': load_seconds, 'hardware': result['hardware'],
        'stages': result['stages'], 'model_revisions': REVISIONS,
    })
    display(pd.DataFrame(result['events'][0]['predictions']))
    print(f'Saved predictions and provenance: {OUTPUT}')

# COMMAND ----------
# MAGIC %md
# MAGIC ## Preparation evidence and compute handoff
# MAGIC Before these notebooks were built, synthetic MP4s (including a 7-frame clip) and two real local D2-City
# MAGIC videos were tested with both pinned models. The real clips had 749 and 738 frames at 25 FPS. Each produced
# MAGIC 10 sampled frames, `[20,3,224,224]` inputs and finite `[20,1000]` logits with probabilities summing to one.
# MAGIC CPU and Apple MPS GPU top-1 results agreed on all 20 frames for both models; maximum logit differences
# MAGIC were 0.0000401 (ResNet-50) and 0.0000191 (ResNet-18). This verifies local GPU execution, not NVIDIA CUDA.
# MAGIC
# MAGIC Azure profile `lakebase-azure` was checked on 2026-10-29. Its benchmark manifest contains 1,980 event
# MAGIC instances. Serverless A10 and H100 dry-runs reported no eligible GPU regions. A classic T4 job submission
# MAGIC was rejected: **Only serverless compute is supported in the workspace.** These are platform constraints,
# MAGIC not successful CUDA tests. Revalidate access before scheduling a GPU workshop.
# MAGIC
# MAGIC Git Folder setup: edit workshop_config.json once, then use notebook 00. Configuration includes
# MAGIC root resource identifiers, device, batch_size and event_limit; participant model/experiment paths are derived. Grant Volume READ/WRITE,
# MAGIC experiment access, USE CATALOG/SCHEMA, CREATE MODEL and ownership/manage rights for alias changes.
# MAGIC The selected schema comes from workshop_config.json; model names are isolated by signed-in participant.
# MAGIC Agenda: classify a clip → validate and benchmark execution paths → version/promotion/rollback → trace success/failure.
# MAGIC Weights, outputs, environment details and fallback artifacts live in the Volume; duplicated clips add load, not diversity.

# MAGIC
# MAGIC Azure Serverless CPU preflight also passed on existing Volume clips evt_000001 and evt_000002
# MAGIC (733 and 744 frames at 25 FPS), with both pinned checkpoints and repeated predictions.
# MAGIC Preflight run: https://adb-7405609708766806.6.azuredatabricks.net/?o=7405609708766806#job/299897120037561/run/857003666063024
