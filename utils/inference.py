"""CPU video decoding and inference shared by the teaching notebooks and UDF workers."""
import builtins
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
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


def versions():
    packages = ['torch', 'torchvision', 'transformers', 'opencv-python-headless', 'numpy', 'pillow',
                'mlflow', 'pandas', 'pyarrow', 'huggingface_hub']
    result = {'python': platform.python_version()}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            if package == 'mlflow':
                try:
                    result[package] = importlib.metadata.version('mlflow-skinny')
                except importlib.metadata.PackageNotFoundError:
                    result[package] = None
            else:
                result[package] = None
    return result


def hardware(device='cpu'):
    import psutil
    import torch
    return {'host': socket.gethostname(), 'logical_cpus': os.cpu_count(),
            'memory_bytes': psutil.virtual_memory().total,
            'process_rss_bytes': psutil.Process().memory_info().rss,
            'device': 'cpu', 'torch_threads': torch.get_num_threads(), 'pid': os.getpid(),
            'dependencies': versions(), 'cpu_math': os.environ.get('DNNL_DEFAULT_FPMATH_MODE')}


def read_manifest(root, limit):
    root = Path(root)
    dataset = json.loads((root / 'dataset.json').read_text())
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
            'dataset_kind': dataset['kind'],
        })
        if not video.is_file():
            raise FileNotFoundError(video)
    if len({e['instance_id'] for e in events}) != len(events):
        raise ValueError('Duplicate manifest instance IDs')
    return events


def manifest_digest(events):
    # Includes sidecar-derived metadata; all execution paths receive these exact serialized events.
    return hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest()


def checkpoint_path(models_root, model_id):
    destination = Path(models_root) / model_id.split('/')[-1] / REVISIONS[model_id]
    for filename in ('config.json', 'preprocessor_config.json', 'model.safetensors'):
        path = destination / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Missing model file {path}. Run optional_prepare_dataset to stage model weights.')
    return str(destination)


def stage_checkpoint(models_root, model_id):
    try:
        return checkpoint_path(models_root, model_id)
    except FileNotFoundError:
        pass
    from huggingface_hub import snapshot_download
    destination = Path(models_root) / model_id.split('/')[-1] / REVISIONS[model_id]
    # Downloads happen only in the driver-side preparation notebook, never in a UDF.
    lock = Path(models_root) / '.preparation.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError('Model preparation is already running, or its lock is stale. Check with the designated preparer.') from None
    try:
        snapshot_download(repo_id=model_id, revision=REVISIONS[model_id], local_dir=str(destination),
                          allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors'])
        return checkpoint_path(models_root, model_id)
    finally:
        lock.rmdir()


def load_engine(model_path, device='cpu', cache=True):
    # Import locally: cloudpickle otherwise captures PyTorch's non-pickleable backend proxy modules.
    import torch
    os.environ['DNNL_DEFAULT_FPMATH_MODE'] = 'STRICT'
    os.environ['ONEDNN_DEFAULT_FPMATH_MODE'] = 'STRICT'
    torch.set_float32_matmul_precision('highest')
    if device != 'cpu':
        raise ValueError('This workshop supports CPU only')
    # A process-local cache avoids reloading weights when a Python worker is reused.
    if not hasattr(builtins, '_isaac_engines'):
        builtins._isaac_engines = {}
    key = (str(model_path), device)
    if cache and key in builtins._isaac_engines:
        return builtins._isaac_engines[key], 0.0
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    started = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True).eval().to(device)
    engine = (processor, model, device)
    if cache:
        builtins._isaac_engines[key] = engine
    return engine, time.perf_counter() - started


def clear_engines():
    if hasattr(builtins, '_isaac_engines'):
        builtins._isaac_engines.clear()
    import gc
    gc.collect()


def warm_engine(engine, batch_size):
    _, model, device = engine
    with torch.inference_mode():
        model(torch.zeros((batch_size, 3, 224, 224), device=device))


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
    # Shared across all CPU execution paths and champion scoring. Batch frames ACROSS events.
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
        'dataset_kind': events[0]['dataset_kind'] if events else None,
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
