"""Standard-library setup helpers shared by the Git Folder notebooks."""
import csv
import hashlib
import json
import re
from pathlib import Path


def participant_id(username):
    label = re.sub(r'[^a-z0-9_]', '_', username.split('@')[0].lower())[:24]
    return (label or 'participant') + '_' + hashlib.sha256(username.encode()).hexdigest()[:8]


def build_config(settings, username):
    volume = str(settings.get('volume_root', '')).rstrip('/')
    parts = Path(volume).parts
    if (len(parts) != 5 or parts[:2] != ('/', 'Volumes') or
            any('CHOOSE' in part or part in ('.', '..') or any(c in part for c in '.`\\')
                for part in parts[2:])):
        raise ValueError('Edit VOLUME_ROOT in 00_configuration: /Volumes/<catalog>/<schema>/<volume>')
    source = settings.get('dataset', 'd2_city')
    if source not in ('d2_city', 'synthetic'):
        raise ValueError('DATASET must be d2_city or synthetic; automatic fallback is not supported')
    config = dict(event_limit=4, frames_per_event=3, batch_size=2,
                  model_id='microsoft/resnet-18', trace_warehouse_id='')
    config.update(settings)
    for key in ('event_limit', 'frames_per_event', 'batch_size'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer in 00_configuration')
    if config['model_id'] not in ('microsoft/resnet-18', 'microsoft/resnet-50'):
        raise ValueError('Choose microsoft/resnet-18 or microsoft/resnet-50')
    person = participant_id(username)
    catalog, schema, _ = parts[2:]
    config.update(volume_root=volume, dataset=source, participant=person,
                  root=f'{volume}/datasets/{source}', models_root=f'{volume}/workshop_models',
                  output_root=f'{volume}/workshop_outputs/{person}',
                  experiment=f'/Users/{username}/isaac_workshop',
                  model_prefix=f'{catalog}.{schema}.isaac_{person}',
                  trace_schema=f'{catalog}.{schema}')
    return config


def dataset_status(root):
    """Check every manifest reference before allowing reuse; never repair/delete data."""
    root = Path(root)
    manifest = root / 'benchmark/manifest.csv'
    if not manifest.exists():
        populated = any(p.exists() and (not p.is_dir() or any(p.iterdir()))
                        for p in (root / 'raw', root / 'benchmark'))
        return {'state': 'incomplete' if populated else 'missing', 'reason': 'No benchmark/manifest.csv'}
    try:
        with manifest.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if not {'event_id', 'file_name', 'source_file', 'copy_index'} <= set(reader.fieldnames or []):
                raise ValueError('Manifest columns do not match the downloader contract')
            count, ids = 0, set()
            for row in reader:
                if not row['event_id'] or row['event_id'] in ids:
                    raise ValueError('Empty or duplicate event ID')
                ids.add(row['event_id'])
                if int(row['copy_index']) < 1:
                    raise ValueError('Invalid copy index')
                for folder, column in [('benchmark', 'file_name'), ('raw', 'source_file')]:
                    relative = Path(row[column])
                    if not row[column] or relative.is_absolute() or '..' in relative.parts:
                        raise ValueError('Unsafe manifest path')
                    path = root / folder / relative
                    if not path.is_file() or path.stat().st_size == 0:
                        raise ValueError(f'Missing or empty video: {path}')
                count += 1
            if not count:
                raise ValueError('Manifest is empty')
        return {'state': 'ready', 'events': count, 'manifest': str(manifest)}
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        return {'state': 'incomplete', 'reason': str(exc)}


def prepare_shared_dataset(root, copies=1, workers=4, prepare=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.dataset-preparation.lock'
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError('Dataset preparation is already in progress. Ask the designated preparer to finish.') from None
    try:
        status = dataset_status(root)
        if status['state'] == 'ready':
            return status
        if status['state'] == 'incomplete':
            raise RuntimeError(f"Incomplete dataset: {status['reason']}. Choose a fresh Volume or inspect partial data; nothing was deleted.")
        if prepare is None:
            from utils.download_d2_city import run_workshop
            prepare = run_workshop
        prepare(root=str(root), copies=copies, workers=workers)
        status = dataset_status(root)
        if status['state'] != 'ready':
            raise RuntimeError(f'Preparation did not produce a valid dataset: {status}')
        return status
    finally:
        lock.rmdir()


def selected_dataset_status(volume_root, source):
    if source not in ('d2_city', 'synthetic'):
        raise ValueError('Choose d2_city or synthetic')
    root = Path(volume_root) / 'datasets' / source
    status = dataset_status(root)
    if status['state'] == 'ready':
        try:
            provenance = json.loads((root / 'dataset.json').read_text())
            if provenance['kind'] != source:
                raise ValueError('Dataset provenance does not match the selected source')
            status['dataset'] = provenance
        except (OSError, ValueError, KeyError, TypeError) as exc:
            status = {'state': 'incomplete', 'reason': f'Dataset provenance missing/invalid: {exc}'}
    return status


def prepare_selected_dataset(volume_root, source, action, prepare_real=None):
    """Never switch datasets in an exception handler. Source and action are both explicit."""
    if source not in ('d2_city', 'synthetic'):
        raise ValueError('Choose d2_city or synthetic in 00_configuration')
    expected = {'Download D2-City': 'd2_city', 'Use synthetic fallback': 'synthetic'}
    if action != 'Reuse existing data' and expected.get(action) != source:
        raise ValueError(f'Action {action!r} requires DATASET={expected.get(action)!r} in 00_configuration; '
                         'select synthetic explicitly for the fallback.')
    if action == 'Reuse existing data':
        status = selected_dataset_status(volume_root, source)
        if status['state'] != 'ready':
            raise RuntimeError(f'Dataset not ready: {status}. Run optional_prepare_dataset with the appropriate action.')
        return status
    root = Path(volume_root) / 'datasets' / source
    marker = root / 'dataset.json'
    if marker.exists():
        provenance = json.loads(marker.read_text())
        if provenance.get('kind') != source:
            raise RuntimeError('Existing dataset provenance conflicts with the selected source; inspect it before proceeding.')

    def prepare(root, copies, workers):
        if source == 'synthetic':
            from utils.synthetic_data import generate_dataset
            generate_dataset(root)
        else:
            if prepare_real is None:
                from utils.download_d2_city import run_workshop
                callback = run_workshop
            else:
                callback = prepare_real
            callback(root=root, copies=copies, workers=workers)

    try:
        prepare_shared_dataset(root, copies=1, workers=4, prepare=prepare)
    except Exception as exc:
        if source == 'd2_city':
            raise RuntimeError(f'D2-City preparation failed: {type(exc).__name__}: {exc}. '
                               'No fallback was selected. To proceed explicitly, set DATASET="synthetic" '
                               'in 00_configuration and choose "Use synthetic fallback" in optional_prepare_dataset. '
                               'Synthetic fallback replaces videos only; model weights still need staging.') from exc
        raise
    if not marker.exists():
        provenance = {'kind': source, 'synthetic': source == 'synthetic',
                      'description': 'Generated moving shapes; predictions have no semantic meaning' if source == 'synthetic'
                      else 'D2-City source videos; repeated copies add load, not diversity'}
        marker.write_text(json.dumps(provenance, indent=2))
    return selected_dataset_status(volume_root, source)
