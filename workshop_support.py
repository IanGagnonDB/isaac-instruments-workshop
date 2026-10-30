"""Standard-library setup helpers shared by the Git Folder notebooks."""
import csv
import hashlib
import json
import re
from pathlib import Path


def participant_id(username):
    label = re.sub(r'[^a-z0-9_]', '_', username.split('@')[0].lower())[:24]
    return (label or 'participant') + '_' + hashlib.sha256(username.encode()).hexdigest()[:8]


def load_config(repository, username):
    config = json.loads((Path(repository) / 'workshop_config.json').read_text())
    for key in ('catalog', 'schema', 'volume', 'trace_warehouse_id'):
        if not config.get(key) or config[key].startswith('CHOOSE_'):
            raise ValueError(f'Edit workshop_config.json: set {key} to an accessible workspace resource.')
    for key in ('catalog', 'schema', 'volume'):
        if any(c in config[key] for c in './\\`'):
            raise ValueError(f'{key} must be a single identifier without dots, slashes or backticks')
    person = participant_id(username)
    config.update(participant=person,
                  root=f"/Volumes/{config['catalog']}/{config['schema']}/{config['volume']}",
                  experiment=f'/Users/{username}/isaac_workshop',
                  model_prefix=f"{config['catalog']}.{config['schema']}.isaac_{person}",
                  trace_schema=f"{config['catalog']}.{config['schema']}")
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
            from download_d2_city import run_workshop
            prepare = run_workshop
        prepare(root=str(root), copies=copies, workers=workers)
        status = dataset_status(root)
        if status['state'] != 'ready':
            raise RuntimeError(f'Preparation did not produce a valid dataset: {status}')
        return status
    finally:
        lock.rmdir()
