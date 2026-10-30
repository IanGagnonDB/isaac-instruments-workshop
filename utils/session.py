"""Notebook context creation; imports happen after the visible pip/restart cells."""
import uuid
from types import SimpleNamespace
from utils import inference
from utils.workshop_support import selected_dataset_status


def create_session(config):
    status = selected_dataset_status(config['volume_root'], config['dataset'])
    if status['state'] != 'ready':
        raise RuntimeError(f"Selected dataset {config['dataset']} is not ready: {status}. "
                           'Open optional_prepare_dataset. A synthetic fallback must be selected explicitly.')
    model_id = config['model_id']
    return SimpleNamespace(
        root=config['root'], model_id=model_id, revision=inference.REVISIONS[model_id],
        model_path=inference.checkpoint_path(config['models_root'], model_id),
        manifest=inference.read_manifest(config['root'], config['event_limit']),
        sampling={'default': config['frames_per_event'], 'snapshot': 1},
        batch_size=config['batch_size'], dataset=status['dataset'],
        output=f"{config['output_root']}/{uuid.uuid4().hex[:12]}",
    )
