"""Local integration checks. Uses existing cached weights only; never downloads."""
import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Iterator


class CPUInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from utils import inference
            import cloudpickle
            import pandas as pd
        except ImportError as exc:
            raise unittest.SkipTest(f'ML dependencies unavailable: {exc}; use the project .venv')
        from utils import benchmark
        from utils.workshop_support import prepare_selected_dataset
        repository = Path(__file__).resolve().parents[1]
        cached = repository / '.cache/huggingface/hub/models--microsoft--resnet-18/snapshots' / inference.REVISIONS['microsoft/resnet-18']
        cls.model_path = os.environ.get('ISAAC_TEST_MODEL', str(cached))
        if not (Path(cls.model_path) / 'model.safetensors').exists():
            raise unittest.SkipTest('No cached checkpoint; set ISAAC_TEST_MODEL to an existing ResNet-18 snapshot')
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        prepare_selected_dataset(cls.directory.name, 'synthetic', 'Use synthetic fallback')
        cls.inf = inference
        cls.benchmark = benchmark
        cls.events = inference.read_manifest(Path(cls.directory.name) / 'datasets/synthetic', 3)
        cls.engine, _ = inference.load_engine(cls.model_path)
        cls.sampling = {'default': 3, 'snapshot': 1}
        cls.model_id = 'microsoft/resnet-18'
        cls.revision = inference.REVISIONS[cls.model_id]
        cls.reference = inference.run_events(cls.events, cls.engine, cls.model_id, cls.revision, 1, cls.sampling)['events']
        # Execute the actual teaching factory without running notebook configuration, pip or Spark cells.
        source = ast.parse((repository / 'notebooks/02_benchmark_udfs.py').read_text())
        factory = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'make_predictors')
        namespace = {'inf': inference, 'predict_packets': benchmark.predict_packets, 'pd': pd, 'Iterator': Iterator}
        exec(compile(ast.Module(body=[factory], type_ignores=[]), '<notebook make_predictors>', 'exec'), namespace)
        cls.predictors = namespace['make_predictors'](cls.model_path, cls.model_id, cls.revision, 2, cls.sampling)
        cloudpickle.register_pickle_by_value(inference)
        cloudpickle.register_pickle_by_value(benchmark)

    def test_sampling_and_dataset_provenance(self):
        self.assertEqual(len(self.reference), 3)
        self.assertTrue(all(event['dataset_kind'] == 'synthetic' for event in self.reference))
        self.assertTrue(all(len(event['predictions']) == 3 for event in self.reference))

    def test_all_three_worker_functions_preserve_predictions(self):
        import pandas as pd
        values = [json.dumps(event) for event in self.events]
        variants = {
            'python': [self.predictors['python'](value) for value in values],
            'pandas': self.predictors['pandas'](pd.Series(values)).tolist(),
            'iterator_pandas': [value for batch in self.predictors['iterator_pandas'](
                iter([pd.Series(values[:2]), pd.Series(values[2:])])) for value in batch],
        }
        for name, packets in variants.items():
            with self.subTest(udf=name):
                self.benchmark.verify_predictions(self.reference, [json.loads(value)['event'] for value in packets],
                                                  [event['instance_id'] for event in self.events])

    def test_predictor_serializes_without_utils_import_on_worker(self):
        import cloudpickle
        import subprocess
        import sys
        payload = Path(self.directory.name) / 'predictor.pkl'
        output = Path(self.directory.name) / 'worker_result.json'
        value = Path(self.directory.name) / 'event.json'
        payload.write_bytes(cloudpickle.dumps(self.predictors['python']))
        value.write_text(json.dumps(self.events[0]))
        script = '''
import importlib.abc, sys, json, cloudpickle
from pathlib import Path
class NoUtils(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'utils' or fullname.startswith('utils.'):
            raise ImportError('utils deliberately unavailable on simulated worker')
sys.meta_path.insert(0, NoUtils())
predict = cloudpickle.loads(Path(sys.argv[1]).read_bytes())
result = predict(Path(sys.argv[2]).read_text())
Path(sys.argv[3]).write_text(result)
'''
        subprocess.run([sys.executable, '-I', '-c', script, str(payload), str(value), str(output)],
                       check=True, cwd=self.directory.name, timeout=60, capture_output=True, text=True)
        actual = [json.loads(output.read_text())['event']]
        self.benchmark.verify_predictions(self.reference[:1], actual, [self.events[0]['instance_id']])
