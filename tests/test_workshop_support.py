import tempfile
import unittest
from pathlib import Path

import workshop_support as support


class DatasetTests(unittest.TestCase):
    def test_missing_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(support.dataset_status(directory)['state'], 'missing')

    def test_partial_dataset_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark = Path(directory) / 'benchmark'
            benchmark.mkdir()
            (benchmark / 'manifest.csv').write_text('event_id,file_name,source_file,copy_index\nevt_1,missing.mp4,a.mp4,1\n')
            self.assertEqual(support.dataset_status(directory)['state'], 'incomplete')

    def test_complete_dataset_is_reused_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'raw').mkdir()
            (root / 'raw/a.mp4').write_bytes(b'video')
            (root / 'benchmark').mkdir()
            (root / 'benchmark/evt_1.mp4').write_bytes(b'video')
            (root / 'benchmark/manifest.csv').write_text('event_id,file_name,source_file,copy_index\nevt_1,evt_1.mp4,a.mp4,1\n')
            self.assertEqual(support.dataset_status(root)['state'], 'ready')
            result = support.prepare_shared_dataset(root, prepare=lambda **kw: self.fail('download called'))
            self.assertEqual(result['state'], 'ready')

    def test_lock_prevents_second_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / '.dataset-preparation.lock').mkdir()
            with self.assertRaisesRegex(RuntimeError, 'preparation'):
                support.prepare_shared_dataset(directory, prepare=lambda **kw: None)

    def test_failed_preparation_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            def fail(**kwargs):
                raise ValueError('download failed')
            with self.assertRaises(ValueError):
                support.prepare_shared_dataset(directory, prepare=fail)
            self.assertFalse((Path(directory) / '.dataset-preparation.lock').exists())

    def test_reject_path_escape_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'benchmark').mkdir()
            (root / 'benchmark/manifest.csv').write_text('event_id,file_name,source_file,copy_index\nevt_1,../outside.mp4,a.mp4,1\n')
            self.assertEqual(support.dataset_status(root)['state'], 'incomplete')


class ConfigTests(unittest.TestCase):
    def test_identity_is_stable_and_distinct(self):
        self.assertEqual(support.participant_id('a@example.com'), support.participant_id('a@example.com'))
        self.assertNotEqual(support.participant_id('a@example.com'), support.participant_id('b@example.com'))


class FreshWorkspaceTests(unittest.TestCase):
    def test_config_derives_shared_data_and_isolated_models(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            config = {'catalog': 'demo', 'schema': 'workshop', 'volume': 'videos',
                      'trace_warehouse_id': 'warehouse', 'device': 'cpu',
                      'compute_mode': 'serverless_cpu', 'batch_size': 8, 'event_limit': 2}
            (Path(directory) / 'workshop_config.json').write_text(json.dumps(config))
            first = support.load_config(directory, 'first@example.com')
            second = support.load_config(directory, 'second@example.com')
            self.assertEqual(first['root'], '/Volumes/demo/workshop/videos')
            self.assertEqual(first['root'], second['root'])
            self.assertNotEqual(first['model_prefix'], second['model_prefix'])
            self.assertNotEqual(first['experiment'], second['experiment'])

    def test_unconfigured_workspace_fails_with_edit_instruction(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'workshop_config.json').write_text(json.dumps({'catalog': 'CHOOSE_CATALOG'}))
            with self.assertRaisesRegex(ValueError, 'Edit workshop_config.json'):
                support.load_config(directory, 'user@example.com')

    def test_partial_data_never_calls_preparer(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'raw').mkdir()
            (Path(directory) / 'raw/partial.mp4').write_bytes(b'partial')
            with self.assertRaisesRegex(RuntimeError, 'Incomplete dataset'):
                support.prepare_shared_dataset(directory, prepare=lambda **kw: self.fail('must not download'))
            self.assertEqual((Path(directory) / 'raw/partial.mp4').read_bytes(), b'partial')

    def test_prepare_missing_dataset_then_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            def prepare(root, copies, workers):
                calls.append((copies, workers))
                base = Path(root)
                (base / 'raw').mkdir()
                (base / 'raw/a.mp4').write_bytes(b'video')
                (base / 'benchmark').mkdir()
                (base / 'benchmark/evt_1.mp4').write_bytes(b'video')
                (base / 'benchmark/manifest.csv').write_text('event_id,file_name,source_file,copy_index\nevt_1,evt_1.mp4,a.mp4,1\n')
            self.assertEqual(support.prepare_shared_dataset(directory, prepare=prepare)['state'], 'ready')
            self.assertEqual(support.prepare_shared_dataset(directory, prepare=prepare)['state'], 'ready')
            self.assertEqual(calls, [(1, 4)])


if __name__ == '__main__':
    unittest.main()
