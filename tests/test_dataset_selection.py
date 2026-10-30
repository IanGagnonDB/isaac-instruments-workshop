import tempfile
import unittest
from pathlib import Path
from utils import workshop_support as support


class SelectionTests(unittest.TestCase):
    def test_failed_real_download_never_creates_synthetic_data(self):
        with tempfile.TemporaryDirectory() as directory:
            def fail(**kwargs):
                raise OSError('network unavailable')
            with self.assertRaisesRegex(RuntimeError, 'Use synthetic fallback'):
                support.prepare_selected_dataset(directory, 'd2_city', 'Download D2-City', prepare_real=fail)
            self.assertFalse((Path(directory) / 'datasets/synthetic').exists())

    def test_fallback_requires_explicit_source_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'synthetic'):
                support.prepare_selected_dataset(directory, 'd2_city', 'Use synthetic fallback')
            self.assertFalse((Path(directory) / 'datasets/synthetic').exists())

    def test_reuse_never_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'not ready'):
                support.prepare_selected_dataset(directory, 'd2_city', 'Reuse existing data',
                                                 prepare_real=lambda **kwargs: self.fail('download called'))

    def test_synthetic_configuration_does_not_change_real_paths(self):
        real = support.build_config({'volume_root': '/Volumes/c/s/v'}, 'user')
        fake = support.build_config({'volume_root': '/Volumes/c/s/v', 'dataset': 'synthetic'}, 'user')
        self.assertNotEqual(real['root'], fake['root'])
        self.assertEqual(real['models_root'], fake['models_root'])

    def test_unknown_dataset_rejected(self):
        with self.assertRaises(ValueError):
            support.build_config({'volume_root': '/Volumes/c/s/v', 'dataset': 'automatic'}, 'user')


class SyntheticTests(unittest.TestCase):
    def test_generated_videos_decode_and_are_labeled_then_reused(self):
        try:
            import cv2
        except ImportError:
            self.skipTest('OpenCV required for video integration test; use .venv/bin/python')
        with tempfile.TemporaryDirectory() as directory:
            real = Path(directory) / 'datasets/d2_city'
            real.mkdir(parents=True)
            sentinel = real / 'existing.txt'
            sentinel.write_text('real data is untouched')
            result = support.prepare_selected_dataset(directory, 'synthetic', 'Use synthetic fallback')
            self.assertEqual(result['state'], 'ready')
            self.assertEqual(result['dataset']['kind'], 'synthetic')
            for video in (Path(directory) / 'datasets/synthetic/benchmark').glob('*.mp4'):
                cap = cv2.VideoCapture(str(video))
                try:
                    count = 0
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        count += 1
                    self.assertEqual(count, 24)
                finally:
                    cap.release()
            self.assertEqual(sentinel.read_text(), 'real data is untouched')
            second = support.prepare_selected_dataset(directory, 'synthetic', 'Reuse existing data')
            self.assertEqual(second['dataset'], result['dataset'])
