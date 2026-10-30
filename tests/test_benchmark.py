import unittest


class CorrectnessTests(unittest.TestCase):
    def setUp(self):
        from utils.benchmark import verify_predictions
        self.verify = verify_predictions
        self.event = {'instance_id': 'one', 'dataset_kind': 'd2_city', 'predictions': [
            {'frame_id': 'one:0', 'class_id': 2, 'probability': 0.75}]}

    def test_matching_outputs_pass(self):
        self.verify([self.event], [self.event], ['one'])

    def test_missing_event_fails(self):
        with self.assertRaises(AssertionError):
            self.verify([self.event], [], ['one'])

    def test_duplicate_event_fails(self):
        with self.assertRaises(AssertionError):
            self.verify([self.event], [self.event, self.event], ['one'])

    def test_prediction_mismatch_fails(self):
        bad = {**self.event, 'predictions': [{'frame_id': 'one:0', 'class_id': 3, 'probability': 0.75}]}
        with self.assertRaises(AssertionError):
            self.verify([self.event], [bad], ['one'])

    def test_dataset_mix_fails(self):
        with self.assertRaises(AssertionError):
            self.verify([self.event], [{**self.event, 'dataset_kind': 'synthetic'}], ['one'])
