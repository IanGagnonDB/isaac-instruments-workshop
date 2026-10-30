"""Bounded event grouping and correctness checks; no Spark context or model downloads."""
import json
import math


def predict_packets(values, engine, load_seconds, model_id, revision, batch_size, sampling, infer):
    output = []
    # Keep decoded images bounded independently of the platform-managed Arrow batch size.
    for offset in range(0, len(values), 2):
        events = [json.loads(value) for value in values[offset:offset + 2]]
        result = infer(events, engine, model_id, revision, batch_size, sampling)
        for index, event in enumerate(result['events']):
            output.append(json.dumps({
                'event': event, 'hardware': result['hardware'],
                'stages': result['stages'] if index == 0 else {},
                'model_load_seconds': load_seconds if not output else 0.0,
            }))
    return output


def verify_predictions(reference, actual, expected_instances):
    expected = set(expected_instances)
    if len(actual) != len(expected_instances) or {e['instance_id'] for e in actual} != expected:
        raise AssertionError('Missing or duplicate event outputs')

    def coverage(events):
        records = {}
        for event in events:
            if not event['predictions']:
                raise AssertionError('Event produced no frame predictions')
            for frame in event['predictions']:
                key = event['instance_id'], frame['frame_id']
                if key in records:
                    raise AssertionError('Duplicate event/frame key')
                records[key] = event['dataset_kind'], frame['class_id'], frame['probability']
        return records

    first, second = coverage(reference), coverage(actual)
    if not first or first.keys() != second.keys():
        raise AssertionError('Frame coverage mismatch')
    for key in first:
        kind, label, probability = first[key]
        actual_kind, actual_label, actual_probability = second[key]
        if kind != actual_kind or label != actual_label:
            raise AssertionError('Dataset or top-1 prediction differs')
        if not math.isfinite(actual_probability) or abs(probability - actual_probability) > 1e-4:
            raise AssertionError('Confidence difference exceeds 1e-4')
