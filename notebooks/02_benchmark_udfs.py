# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Benchmark Python and pandas UDFs
# MAGIC Compare three CPU Spark UDF implementations on the same video events and pinned model.
# MAGIC Select **Serverless → ML base environment → version 6**. Then **Run All**.
# MAGIC The install cell adds notebook dependencies; the ML base supplies Torch, torchvision, transformers,
# MAGIC NumPy and pandas. The next cell restarts Python before imports. No manual dependency-panel editing.

# COMMAND ----------
# MAGIC %pip install "opencv-python-headless==4.11.0.86"

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_configuration

# COMMAND ----------
import json
import time
from typing import Iterator
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from pyspark.sql import functions as F
from pyspark import cloudpickle
from utils import inference as inf
from utils import benchmark
from utils.benchmark import predict_packets, verify_predictions
from utils.session import create_session

ctx = create_session(WORKSHOP)
# Ship helper function definitions with UDFs; workers need neither a Git clone nor imports from utils/.
cloudpickle.register_pickle_by_value(inf)
cloudpickle.register_pickle_by_value(benchmark)
mlflow.tracing.disable()

for name, default in [('repeats', '3'), ('partitions', '2'), ('inference_batch_sizes', '1,2')]:
    try:
        dbutils.widgets.get(name)
    except Exception:
        dbutils.widgets.text(name, default)
REPEATS = int(dbutils.widgets.get('repeats'))
PARTITIONS = int(dbutils.widgets.get('partitions'))
BATCH_SIZES = [int(value) for value in dbutils.widgets.get('inference_batch_sizes').split(',')]
if REPEATS < 2 or PARTITIONS < 1 or not BATCH_SIZES or min(BATCH_SIZES) < 1 or max(BATCH_SIZES) > 8:
    raise ValueError('Use >=2 repeats, >=1 partition, and inference batches of 1–8 for the CPU starter.')
print(f'Dataset: {ctx.dataset}; model: {ctx.model_id}; events: {len(ctx.manifest)}')

# COMMAND ----------
# MAGIC %md
# MAGIC ## Three ways to process the same events
# MAGIC | UDF | Input to each call | Model lifetime | Frame batching |
# MAGIC |---|---|---|---|
# MAGIC | Python | One video event | Cached within a reused Python worker | Within one event |
# MAGIC | Scalar pandas | A Series of events | Same worker cache, checked each call | Across up to two events |
# MAGIC | Iterator pandas | An iterator of Series | Loaded before the iterator loop | Across up to two events |
# MAGIC
# MAGIC All implementations reuse the same decoding, preprocessing and FP32 model logic. None reloads weights
# MAGIC deliberately per row. Iterator UDFs are not guaranteed to outperform scalar pandas UDFs with worker caching.
# MAGIC **Inference batch size** controls frames passed to Torch; it does not control the Arrow batch size.
# MAGIC Serverless manages Arrow batching. We process at most two events at a time to bound decoded-image memory.

# COMMAND ----------
def make_predictors(model_path, model_id, revision, batch_size, sampling):
    # Capture only plain configuration and serializable helpers, never the loaded model or notebook session.
    load_engine, infer = inf.load_engine, inf.run_events

    def python_predict(value):
        engine, load_seconds = load_engine(model_path)
        return predict_packets([value], engine, load_seconds, model_id, revision, batch_size, sampling, infer)[0]

    def pandas_predict(values: pd.Series) -> pd.Series:
        engine, load_seconds = load_engine(model_path)
        return pd.Series(predict_packets(values.tolist(), engine, load_seconds, model_id, revision,
                                         batch_size, sampling, infer))

    def iterator_predict(batches: Iterator[pd.Series]) -> Iterator[pd.Series]:
        engine, load_seconds = load_engine(model_path)
        for values in batches:
            yield pd.Series(predict_packets(values.tolist(), engine, load_seconds, model_id, revision,
                                            batch_size, sampling, infer))
            if len(values):
                load_seconds = 0.0

    return {'python': python_predict, 'pandas': pandas_predict, 'iterator_pandas': iterator_predict}


def make_udfs(batch_size):
    predictors = make_predictors(ctx.model_path, ctx.model_id, ctx.revision, batch_size, ctx.sampling)
    return {
        'python': F.udf(predictors['python'], 'string', useArrow=False),
        'pandas': F.pandas_udf(predictors['pandas'], 'string'),
        'iterator_pandas': F.pandas_udf(predictors['iterator_pandas'], 'string'),
    }

# COMMAND ----------
# MAGIC %md
# MAGIC ## Establish correctness before comparing speed
# MAGIC The sequential reference is scored in the notebook process and is not a hardware-matched speed baseline.
# MAGIC Every UDF must preserve all event/frame IDs, dataset labels, top-1 classes and confidence within tolerance.
# MAGIC Model weights and videos are already staged in the Volume; no UDF accesses the internet.

# COMMAND ----------
engine, _ = inf.load_engine(ctx.model_path)
reference = inf.run_events(ctx.manifest, engine, ctx.model_id, ctx.revision, 1, ctx.sampling)['events']
del engine
inf.clear_engines()
source = spark.createDataFrame([(json.dumps(event),) for event in ctx.manifest], 'event_json string').repartition(PARTITIONS)
expected_ids = [event['instance_id'] for event in ctx.manifest]
measurements, probes = [], []
inf.write_json(f'{ctx.output}/02_configuration.json', {
    'dataset': ctx.dataset, 'manifest': ctx.manifest, 'manifest_sha256': inf.manifest_digest(ctx.manifest),
    'model_id': ctx.model_id, 'revision': ctx.revision, 'sampling': ctx.sampling,
    'partitions': PARTITIONS, 'inference_batch_sizes': BATCH_SIZES, 'repeats': REPEATS,
    'dependencies': inf.versions(), 'cache_policy': 'warm_best_effort',
    'timing': 'Spark scheduling through completed prediction write; readback excluded',
})


def execute(udf, destination):
    started = time.perf_counter()
    scored = source.select(udf('event_json').alias('payload'))
    # Completing this write forces model inference. count() could prune the UDF column entirely.
    scored.write.mode('overwrite').parquet(destination)
    elapsed = time.perf_counter() - started
    packets = [json.loads(row.payload) for row in spark.read.parquet(destination).collect()]
    verify_predictions(reference, [packet['event'] for packet in packets], expected_ids)
    return elapsed, packets

# COMMAND ----------
# MAGIC %md
# MAGIC ## Warm up, then repeat in rotating order
# MAGIC First-use rounds include worker/model initialization and a completed write. Repeated rounds may still
# MAGIC initialize replacement workers; model-load times are recorded rather than assuming warm workers.
# MAGIC Wall time includes scheduling, I/O, decode, preprocessing, inference and output writing, but not readback.
# MAGIC Worker stage times sum CPU work and are not a wall-time decomposition when workers run concurrently.
# MAGIC Serverless resources and caches are not fixed: these are observed results, not guaranteed hardware speedups.
# MAGIC Keep the small default workload for smoke testing; increase events only after all correctness probes pass.

# COMMAND ----------
for batch_size in BATCH_SIZES:
    udfs = make_udfs(batch_size)
    names = list(udfs)
    for name in names:
        try:
            elapsed, packets = execute(udfs[name], f'{ctx.output}/02/b{batch_size}/{name}/first_use')
        except Exception as exc:
            probes.append({'udf': name, 'batch_size': batch_size, 'status': 'failed',
                           'error': f'{type(exc).__name__}: {exc}'[:2000]})
            inf.write_json(f'{ctx.output}/02_correctness.json', probes)
            raise RuntimeError(f'{name} correctness/preflight failed. Inspect {ctx.output}/02_correctness.json') from exc
        probes.append({'udf': name, 'batch_size': batch_size, 'status': 'passed',
                       'first_use_seconds': elapsed,
                       'model_load_worker_seconds': sum(p['model_load_seconds'] for p in packets)})
    inf.write_json(f'{ctx.output}/02_correctness.json', probes)
    for repeat in range(REPEATS):
        # Rotate order to reduce systematic first/last-path bias.
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        for name in order:
            elapsed, packets = execute(udfs[name], f'{ctx.output}/02/b{batch_size}/{name}/r{repeat}')
            frames = sum(len(p['event']['predictions']) for p in packets)
            measurements.append({
                'udf': name, 'batch_size': batch_size, 'repeat': repeat, 'dataset_kind': WORKSHOP['dataset'],
                'seconds': elapsed, 'events_per_minute': len(ctx.manifest) * 60 / elapsed,
                'frames_per_second': frames / elapsed, 'events': len(ctx.manifest), 'frames': frames,
                'model_load_worker_seconds': sum(p['model_load_seconds'] for p in packets),
                'worker_stages': {stage: sum(p['stages'].get(stage, 0.0) for p in packets)
                                  for stage in ['decode_seconds', 'preprocess_seconds', 'inference_seconds', 'aggregate_seconds']},
                'workers': list({json.dumps(p['hardware'], sort_keys=True) for p in packets}),
            })
            inf.write_json(f'{ctx.output}/02_measurements.json', measurements)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Compare results
# MAGIC Treat the starter sample as a functional test; scheduling overhead can dominate a few short clips.
# MAGIC Increase the workload for meaningful throughput measurements. Repeated copies add load, not data diversity.
# MAGIC Re-run with more events, partitions or inference batch sizes one setting at a time. Do not assume more
# MAGIC partitions are faster. Model selection can optionally compare ResNet-18 with ResNet-50 in a separate run.

# COMMAND ----------
results = pd.DataFrame(measurements)
summary = results.groupby(['batch_size', 'udf']).agg(
    mean_seconds=('seconds', 'mean'), std_seconds=('seconds', 'std'),
    mean_frames_per_second=('frames_per_second', 'mean'),
    std_frames_per_second=('frames_per_second', 'std'),
    mean_events_per_minute=('events_per_minute', 'mean'),
    mean_model_load_worker_seconds=('model_load_worker_seconds', 'mean'),
)
display(summary.reset_index())
display(pd.DataFrame(probes))
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
summary['mean_frames_per_second'].unstack('udf').plot.bar(
    yerr=summary['std_frames_per_second'].unstack('udf').fillna(0), ax=axes[0], title='Observed frames / second')
summary['mean_seconds'].unstack('udf').plot.bar(
    yerr=summary['std_seconds'].unstack('udf').fillna(0), ax=axes[1], title='Completed pipeline wall time (seconds)')
fig.suptitle(f"CPU Serverless | {WORKSHOP['dataset']} | {ctx.model_id} | {len(ctx.manifest)} events")
plt.tight_layout()
plt.show()
print(f'All three UDFs passed correctness checks. Evidence: {ctx.output}')
