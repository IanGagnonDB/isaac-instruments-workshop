# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Sequential vs Ray vs Spark + iterator Pandas UDF
# MAGIC Sequential is a mock reference, not ISAAC's measured VM performance. No speedup is assumed.
# MAGIC Run correctness probes before timing; retain failures and unsupported paths in the compatibility report.
# MAGIC
# MAGIC **Choose a compute mode explicitly:**
# MAGIC | Mode | Sequential | Ray | Spark UDF | Comparison |
# MAGIC |---|---|---|---|---|
# MAGIC | `serverless_cpu` | CPU driver | unavailable in this supported configuration | CPU executor | unmatched diagnostics |
# MAGIC | `serverless_gpu` | CUDA attached node | AI Runtime `ray_init()` | probe separately; not assumed to have CUDA | unmatched diagnostics |
# MAGIC | `classic_single_node` | dedicated node | local Ray on same node | Spark local mode on same node | eligible for a matched comparison after probes |
# MAGIC | `ray_on_spark` | driver | `setup_ray_cluster` on executors | executors | unmatched until budgets/topology are explicitly reconciled |
# MAGIC
# MAGIC Current Azure workspace allows only serverless and has no eligible Serverless GPU region. For a complete
# MAGIC three-way matched CPU comparison use a compatible workspace with a dedicated single-node CPU cluster;
# MAGIC for CUDA use a dedicated single-node GPU ML cluster. Equal driver and executor node *types* alone do not
# MAGIC give equal total budgets. This notebook only reports matched speedups after all three probes pass on the
# MAGIC same physical host, CPU/memory configuration and single GPU (or all CPU), with concurrency=partitions=1.
# MAGIC This intentionally small budget gives a fair starting point. Scaling concurrency is a separate experiment.
# MAGIC
# MAGIC Classic single-node configuration: `num_workers=0`, `spark.databricks.cluster.profile=singleNode`,
# MAGIC `spark.master=local[*, 4]`, dedicated access mode, identical node for all paths. For CUDA set
# MAGIC `spark.task.resource.gpu.amount=1` at cluster creation and confirm the UDF has a GPU resource assignment.
# MAGIC For Ray-on-Spark use compatible dedicated classic compute, the Ray package included with DBR ML,
# MAGIC and explicit worker CPU/GPU resources. Release the Ray-on-Spark cluster before running Spark UDFs.
# MAGIC See [Ray on AI Runtime](https://docs.databricks.com/machine-learning/ai-runtime/ray),
# MAGIC [Ray-on-Spark](https://docs.databricks.com/machine-learning/ray/ray-create), and
# MAGIC [Pandas UDFs](https://docs.databricks.com/udf/pandas). Recheck Azure availability, not just AWS examples.

# COMMAND ----------
# MAGIC %run ./01_classify_video $library_only="true"

# COMMAND ----------
from typing import Iterator
import mlflow
from pyspark.sql import functions as F

COMPUTE_MODE = WORKSHOP['compute_mode']
REPEATS = int(parameter('repeats', '3'))
BATCH_SIZES = [int(x) for x in parameter('batch_sizes', '4,16').split(',')]
WORKERS = int(parameter('worker_concurrency', '1'))
PARTITIONS = int(parameter('spark_partitions', '1'))
HOURLY_COST = parameter('attributable_hourly_cost', '')
COST_BASIS = parameter('cost_basis', '')  # Describe included VM, DBU and allocation basis; never invent prices.
if COMPUTE_MODE not in {'serverless_cpu', 'serverless_gpu', 'classic_single_node', 'ray_on_spark'}:
    raise ValueError('Unknown compute_mode')
if REPEATS < 2 or min(BATCH_SIZES) < 1 or WORKERS != 1 or PARTITIONS != 1:
    raise ValueError('Use >=2 repeats, positive batches, and one worker/partition for this bounded baseline')
if bool(HOURLY_COST) != bool(COST_BASIS):
    raise ValueError('Cost requires both attributable_hourly_cost and a documented cost_basis')
if HOURLY_COST and float(HOURLY_COST) < 0:
    raise ValueError('Hourly cost must be nonnegative')
if COMPUTE_MODE == 'classic_single_node':
    if not spark.sparkContext.master.startswith('local'):
        raise RuntimeError('classic_single_node requires Spark local mode on the same physical node')
    if DEVICE == 'cuda' and torch.cuda.device_count() != 1:
        raise RuntimeError('This matched baseline requires exactly one physical GPU')
mlflow.tracing.disable()  # No traces in timed runs; 04 explicitly re-enables tracing.
manifest = read_manifest(ROOT, LIMIT)
checkpoints = prepare_checkpoints(ROOT)  # Outside every timing interval.
manifest_sha = manifest_digest(manifest)
BENCH_OUTPUT = f'{OUTPUT}/02'
compatibility, measurements = [], []


def spark_gpu_device(device):
    if device == 'cuda':
        from pyspark import TaskContext
        resource = TaskContext.get().resources().get('gpu')
        if resource is None or len(resource.addresses) != 1:
            raise RuntimeError('CUDA UDF requires one explicitly assigned Spark GPU resource')
        if not os.environ.get('CUDA_VISIBLE_DEVICES'):
            torch.cuda.set_device(int(resource.addresses[0]))
        if not torch.cuda.is_available():
            raise RuntimeError('Pandas UDF worker has no CUDA access')
    return device


def make_spark_udf(model_path, model_id, revision, device, batch_size, sampling):
    @F.pandas_udf('string')
    def classify(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
        selected_device = spark_gpu_device(device)
        engine, load_seconds = load_engine(model_path, selected_device)
        first = True
        for series in iterator:
            output = []
            # Same event grouping as the other paths, independent of Arrow batch size.
            events = [json.loads(value) for value in series]
            for offset in range(0, len(events), 4):
                result = run_events(events[offset:offset + 4], engine, model_id, revision, batch_size, sampling)
                for i, event in enumerate(result['events']):
                    output.append(json.dumps({
                        'event': event, 'hardware': result['hardware'],
                        # Attribute each chunk's stage time once, not once per event.
                        'stages': result['stages'] if i == 0 else {},
                        'load_seconds': load_seconds if first else 0.0,
                    }))
                    first = False
            yield pd.Series(output)
    return classify


def normalize_packets(packets):
    return {
        'events': [p['event'] for p in packets],
        'stages': {key: sum(p.get('stages', {}).get(key, 0.0) for p in packets) for key in
                   ['decode_seconds', 'preprocess_seconds', 'inference_seconds', 'aggregate_seconds']},
        'workers': list({(p['hardware']['host'], p['hardware']['pid']): p['hardware'] for p in packets}.values()),
        'load_seconds': sum(p.get('load_seconds', 0.0) for p in packets),
    }


def initialize_ray():
    if COMPUTE_MODE == 'serverless_cpu':
        raise RuntimeError('Ray on Serverless CPU is not a validated/supported execution path here')
    import ray
    if COMPUTE_MODE == 'serverless_gpu':
        try:
            from databricks.air import ray_init
        except ImportError:
            from serverless_gpu import ray_init  # AI Runtime environment v5
        ray_init()
    elif COMPUTE_MODE == 'classic_single_node':
        if not ray.is_initialized():
            ray.init(num_cpus=WORKERS, num_gpus=WORKERS if DEVICE == 'cuda' else 0,
                     include_dashboard=False)
    elif COMPUTE_MODE == 'ray_on_spark':
        from ray.util.spark import setup_ray_cluster
        import inspect
        parameters = inspect.signature(setup_ray_cluster).parameters
        # Ray 2.49 uses max/min_worker_nodes and *_worker_node; older DBR integrations use *_per_node.
        resources = {'num_cpus_head_node': 0, 'num_gpus_head_node': 0, 'strict_mode': True}
        if 'max_worker_nodes' in parameters:
            resources.update(max_worker_nodes=WORKERS, min_worker_nodes=WORKERS,
                             num_cpus_worker_node=1, num_gpus_worker_node=1 if DEVICE == 'cuda' else 0)
        else:
            resources.update(num_worker_nodes=WORKERS, num_cpus_per_node=1,
                             num_gpus_per_node=1 if DEVICE == 'cuda' else 0)
        setup_ray_cluster(**resources)
        ray.init()
    else:
        raise RuntimeError('Ray on Serverless CPU is not a validated/supported execution path here')
    return ray


def build_backend(path, model_path, model_id, revision, batch_size):
    # Return run + cleanup. Each backend is shut down before the next claims the same budget.
    device, sampling = DEVICE, dict(SAMPLING)
    if path == 'sequential':
        engine, load_seconds = load_engine(model_path, device)
        warm_engine(engine, batch_size)
        def run(events, destination):
            packets = []
            for offset in range(0, len(events), 4):
                r = run_events(events[offset:offset + 4], engine, model_id, revision, batch_size, sampling)
                packets.extend({'event': e, 'hardware': r['hardware'],
                                'stages': r['stages'] if i == 0 else {}, 'load_seconds': 0.0}
                               for i, e in enumerate(r['events']))
            save_predictions(destination + '/predictions.jsonl', [p['event'] for p in packets])
            return normalize_packets(packets)
        def cleanup():
            nonlocal engine
            engine = None
            clear_engines()
        return run, cleanup, load_seconds
    if path == 'ray':
        ray = initialize_ray()
        @ray.remote(num_cpus=1, num_gpus=1 if device == 'cuda' else 0)
        class InferenceWorker:
            def __init__(self):
                self.engine, self.load_seconds = load_engine(model_path, device)
                warm_engine(self.engine, batch_size)
            def ready(self):
                return {'load_seconds': self.load_seconds, 'hardware': hardware(device)}
            def predict(self, events):
                return run_events(events, self.engine, model_id, revision, batch_size, sampling)
        try:
            actor = InferenceWorker.remote()
            info = ray.get(actor.ready.remote(), timeout=300)
        except Exception:
            ray.shutdown()
            if COMPUTE_MODE == 'ray_on_spark':
                from ray.util.spark import shutdown_ray_cluster
                shutdown_ray_cluster()
            raise
        def run(events, destination):
            packets = []
            for offset in range(0, len(events), 4):
                r = ray.get(actor.predict.remote(events[offset:offset + 4]), timeout=300)
                packets.extend({'event': e, 'hardware': r['hardware'],
                                'stages': r['stages'] if i == 0 else {}, 'load_seconds': 0.0}
                               for i, e in enumerate(r['events']))
            save_predictions(destination + '/predictions.jsonl', [p['event'] for p in packets])
            return normalize_packets(packets)
        def cleanup():
            ray.kill(actor)
            ray.shutdown()
            if COMPUTE_MODE == 'ray_on_spark':
                from ray.util.spark import shutdown_ray_cluster
                shutdown_ray_cluster()
        return run, cleanup, info['load_seconds']
    if path == 'spark':
        classify = make_spark_udf(model_path, model_id, revision, device, batch_size, sampling)
        def run(events, destination):
            source = spark.createDataFrame([(json.dumps(e),) for e in events], 'event_json string').repartition(PARTITIONS)
            scored = source.select(classify('event_json').alias('payload'))
            # Each destination is scoped to a fresh run UUID. Serverless supports overwrite, not errorifexists.
            # The completed write forces UDF execution; count() could prune the prediction column.
            scored.write.mode('overwrite').parquet(destination + '/spark_predictions')
            # Read persisted output, never recompute the UDF when checking correctness.
            packets = [json.loads(r.payload) for r in spark.read.parquet(destination + '/spark_predictions').collect()]
            return normalize_packets(packets)
        return run, lambda: None, None  # Loading is observed inside executor packets.
    raise ValueError(path)


def coverage(result):
    records = {}
    for event in result['events']:
        for frame in event['predictions']:
            key = (event['instance_id'], frame['frame_id'])
            if key in records:
                raise AssertionError('Duplicate event/frame key')
            records[key] = (frame['class_id'], frame['probability'])
    return records


def verify(reference, actual, expected_instances):
    a, b = coverage(reference), coverage(actual)
    assert set(e['instance_id'] for e in actual['events']) == set(expected_instances), 'Event coverage mismatch'
    assert len(actual['events']) == len(expected_instances), 'Duplicate event outputs'
    assert a.keys() == b.keys() and len(a) > 0, 'Frame coverage mismatch'
    assert all(a[k][0] == b[k][0] for k in a), 'Top-1 predictions differ'
    max_delta = max(abs(a[k][1] - b[k][1]) for k in a)
    assert max_delta <= 1e-4, f'Confidence max absolute difference {max_delta:.6g} exceeds 1e-4'


def same_budget(worker_reports):
    if COMPUTE_MODE != 'classic_single_node':
        return False
    fields = ['host', 'logical_cpus', 'memory_bytes', 'device', 'gpu', 'gpu_memory_bytes', 'torch_threads',
              'dependencies', 'torch_default_dtype', 'matmul_precision', 'mkldnn_enabled',
              'dnnl_fpmath_mode', 'onednn_fpmath_mode']
    fingerprints = {json.dumps({k: r[k] for k in fields}, sort_keys=True) for r in worker_reports}
    return len(fingerprints) == 1 and bool(worker_reports)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Probe, warm up, then measure
# MAGIC All paths read the same manifest and staged checkpoints, use FP32, one Torch/OpenCV thread per worker,
# MAGIC strict oneDNN CPU math, identical frame rules and batch sizes. Four events form a bounded input chunk; inference batches span events.
# MAGIC Each path first processes the whole manifest as its correctness probe/warm-up, including a completed write.
# MAGIC Measured rounds then run the same workload. OS/Volume caches cannot be flushed on Serverless: the declared
# MAGIC cache policy is **warm best effort**, not a controlled cold-cache test. Downloads are excluded.
# MAGIC
# MAGIC Initialization is measured separately. Full-pipeline wall time includes scheduling, decode, preprocessing,
# MAGIC transfer/inference, aggregation and completed output persistence/readback. Inference timers synchronize CUDA
# MAGIC and include H2D/D2H and softmax. Stage durations are worker time; they are not additive wall time at concurrency >1.
# MAGIC Spark may replace Python workers: a measured round with model loading is labeled cold and excluded from
# MAGIC steady-state summaries. Ray-on-Spark teardown avoids competing with the following Spark job.

# COMMAND ----------
for model_id, revision in REVISIONS.items():
    for batch_size in BATCH_SIZES:
        reference = None
        reports = []
        successful_paths = []
        group_rows = []
        for path in ['sequential', 'ray', 'spark']:
            cleanup = lambda: None
            probe = None
            try:
                clear_engines()
                started = time.perf_counter()
                run, cleanup, model_load = build_backend(path, checkpoints[model_id], model_id, revision, batch_size)
                initialization_seconds = time.perf_counter() - started
                warmup_started = time.perf_counter()
                probe = run(manifest, f'{BENCH_OUTPUT}/{model_id.split("/")[-1]}/b{batch_size}/{path}/warmup')
                if reference is None:
                    reference = probe
                verify(reference, probe, [e['instance_id'] for e in manifest])
                reports.extend(probe['workers'])
                compatibility.append({'model': model_id, 'batch_size': batch_size, 'path': path,
                                      'status': 'passed', 'workers': probe['workers'],
                                      'initialization_seconds': initialization_seconds,
                                      'warmup_pipeline_seconds': time.perf_counter() - warmup_started,
                                      'initial_model_load_worker_seconds': model_load if model_load is not None else probe['load_seconds']})
                for repeat in range(REPEATS):
                    synchronize(DEVICE)
                    started = time.perf_counter()
                    result = run(manifest, f'{BENCH_OUTPUT}/{model_id.split("/")[-1]}/b{batch_size}/{path}/r{repeat}')
                    synchronize(DEVICE)
                    elapsed = time.perf_counter() - started
                    verify(reference, result, [e['instance_id'] for e in manifest])
                    reports.extend(result['workers'])
                    frames = len(coverage(result))
                    group_rows.append({
                        'model': model_id, 'revision': revision, 'batch_size': batch_size, 'path': path, 'repeat': repeat,
                        'full_pipeline_seconds': elapsed, 'frames_per_second': frames / elapsed,
                        'events_per_minute': len(manifest) * 60 / elapsed, 'frames': frames, 'events': len(manifest),
                        'worker_load_seconds': result['load_seconds'], 'steady': result['load_seconds'] == 0,
                        'initialization_seconds': initialization_seconds, **result['stages'],
                        'worker_concurrency': WORKERS, 'spark_partitions': PARTITIONS, 'device': DEVICE,
                        'precision': 'fp32', 'cache_policy': 'warm_best_effort', 'manifest_sha256': manifest_sha,
                        'cost_per_event': float(HOURLY_COST) * elapsed / 3600 / len(manifest) if HOURLY_COST else None,
                        'cost_basis': COST_BASIS or None,
                    })
                successful_paths.append(path)
            except Exception as exc:
                compatibility.append({'model': model_id, 'batch_size': batch_size, 'path': path,
                                      'status': 'failed_or_unsupported', 'error': f'{type(exc).__name__}: {exc}'[:1500],
                                      'workers': probe['workers'] if probe else []})
                print(f'{model_id} / batch {batch_size} / {path}: {type(exc).__name__}: {exc}')
            finally:
                try:
                    cleanup()
                except Exception as cleanup_error:
                    print(f'Cleanup failed for {path}: {cleanup_error}')
                clear_engines()
                write_json(f'{BENCH_OUTPUT}/compatibility.json', compatibility)
        matched = set(successful_paths) == {'sequential', 'ray', 'spark'} and same_budget(reports)
        for row in group_rows:
            row['comparison'] = 'matched' if matched else 'unmatched_diagnostic'
        measurements.extend(group_rows)
write_json(f'{BENCH_OUTPUT}/compatibility.json', compatibility)
write_json(f'{BENCH_OUTPUT}/measurements.json', measurements)
write_json(f'{BENCH_OUTPUT}/configuration.json', {
    'mode': COMPUTE_MODE, 'dependencies': versions(), 'manifest': manifest, 'manifest_sha256': manifest_sha,
    'sampling': SAMPLING, 'revisions': REVISIONS, 'batches': BATCH_SIZES, 'repeats': REPEATS,
    'worker_concurrency': WORKERS, 'spark_partitions': PARTITIONS, 'cost_basis': COST_BASIS,
    'tracing': False, 'cache_policy': 'warm_best_effort',
})
display(pd.DataFrame(compatibility).drop(columns=['workers'], errors='ignore'))
resources = pd.DataFrame([{
    'model': row['model'], 'batch_size': row['batch_size'], 'path': row['path'],
    'cpu': worker['cpu'], 'logical_cpus': worker['logical_cpus'],
    'reported_memory_gib': worker['memory_bytes'] / 2**30, 'device': worker['device'],
    'gpu': worker['gpu'], 'torch_threads': worker['torch_threads'],
    'cpu_math': worker.get('dnnl_fpmath_mode'),
} for row in compatibility for worker in row.get('workers', [])])
if not resources.empty:
    display(resources)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Compare only correct, warm results
# MAGIC Error bars show run-to-run standard deviation. Speedup is shown only for validated matched groups.
# MAGIC Unmatched throughput/runtime charts remain useful diagnostics and are labeled accordingly.
# MAGIC Worker stage times are shown separately from full-pipeline wall time and initialization.
# MAGIC Null cost means no attributable price was supplied. Startup billing and idle allocations require separate attribution.

# COMMAND ----------
import matplotlib.pyplot as plt
results = pd.DataFrame(measurements)
if results.empty:
    raise RuntimeError(f'No execution path completed. Inspect {BENCH_OUTPUT}/compatibility.json')
display(results.groupby(['model', 'batch_size', 'path']).agg(
    measured_runs=('repeat', 'count'), warm_runs=('steady', 'sum'),
    model_load_worker_seconds=('worker_load_seconds', 'sum')).reset_index())
steady = results[results.steady].copy()
if steady.empty:
    print('No warm rounds: model loading occurred in every round. Rehearse worker reuse before comparing performance.')
else:
    group = ['model', 'batch_size', 'path', 'comparison']
    summary = steady.groupby(group).agg(
        runtime_mean=('full_pipeline_seconds', 'mean'), runtime_std=('full_pipeline_seconds', 'std'),
        fps_mean=('frames_per_second', 'mean'), fps_std=('frames_per_second', 'std'),
        events_per_minute=('events_per_minute', 'mean'), warm_runs=('repeat', 'count'),
        cost_per_event=('cost_per_event', 'mean'), initialization_seconds=('initialization_seconds', 'mean'),
    ).reset_index()
    summary['speedup_vs_sequential'] = np.nan
    for (model_id, batch_size), indices in summary.groupby(['model', 'batch_size']).groups.items():
        subset = summary.loc[indices]
        if (set(subset.path) == {'sequential', 'ray', 'spark'}
                and (subset.comparison == 'matched').all() and (subset.warm_runs >= 2).all()):
            baseline = float(subset.loc[subset.path == 'sequential', 'runtime_mean'].iloc[0])
            summary.loc[indices, 'speedup_vs_sequential'] = baseline / subset.runtime_mean
    display(summary)
    for (model_id, batch_size), subset in steady.groupby(['model', 'batch_size']):
        means = subset.groupby('path').mean(numeric_only=True)
        std = subset.groupby('path').std(numeric_only=True).fillna(0)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        means.frames_per_second.plot.bar(ax=axes[0], yerr=std.frames_per_second, title='Frames/s ± run SD')
        means.full_pipeline_seconds.plot.bar(ax=axes[1], yerr=std.full_pipeline_seconds, title='Full pipeline wall seconds')
        means[['decode_seconds', 'preprocess_seconds', 'inference_seconds', 'aggregate_seconds']].plot.bar(
            ax=axes[2], stacked=True, title='Stage worker seconds (not wall time)')
        comparison = subset.comparison.iloc[0]
        fig.suptitle(f'{model_id} · batch={batch_size} · {DEVICE} · workers={WORKERS}, partitions={PARTITIONS} · {comparison}')
        plt.tight_layout()
        plt.show()
    eligible = summary.dropna(subset=['speedup_vs_sequential'])
    if not eligible.empty:
        eligible.assign(label=lambda d: d.model + ' / b' + d.batch_size.astype(str) + ' / ' + d.path).plot.bar(
            x='label', y='speedup_vs_sequential', title='Matched speedup vs Sequential (higher is faster)', legend=False)
        plt.axhline(1, color='gray', linestyle='--')
        plt.show()
    else:
        print('No matched three-way group: speedup withheld. See compatibility report; do not infer GPU speedup.')
    summary.to_csv(f'{BENCH_OUTPUT}/summary.csv', index=False)
print(f'Fallback artifacts for rehearsal: {BENCH_OUTPUT}')

# COMMAND ----------
# MAGIC %md
# MAGIC Local preparation verification: real Ray 2.49.2 CPU actors and Sequential produced consistent
# MAGIC predictions for both pinned models at batch sizes 4 and 16, including 7-frame short clips and metadata IDs.
# MAGIC This is local path verification, not an Azure Ray/GPU benchmark or a matched three-way performance result.

# COMMAND ----------
# MAGIC %md
# MAGIC Serverless numerical preflight found identical decoded images and preprocessed tensor hashes in the driver
# MAGIC and UDF, but the driver inherited `DNNL_DEFAULT_FPMATH_MODE=BF16`. Shared setup explicitly selects `STRICT`
# MAGIC before Torch initialization; tensor dtype alone does not establish FP32 execution. The original `1e-4`
# MAGIC confidence gate remains in place. Different Serverless driver/UDF resource budgets still prevent matched speedups.

# COMMAND ----------
# MAGIC %md
# MAGIC Final Azure CPU rehearsal passed Sequential/Spark output consistency for both pinned models and batch sizes
# MAGIC 4 and 16, with 2 measured repetitions per path/configuration: 16 warm measured rounds in total.
# MAGIC Throughput, variation, stage timings, configuration and compatibility artifacts are pre-staged at
# MAGIC `/Volumes/iangagnon_azure_one_env/isaac_workshop/videos/workshop_outputs/56e92001473f/02/`.
# MAGIC [Open the verified rehearsal](https://adb-7405609708766806.6.azuredatabricks.net/?o=7405609708766806#job/149377190855082/run/554334902757020).
# MAGIC These are unmatched Serverless diagnostics; speedup and cost fields remain blank. Ray CPU consistency was
# MAGIC verified locally on both models/batch sizes. A matched three-way run still requires compatible compute.
