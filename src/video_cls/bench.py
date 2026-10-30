"""Benchmark helpers: single-node timing loop, cluster shape, and parameter parsing."""
import os
import time

import numpy as np

from video_cls.models import VARIANTS


def parse_runs(spec):
    """'resnet18:fp32,resnet50:int8' -> [('resnet18', 'fp32'), ('resnet50', 'int8')]."""
    runs = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        model, _, variant = item.partition(":")
        if variant not in VARIANTS:
            raise ValueError(f"{item!r}: variant must be one of {VARIANTS}")
        runs.append((model, variant))
    if not runs:
        raise ValueError("No model:variant runs given")
    return runs


def time_predictor(predictor, frames, batch_size, n_frames=500, warmup_batches=3):
    """Warm up, then time n_frames of preprocessed input. Returns (seconds, frames_timed)."""
    for _ in range(warmup_batches):
        predictor(frames[:batch_size])
    idx = np.arange(n_frames) % len(frames)
    started = time.perf_counter()
    for start in range(0, n_frames, batch_size):
        predictor(frames[idx[start:start + batch_size]])
    return time.perf_counter() - started, n_frames


def cpu_count():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


def wait_for_workers(spark, n_workers, timeout_s=900, poll_s=10):
    """Block until n_workers executors have registered, so timing uses the full cluster.

    Job clusters can start the notebook before every worker has joined.
    """
    sc = spark.sparkContext
    deadline = time.monotonic() + timeout_s
    while True:
        executors = sc._jsc.sc().getExecutorMemoryStatus().size() - 1   # minus the driver
        if executors >= n_workers or n_workers == 0:
            return executors
        if time.monotonic() > deadline:
            raise TimeoutError(f"Only {executors}/{n_workers} workers registered after {timeout_s}s")
        time.sleep(poll_s)


def cluster_shape(spark):
    """Total worker cores, task slots and node type for the current classic cluster."""
    conf = spark.sparkContext.getConf()
    total_cores = spark.sparkContext.defaultParallelism
    task_cpus = int(conf.get("spark.task.cpus", "1"))
    return {
        "total_cores": total_cores,
        "task_cpus": task_cpus,
        "task_slots": max(1, total_cores // task_cpus),
        "node_type": spark.conf.get("spark.databricks.clusterUsageTags.clusterNodeType", "unknown"),
    }


class Stopwatch:
    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.started
