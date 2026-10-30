"""Log the benchmark tables as an MLflow experiment: one run per measured configuration.

Read-only over bench_single_node, bench_cluster and model_exports: nothing is re-measured.
Each source gets a parent run with a summary chart and one nested child run per configuration,
so the MLflow UI can compare, sort and chart them. Parent runs are tagged with the newest
run_ts they cover; logging the same benchmark run twice is skipped.
"""
import mlflow
from matplotlib.figure import Figure

SINGLE_PARAMS = ["model", "variant", "backend", "batch_size", "threads", "node_type", "frames"]
SINGLE_METRICS = ["ms_per_frame", "frames_per_s"]
EXPORT_METRICS = ["top1_agreement_vs_torch", "size_mb"]
CLUSTER_PARAMS = ["run_label", "num_workers", "total_cores", "task_cpus", "node_type", "model", "variant",
                  "backend", "n_partitions", "batch_size", "ort_threads", "clips", "frames"]
CLUSTER_METRICS = ["extract_s", "inference_s", "inference_frames_per_s", "end_to_end_frames_per_s"]


def latest_single_node(single):
    return single[single.run_ts == single.run_ts.max()]


def latest_cluster(cluster):
    """Latest run per cluster configuration (run_label), as in 06_report."""
    return cluster[cluster.run_ts == cluster.groupby("run_label").run_ts.transform("max")]


def _existing_parent(experiment_id, benchmark, source_run_ts, table_name):
    found = mlflow.search_runs([experiment_id], max_results=1, filter_string=(
        f"tags.benchmark = '{benchmark}' AND tags.source_run_ts = '{source_run_ts}' "
        f"AND tags.source_table = '{table_name}'"))           # only parent runs carry source_table
    return None if found.empty else found.run_id.iloc[0]


def _log_children(experiment_id, rows, benchmark, source_run_ts, name, params, metrics):
    for _, row in rows.iterrows():
        with mlflow.start_run(experiment_id=experiment_id, run_name=name(row), nested=True):
            mlflow.set_tags({"benchmark": benchmark, "source_run_ts": source_run_ts})
            mlflow.log_params({p: row[p] for p in params})
            mlflow.log_metrics({m: float(row[m]) for m in metrics if m in row and row[m] == row[m]})  # skip NaN


def _log_benchmark(experiment_id, benchmark, rows, table_name, figure, name, params, metrics):
    """Parent run + one child per row. Returns (parent_run_id, newly_logged)."""
    source_run_ts = rows.run_ts.max().isoformat()
    existing = _existing_parent(experiment_id, benchmark, source_run_ts, table_name)
    if existing:
        return existing, False
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{benchmark} benchmark") as parent:
        mlflow.set_tags({"benchmark": benchmark, "source_run_ts": source_run_ts, "source_table": table_name})
        mlflow.log_params({"configurations": len(rows), "node_type": rows.node_type.iloc[0]})
        mlflow.log_figure(figure, f"{benchmark}.png")
        _log_children(experiment_id, rows, benchmark, source_run_ts, name, params, metrics)
    return parent.info.run_id, True


def log_single_node(experiment_id, single, exports, table_name):
    """bench_single_node (latest run) joined with model_exports' agreement and size per model/variant."""
    rows = latest_single_node(single).merge(
        exports[["model", "variant"] + EXPORT_METRICS], on=["model", "variant"], how="left")
    fig = Figure(figsize=(9, 4))
    ax = fig.subplots()
    table = rows[rows.batch_size == rows.batch_size.max()].pivot_table(
        index="model", columns="variant", values="ms_per_frame")
    table.plot.bar(ax=ax, rot=0)
    ax.set_ylabel("ms / frame (lower is better)")
    ax.set_title(f"Single node, batch {rows.batch_size.max()}: torch = PyTorch, fp32/int8 = ONNX Runtime")
    return _log_benchmark(experiment_id, "single_node", rows, table_name, fig,
                          lambda r: f"{r.model} {r.variant} bs{r.batch_size}",
                          SINGLE_PARAMS, SINGLE_METRICS + EXPORT_METRICS)


def log_cluster(experiment_id, cluster, table_name):
    """bench_cluster (latest run per run_label), one child per cluster x model/variant."""
    rows = latest_cluster(cluster)
    fig = Figure(figsize=(9, 4.5))
    ax = fig.subplots()
    scaling = rows[rows.task_cpus == 1].sort_values("total_cores")
    for (model, variant), g in scaling.groupby(["model", "variant"]):
        ax.plot(g.total_cores, g.inference_frames_per_s, marker="o", label=f"{model} {variant}")
    ax.set_xlabel("total worker cores")
    ax.set_ylabel("inference frames / s")
    ax.set_title("Inference throughput vs cores")
    ax.legend(fontsize=8)
    return _log_benchmark(experiment_id, "cluster", rows, table_name, fig,
                          lambda r: f"{r.run_label} {r.model} {r.variant}",
                          CLUSTER_PARAMS, CLUSTER_METRICS)
