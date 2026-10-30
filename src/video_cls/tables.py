"""Table naming shared by the benchmark tasks (which run in parallel) and the downstream notebooks."""
import re

# bench_cluster is appended to by several clusters at once. It is created up front with a fixed
# schema, because Delta blind appends never conflict but concurrent CREATEs can.
BENCH_CLUSTER_SCHEMA = (
    "run_ts timestamp, run_label string, num_workers int, total_cores int, task_cpus int, "
    "node_type string, model string, variant string, backend string, n_partitions int, batch_size int, "
    "ort_threads int, clips bigint, frames bigint, extract_s double, inference_s double, "
    "inference_frames_per_s double, end_to_end_frames_per_s double"
)


def bench_cluster_df(spark, rows):
    """bench_cluster rows (dicts) -> DataFrame with exactly BENCH_CLUSTER_SCHEMA, in column order."""
    columns = [c.strip().split(" ")[0] for c in BENCH_CLUSTER_SCHEMA.split(",")]
    return spark.createDataFrame([tuple(row[c] for c in columns) for row in rows], BENCH_CLUSTER_SCHEMA)


def label_suffix(run_label):
    """'4w_task_cpus_2' -> '4w_task_cpus_2'; anything else becomes a safe identifier fragment."""
    suffix = re.sub(r"[^0-9a-zA-Z_]", "_", run_label or "").strip("_").lower()
    if not suffix:
        raise ValueError("run_label is required so parallel benchmark tasks write separate tables")
    return suffix


def ensure_table(spark, name, schema_ddl):
    """CREATE TABLE IF NOT EXISTS that tolerates another cluster creating it at the same moment."""
    try:
        spark.sql(f"CREATE TABLE IF NOT EXISTS {name} ({schema_ddl})")
    except Exception as exc:                             # lost the race: fine if it now exists
        if not spark.catalog.tableExists(name):
            raise exc


def find_table(spark, catalog, schema, base):
    """Return <base> if it exists, else the first <base>_<label> table (sorted), else None.

    The benchmark writes frames_<label> / frame_predictions_<label>, one per cluster.
    Every cluster extracts the same frames, so any of them can feed scoring.
    """
    full = f"{catalog}.{schema}.{base}"
    if spark.catalog.tableExists(full):
        return full
    names = sorted(r.tableName for r in spark.sql(f"SHOW TABLES IN `{catalog}`.`{schema}` LIKE '{base}_*'").collect())
    return f"{catalog}.{schema}.{names[0]}" if names else None
