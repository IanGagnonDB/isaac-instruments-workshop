from datetime import datetime, timezone

import pytest

from video_cls.tables import BENCH_CLUSTER_SCHEMA, bench_cluster_df, ensure_table, find_table, label_suffix


def test_label_suffix():
    assert label_suffix("4w_task_cpus_2") == "4w_task_cpus_2"
    assert label_suffix("8W-run.1") == "8w_run_1"
    with pytest.raises(ValueError):
        label_suffix("")


def test_bench_row_matches_fixed_schema_and_appends(spark):
    spark.sql("CREATE DATABASE IF NOT EXISTS t")
    ensure_table(spark, "t.bench_cluster", BENCH_CLUSTER_SCHEMA)
    ensure_table(spark, "t.bench_cluster", BENCH_CLUSTER_SCHEMA)     # second create is a no-op
    row = {"run_ts": datetime.now(timezone.utc), "run_label": "2w", "num_workers": 2, "total_cores": 32,
           "task_cpus": 1, "node_type": "Standard_F16s_v2", "model": "resnet18", "variant": "int8",
           "backend": "onnxruntime", "n_partitions": 96, "batch_size": 64, "ort_threads": 1, "clips": 400,
           "frames": 4000, "extract_s": 61.0, "inference_s": 12.5, "inference_frames_per_s": 320.0,
           "end_to_end_frames_per_s": 54.4}
    for _ in range(2):                                     # same path as 03: fixed schema, blind append
        bench_cluster_df(spark, [row]).write.mode("append").saveAsTable("t.bench_cluster")
    assert spark.table("t.bench_cluster").count() == 2
    spark.sql("DROP TABLE t.bench_cluster")


def test_find_table_prefers_base_then_first_label(spark):
    spark.sql("CREATE DATABASE IF NOT EXISTS spark_catalog.s")
    assert find_table(spark, "spark_catalog", "s", "frames") is None
    for name in ("frames_8w", "frames_2w"):
        spark.sql(f"CREATE TABLE spark_catalog.s.{name} (x int)")
    assert find_table(spark, "spark_catalog", "s", "frames") == "spark_catalog.s.frames_2w"
    spark.sql("CREATE TABLE spark_catalog.s.frames (x int)")
    assert find_table(spark, "spark_catalog", "s", "frames") == "spark_catalog.s.frames"
    for name in ("frames", "frames_8w", "frames_2w"):
        spark.sql(f"DROP TABLE spark_catalog.s.{name}")
