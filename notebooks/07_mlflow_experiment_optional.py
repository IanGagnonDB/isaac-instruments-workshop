# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Optional: the benchmark as an MLflow experiment
# MAGIC
# MAGIC Off the critical path, and nothing is re-measured: this notebook reads `bench_single_node`,
# MAGIC `model_exports` and (if the benchmark job has run) `bench_cluster`, then logs every measured
# MAGIC configuration as an MLflow run.
# MAGIC
# MAGIC * One parent run per benchmark (`single_node`, `cluster`) with a summary chart, and one nested child run
# MAGIC   per configuration. Parameters: model, backend, batch size, threads, cluster size. Metrics: ms/frame,
# MAGIC   frames/s, extract/inference seconds, int8 top-1 agreement and model size.
# MAGIC * In the experiment, expand a parent run, select its child runs and click **Compare**: sort by
# MAGIC   `frames_per_s`, or chart `inference_frames_per_s` against `total_cores`.
# MAGIC * Rerunning is safe: a benchmark run that is already logged (same newest `run_ts`) is skipped.

# COMMAND ----------
# MAGIC %pip install -q mlflow-skinny==3.17.2

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import os
import sys

# Import video_cls from this Git folder (no wheel or install).
sys.path.insert(0, os.path.abspath("../src"))
import video_cls

PARAMS = [("catalog", ""), ("schema", ""), ("experiment_path", "")]
for name, default in PARAMS:
    dbutils.widgets.text(name, default)
p = {name: dbutils.widgets.get(name) for name, _ in PARAMS}

catalog, schema = p["catalog"], p["schema"]
video_cls.check_location(catalog, schema)
user = spark.sql("SELECT current_user()").first()[0]
experiment_path = p["experiment_path"] or f"/Users/{user}/isaac-video-classification"

tables = {name: f"{catalog}.{schema}.{name}" for name in ("bench_single_node", "model_exports", "bench_cluster")}
missing = [tables[n] for n in ("bench_single_node", "model_exports") if not spark.catalog.tableExists(tables[n])]
if missing:
    raise RuntimeError(f"{', '.join(missing)} missing: run the setup job for schema '{schema}' first.")

# COMMAND ----------
import mlflow

from video_cls.tracking import log_cluster, log_single_node

experiment_id = mlflow.set_experiment(experiment_path).experiment_id
results = {"single_node": log_single_node(
    experiment_id, spark.table(tables["bench_single_node"]).toPandas(),
    spark.table(tables["model_exports"]).toPandas(), tables["bench_single_node"])}
if spark.catalog.tableExists(tables["bench_cluster"]):
    results["cluster"] = log_cluster(experiment_id, spark.table(tables["bench_cluster"]).toPandas(),
                                     tables["bench_cluster"])
else:
    print(f"{tables['bench_cluster']} not found: run the benchmark job to add the cluster runs.")

host = spark.conf.get("spark.databricks.workspaceUrl", "")
print(f"Experiment {experiment_path}" + (f": https://{host}/ml/experiments/{experiment_id}" if host else ""))
for benchmark, (run_id, logged) in results.items():
    print(f"  {benchmark:11s} {'logged' if logged else 'already logged, skipped'}: parent run {run_id}")
