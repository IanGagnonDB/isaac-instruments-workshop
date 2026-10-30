# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Report
# MAGIC 1. ms/frame per model x backend on one node (`bench_single_node`)
# MAGIC 2. Inference frames/s vs total worker cores, one line per model/variant, with perfect linear scaling
# MAGIC 3. Share of pipeline time in frame extraction vs inference (`bench_cluster`)
# MAGIC 4. Conclusion generated from the numbers

# COMMAND ----------
PARAMS = [("catalog", ""), ("schema", ""), ("volume", "video")]
for name, default in PARAMS:
    dbutils.widgets.text(name, default)
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
if not catalog or not schema or "CHANGE_ME" in catalog + schema:
    raise ValueError("Set catalog and schema under `variables:` in databricks.yml and redeploy.")

# COMMAND ----------
import matplotlib.pyplot as plt

single = spark.table(f"{catalog}.{schema}.bench_single_node")
latest_single = single.agg({"run_ts": "max"}).first()[0]
single = single.where(single.run_ts == latest_single).toPandas()

cluster = spark.table(f"{catalog}.{schema}.bench_cluster").toPandas()
# Latest run per cluster configuration (run_label), so reruns replace older numbers.
cluster = cluster[cluster.run_ts == cluster.groupby("run_label").run_ts.transform("max")]
cluster["series"] = cluster.model + " " + cluster.variant
scaling = cluster[cluster.task_cpus == 1]

# COMMAND ----------
# Chart 1: single-node latency, one bar group per model x backend, one bar per batch size.
table = single.pivot_table(index=["model", "variant"], columns="batch_size", values="ms_per_frame")
ax = table.plot.bar(figsize=(10, 4.5), rot=0)
ax.set_ylabel("ms / frame (lower is better)")
ax.set_xlabel("model, backend (torch = PyTorch, fp32/int8 = ONNX Runtime)")
ax.set_title(f"Single node, {int(single.threads.iloc[0])} threads, {single.node_type.iloc[0]}")
ax.legend(title="batch size")
plt.tight_layout()
plt.show()

# COMMAND ----------
# Chart 2: scaling with cores, dashed perfect-linear line from each series' smallest cluster.
fig, ax = plt.subplots(figsize=(9, 5))
for series, g in scaling.sort_values("total_cores").groupby("series"):
    line, = ax.plot(g.total_cores, g.inference_frames_per_s, marker="o", label=series)
    base = g.iloc[0]
    ax.plot(g.total_cores, base.inference_frames_per_s * g.total_cores / base.total_cores,
            linestyle="--", color=line.get_color(), alpha=0.5)
ax.set_xlabel("total worker cores")
ax.set_ylabel("inference frames / s")
ax.set_xticks(sorted(scaling.total_cores.unique()))
ax.set_title("Throughput vs cores (dashed: perfect linear scaling)")
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------
# Chart 3: where the time goes. Extraction is shared by all variants on a cluster.
share = scaling.assign(cluster=scaling.num_workers.astype(str) + "w").sort_values("total_cores")
share = share.assign(extract_pct=100 * share.extract_s / (share.extract_s + share.inference_s))
pivot = share.pivot_table(index="cluster", columns="series", values="extract_pct", sort=False)
ax = pivot.plot.bar(figsize=(10, 4.5), rot=0)
ax.axhline(50, color="grey", linewidth=0.8)
ax.set_ylabel("% of time in frame extraction (rest = inference)")
ax.set_title("Extract vs inference share of wall-clock")
plt.tight_layout()
plt.show()

# COMMAND ----------
# Conclusion, filled in from the numbers above.
lines = []
bs = single.batch_size.max()
for model, g in single[single.batch_size == bs].groupby("model"):
    ms = g.set_index("variant").ms_per_frame
    lines.append(f"<b>{model}</b> at batch {bs}: PyTorch {ms['torch']:.1f} ms/frame, ONNX fp32 {ms['fp32']:.1f} "
                 f"({ms['torch'] / ms['fp32']:.1f}x faster), int8 {ms['int8']:.1f} ({ms['torch'] / ms['int8']:.1f}x).")
for series, g in scaling.sort_values("total_cores").groupby("series"):
    if len(g) > 1:
        lo, hi = g.iloc[0], g.iloc[-1]
        speedup = hi.inference_frames_per_s / lo.inference_frames_per_s
        ideal = hi.total_cores / lo.total_cores
        lines.append(f"<b>{series}</b>: {lo.total_cores} -> {hi.total_cores} cores gives {speedup:.1f}x "
                     f"of an ideal {ideal:.0f}x ({100 * speedup / ideal:.0f}% scaling efficiency), "
                     f"{hi.inference_frames_per_s:,.0f} frames/s at {hi.total_cores} cores.")
extract_pct = share.extract_pct.median()
lines.append(f"Frame extraction (seek + decode + JPEG) is a median <b>{extract_pct:.0f}%</b> of pipeline time: "
             + ("decode, not the CNN, is the bottleneck." if extract_pct > 50 else "the CNN dominates."))
ht = cluster[cluster.task_cpus == 2]
for _, row in ht.iterrows():
    base = scaling[(scaling.num_workers == row.num_workers) & (scaling.series == row.series)]
    if len(base):
        ratio = row.inference_frames_per_s / base.iloc[0].inference_frames_per_s
        lines.append(f"{row.series}, {row.num_workers} workers with <code>spark.task.cpus=2</code>, 2 threads: "
                     f"{ratio:.2f}x the 1-thread-per-vCPU throughput.")

displayHTML("<h3>Conclusion</h3><ul>" + "".join(f"<li>{line}</li>" for line in lines) + "</ul>")
