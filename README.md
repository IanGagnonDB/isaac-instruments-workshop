# ISAAC Instruments: CPU image classification at scale with Spark

A Databricks Asset Bundle that pulls ~10 frames from each dashcam clip with OpenCV, then classifies them
with ResNet-18/50. It compares **PyTorch, ONNX Runtime fp32 and ONNX int8**, measures how throughput
scales from 2 to 8 workers using **Spark only** (`mapInPandas` and pandas UDFs, no Ray), and scores
rolling-stop events by joining predictions with speed telemetry.

CPU only, on Azure Databricks (built and run in Canada Central). Setup and benchmarks use classic CPU job
clusters. Scoring and the report run on serverless. Everything environment-specific is a bundle
variable.

> **Branch `low-vcpu`: for a ~20-vCPU quota and a required compute policy.** Differences from `main`:
>
> * **4-vCPU nodes** (`Standard_F4s_v2`) everywhere. The largest cluster is the benchmark at **20 vCPUs**
>   (driver + 4 workers); setup and fine-tune use 8. Scoring and MLflow run on serverless, which does not
>   use your Azure vCPU quota.
> * **One benchmark cluster, core sweep.** Instead of one cluster per size, `01-video-cls-benchmark` runs
>   the pipeline on one cluster at 4, 8 and 16 concurrent tasks (`task_caps`). That scales cores, not
>   machines, with a single cluster start-up.
> * **Compute policy on every classic job cluster.** Set the policy *name* in `databricks.yml`
>   (`policy_id` → `lookup: cluster_policy:`); the bundle looks up its ID at deploy time. Clusters set
>   `apply_policy_default_values: true` and on-demand VMs.
> * **No single-node clusters.** Setup and fine-tune run on a driver + 1 worker, because policies such as
>   *Job Compute* forbid `spark.databricks.cluster.profile`. The notebooks still do that work on the driver.
>
> The reference numbers further down were measured with `main`'s 16-vCPU configuration.

```
clips + JSON in a UC Volume  (/Volumes/<catalog>/<schema>/<volume>/clips, /json)
  -> events table            Auto Loader on the JSON (event_id, clip_path, vehicle, driver, speed trace)
  -> repartition(N)          spread clips across every core
  -> mapInPandas             extract_frames: OpenCV seek to 10 timestamps -> one row per frame (JPEG)
  -> frames_<label>          written between stages so extract and inference are timed separately
  -> repartition(N)          rebalance frames
  -> pandas UDF classify     ONNX Runtime or PyTorch, model loaded once per task, Arrow-batched
  -> frame_predictions_<label>
  -> SQL                     join speed telemetry, rolling-stop rule -> event_scores table
```

`<label>` is the benchmark task: `sweep` on this branch, with result rows labelled `sweep_cap4`,
`sweep_cap8` and `sweep_cap16` (on `main`: `2w`, `4w`, `8w`, `4w_task_cpus_2`, one cluster each, in parallel).

## Known limitation

ImageNet-1k has **no "stop sign" class**, and a classifier returns one label per frame, not a location.

* That doesn't matter for the benchmark, which measures speed and scaling.
* The rolling-stop demo uses a clearly labelled **proxy class**: the `sign_class` parameter, default
  **920 "traffic light"**, which is top-1 on ~15% of D2-City frames. The closer class, 919 "street sign",
  is almost never a frame's top-1 label on this footage. In the reference run, ResNet-18 had 0 of 3,992
  frames and every event scored `none`. Use `--params sign_class=919` to reproduce that.
* The optional notebook 05 retrains the last layer as `stop_sign_visible: yes/no` on frames you label.
  **It is not runnable with the current dataset:** D2-City has no stop-sign labels, so `labels/yes` and
  `labels/no` are empty and the job exits with "Skipped". It needs a hand-labelled set (for example
  ISAAC's own frames) or a dataset with stop-sign annotations.
* To know *where* the sign is and how close, the next step is a COCO-trained detector (DETR-ResNet-50
  or RT-DETR, both Apache 2.0) that has a real "stop sign" class. That is out of scope here.
* `04_score_events` is a demonstration rule, not a validated severity model. The speed traces are
  synthetic and not synced to the video.

## Data

The default source is **D2-City**, public dashcam footage (~30 s, 1080p, H.264, 25 fps). No video is
stored in Git. `00_setup` gets source clips in one of two ways:

* `source_clips_path` set: copies the MP4s under that folder, for example an existing D2-City `raw/`
  folder in another volume. The folder must be readable from the deploying workspace.
* `source_clips_path` empty: downloads the D2-City ZIP once into `<volume>/d2_city/` with
  `video_cls/d2_city.py`, then reuses it.

The sources are replicated to `n_clips` clip_ids (default 1,000) under `clips/` as plain byte copies,
with no re-encoding. Each clip gets a synthetic event JSON with a 2 Hz `speed_kmh` trace: about 35% full
stops, 40% rolling stops at 5–15 km/h, and 25% with no real slowdown. The benchmark uses the first
`max_clips` of them (default 400).

## Prerequisites

* A workspace that allows **classic job clusters** (serverless-only workspaces reject the setup and
  benchmark jobs) and **serverless compute** (scoring, and deploying from the workspace UI).
* **Workspace files** enabled and **Git folders** available (the default).
* An existing **catalog** you can `USE` and `CREATE SCHEMA` in, or an existing schema with `CREATE TABLE`
  and `CREATE VOLUME`.
* **vCPU quota: 20 Fsv2 vCPUs** with the defaults: the benchmark cluster is (`bench_workers` + 1) × 4.
  Run the jobs one at a time; setup (8 vCPUs) and the benchmark (20) never need to overlap. Azure
  quotas are per VM family *and* per region, so check both.
* **A compute policy** your job clusters may use. It must allow dedicated (single user) access mode,
  `runtime_engine: STANDARD`, the node types and on-demand VMs. Ask your admin if the deploy or a
  cluster start is rejected by the policy.
* Egress to `download.pytorch.org`, or to `huggingface.co` as the automatic fallback, for model weights.
  For the D2-City download only: egress to `download.scidb.cn`.

No local tools are required. The Databricks CLI is optional (see below). Nothing is built at deploy time:
notebooks import `src/video_cls` from the repository folder and ship it to Spark executors by value
(`cloudpickle.register_pickle_by_value`).

## Bundle variables

Set them in the `variables:` section of [`databricks.yml`](databricks.yml). From the CLI you can also pass
`--var name=value`.

| Variable | Default | Notes |
|---|---|---|
| `catalog` | `CHANGE_ME_CATALOG` | **Edit before deploying.** Existing catalog; never created by the bundle |
| `schema` | `CHANGE_ME_SCHEMA` | **Edit before deploying.** Created by `00_setup` if missing |
| `volume` | `video` | Created by `00_setup` if missing |
| `source_clips_path` | empty | Folder of source MP4s. Empty = download D2-City |
| `n_clips` | `1000` | clip_ids created by replication |
| `policy_id` | lookup of policy **`Job Compute`** | **Edit the policy name** (`lookup: cluster_policy:`). Attached to every classic job cluster |
| `node_type_id` | `Standard_F4s_v2` | Workers, and the setup driver (single-node benchmark node) |
| `driver_node_type_id` | `Standard_F4s_v2` | Driver of the benchmark cluster |
| `bench_workers` | `4` | Benchmark workers; peak vCPUs = (`bench_workers` + 1) × 4 |
| `spark_version` | `18.x-cpu-ml-scala2.13` | 18 LTS ML (CPU). Check with `databricks clusters spark-versions` |

Every notebook stops with a clear error if `catalog` or `schema` still contains `CHANGE_ME`.

Every classic cluster uses **dedicated (single user) access mode**, so `/Volumes` paths work inside UDFs
on executors. Photon is off (`runtime_engine: STANDARD`), because it does not speed up Python UDFs and
costs more DBUs. Libraries are pinned against the DBR 18 ML stack (protobuf 5.29, numpy 2.1):
`onnx==1.21.0`, `onnxruntime==1.31.0`, `opencv-python-headless==4.11.0.86`. Torch and torchvision
come from the runtime.

## Deploy and run from a Git folder (no CLI)

1. **Workspace → Create → Git folder**, and paste this repository's URL.
2. In the Git folder, open **`databricks.yml`** and set `catalog` and `schema` under `variables:`, and the
   compute policy name under `policy_id`. Optionally also set `source_clips_path`, `n_clips` or the node
   types.
3. With `databricks.yml` open, click the **Deployments** icon. Choose the
   **dev** target and click **Deploy**. Review the validation summary and confirm.
4. In **Bundle resources**, run the jobs in the order of their number prefix with the ▶ icon:
   * `00-video-cls-setup`: setup → export_models → single_node_bench (classic, driver + 1 worker,
     8 vCPUs; estimate ~30 min on 4-vCPU nodes)
   * `01-video-cls-benchmark`: one classic cluster (20 vCPUs), core sweep at 4, 8 and 16 concurrent tasks
     on 48 clips per pass (estimate ~7 min start-up + ~10 min)
   * `02-video-cls-scoring`: score_events → report (serverless, ~1 min)
   * *(optional)* `03-video-cls-mlflow-optional`: logs the benchmark results as an MLflow experiment
     (serverless, ~3 min). See [Optional: the benchmark in MLflow](#optional-the-benchmark-in-mlflow)
   * *(optional)* `04-video-cls-finetune-optional`: needs labelled frames, which D2-City doesn't have

   In a `dev` deployment the names get a `[dev <you>]` prefix, so they still sort in this order.
5. To change job parameters for one run (for example `max_clips` or `sign_class`), open the job from
   **Bundle resources** and choose **Run now with different parameters**.

Development-target deployments from the workspace are **source-linked**: jobs run the notebooks directly
from the Git folder, so after a `git pull` the next run uses the new code. Redeploy only when
`databricks.yml` or `resources/*.yml` change. Deploying runs the bundle as you; deploy only repositories
you trust. The workspace UI can deploy only to the workspace that holds the Git folder.

## Deploy and run with the CLI (optional)

Use a local Databricks CLI v0.250+, or the one in the workspace **web terminal**.

```bash
databricks bundle deploy -t dev --var catalog=my_catalog --var schema=video_demo -p <profile>

databricks bundle run video_cls_setup     -t dev --var catalog=my_catalog --var schema=video_demo -p <profile>
databricks bundle run video_cls_benchmark -t dev --var catalog=my_catalog --var schema=video_demo -p <profile>
databricks bundle run video_cls_scoring   -t dev --var catalog=my_catalog --var schema=video_demo -p <profile>
# optional, needs labelled frames (not available for D2-City): fine-tune, then rerun scoring with sign_source=finetuned
databricks bundle run video_cls_finetune_optional -t dev ... --params export_unlabeled=300
# optional: log the benchmark tables as an MLflow experiment
databricks bundle run video_cls_mlflow_optional   -t dev --var catalog=my_catalog --var schema=video_demo -p <profile>
```

`--var` can be dropped if `databricks.yml` already has your values. Use `-t prod` for a
non-development deployment. Job parameters can be overridden per run with `--params`, for example
`--params runs=resnet18:int8,resnet50:int8,max_clips=800`.

## Notebooks

| Notebook | Compute | What it does | Writes |
|---|---|---|---|
| `00_setup` | classic, driver + 1 worker | Schema/volume, source clips, replication, synthetic JSON, Auto Loader | `events` |
| `01_export_models` | classic, driver + 1 worker | torchvision weights (HF fallback) → `.pt`, ONNX fp32 opset 17, int8 static QDQ; top-1 agreement on ~100 calibration frames | `model_exports`, `models/` |
| `02_single_node_bench` | classic, driver + 1 worker | {resnet18, resnet50} × {torch, fp32, int8} × batch {1, 16, 64}, 500 frames, driver only | `bench_single_node` |
| `03_cluster_bench` | classic, 4 workers | Full Spark pipeline; per `task_caps` pass: extraction, then one inference pass per `runs` entry | `frames_<label>`, `frame_predictions_<label>`, `bench_cluster` |
| `04_score_events` | serverless | Sign visible (`sign_class` proxy or fine-tuned) + min speed within `window_s` → severity | `event_scores` |
| `05_finetune_optional` | classic, driver + 1 worker | Frozen ResNet-18, new 2-class head from `labels/yes`, `labels/no`; ONNX + int8 export. Not runnable on D2-City (no labels) | `models/stopsign_resnet18*` |
| `06_report` | serverless | ms/frame per backend; frames/s vs cores with a linear-scaling line; extract vs inference share; conclusion | — |
| `07_mlflow_experiment_optional` | serverless | Logs `bench_single_node`, `model_exports` and `bench_cluster` as MLflow runs; measures nothing | MLflow experiment |

`03_cluster_bench` job parameters: `runs` (comma list of `model:variant`, variant ∈ `torch`, `fp32`,
`int8`; default resnet18 × all three), `task_caps` (default `4,8,16`; each value ≤ `bench_workers` × 4),
`max_clips` (clips **per pass**, default 48, so the sweep reads 144 distinct clips), `batch_size` (Arrow
`maxRecordsPerBatch` = images per forward pass), `n_frames`. With `task_caps`, each pass uses exactly
*cap* partitions and `n_partitions` is ignored; with `task_caps` empty, 03 makes one pass on every slot,
as on `main`. `bench_cluster` is created with a fixed schema and only receives appends.

`04_score_events` parameters: `model` / `variant` (default resnet18 **fp32**; int8 disagrees with PyTorch
too often, see below), `sign_class` (default 920 "traffic light"), `score_threshold` (0.3), `window_s` (3) and the
severity thresholds. It reuses a benchmark `frame_predictions_<label>` table and scores only the events
those predictions cover (the first `max_clips` clips). Without benchmark predictions, or with
`sign_source=finetuned`, it classifies frames itself with a UDF that loads the model from `/Volumes`.
That is guaranteed on dedicated classic clusters but not verified on serverless, so run the notebook on a
classic cluster if serverless rejects it.

## Optional: the benchmark in MLflow

`07_mlflow_experiment_optional` (job `03-video-cls-mlflow-optional`) turns the benchmark tables into an MLflow
experiment, `/Users/<you>/isaac-video-classification` by default (`experiment_path` parameter). It is off
the critical path and re-measures nothing.

* Two parent runs, `single_node benchmark` and `cluster benchmark`, each with a summary chart and one nested
  child run per measured configuration: 18 single-node runs (model × backend × batch size) and 24 cluster
  runs (cluster × model × backend) in the reference run.
* Child runs carry the configuration as **parameters** (model, variant, backend, batch size, threads,
  workers, cores, partitions) and the results as **metrics** (ms/frame, frames/s, extract and inference
  seconds, int8 top-1 agreement, model size).
* To demo: open the experiment, expand a parent run, select its children and click **Compare**. Sort by
  `frames_per_s`, or chart `inference_frames_per_s` against `total_cores` with one line per variant.
* It only needs the setup job's tables; the cluster runs are added once the benchmark job has run.
  Rerunning is safe: a benchmark run that is already logged (same newest `run_ts`) is skipped.

## Point it at your own clips

1. Upload clips to any UC volume folder. Set `source_clips_path` to that folder in `databricks.yml` (or
   `--var source_clips_path=/Volumes/.../clips`). Set `n_clips` to replicate them for scale testing.
2. To use real telemetry instead of the synthetic JSON, write one JSON per clip into
   `/Volumes/<catalog>/<schema>/<volume>/json/` with the fields `event_id, clip_id, clip_path,
   vehicle_no, driver_no, event_type, start_ts, speed_kmh` (2 Hz array). Run `00_setup` with
   `n_clips=0` so it only runs Auto Loader. Use `reset=true` to rebuild `events` from scratch.

## Reference run (Canada Central, Standard_F16s_v2 workers)

Measured with `main`'s configuration (16-vCPU nodes, one cluster per size), not this branch's defaults.

**Export:** ONNX fp32 matched PyTorch top-1 on **100%** of calibration frames for both models. Static
int8 agreed on only **77% (resnet18)** and **62% (resnet50)**. Dashcam frames give low-confidence,
near-tie ImageNet predictions, and the calibration used plain min/max on 100 frames. Percentile or entropy
calibration, `reduce_range=True` and held-out evaluation are untested next steps. Until then, use int8
for speed comparisons and fp32 for decisions.

**Single node, 16 threads, batch 64 (ms/frame):**

| | PyTorch | ONNX fp32 | ONNX int8 |
|---|---|---|---|
| resnet18 | 11.1 | 4.9 | 3.3 |
| resnet50 | 37.2 | 10.5 | 6.4 |

**Cluster, 400 clips / 3,992 frames, four clusters in parallel:**

| Workers (cores) | Extraction | resnet18 int8 inference | resnet18 fp32 inference |
|---|---|---|---|
| 2 (32) | 224 s | 367 frames/s | 190 frames/s |
| 4 (64) | 130 s | 480 frames/s | 198 frames/s |
| 8 (128) | 83 s | 660 frames/s | 186 frames/s |

* **Frame extraction dominates** (80–95% of pipeline time) and scales 2.7× from 2 to 8 workers (68% of
  linear). Seeking in 1080p H.264 runs at about 1 frame/s per vCPU. The four clusters shared storage
  bandwidth, so isolated runs may extract faster.
* **Inference barely scales at 4,000 frames.** At ~13 frames/s per core (resnet18 fp32), that is ~10 s
  of model compute on 32 cores and ~2.5 s on 128. Each run took ~21 s regardless, so ~18 s is fixed
  overhead: Spark job start, reading frames, the shuffle, per-task model loading and the Delta write. To
  see inference scaling, give the stage more frames. `max_clips=0` (all 1,000 default clips, 10,000 frames) helps but also makes
  extraction ~2.5× longer. Even then the 8-worker point is only ~6 s of compute.
* `spark.task.cpus=2` with 2 threads per task was about the same as 1 thread per vCPU.
* Wall-clock: setup job 20 min with 2,000 clips (15 of them D2-City download + copy). Benchmark 16 min (~7 min cluster
  start + 5–8 min work, in parallel). Scoring + report about 1 min on serverless.
* A 1-worker task was dropped: at 2,000 clips its extraction alone ran past 17 minutes.
* `sign_class=920` scored 138 of 400 events. Full stops landed in `ok`, rolling stops in `low`/`medium`,
  and no-stops only in `high`.

## Parallelism rules applied

1. Repartition the clip list to 2–4× total cores (default 3× task slots). A small table of paths
   otherwise sits in one or two partitions.
2. Repartition again after extraction to rebalance frames from uneven clips.
3. One ONNX/PyTorch thread per Spark task (`intra_op_num_threads`/`torch.set_num_threads` =
   `spark.task.cpus`).
4. `spark.sql.execution.arrow.maxRecordsPerBatch` = 64. The default of 10,000 can run executors out of
   memory with image rows.
5. Seek, don't decode: `CAP_PROP_POS_MSEC` to 10 timestamps. Frames are downscaled to 640 px wide before
   JPEG encoding to keep shuffle rows small, which has no effect after the 224 × 224 resize. Limiting
   OpenCV to one decode thread per task was tested locally: +20–30% when all cores are busy, but 6×
   slower when a clip has a core to itself, so it is not applied.
6. Photon off. Compute-optimized 16-core workers, scaled by adding workers.
7. Classic clusters for benchmarking: repeatable, full Spark config control, and `sparkContext`
   available. Serverless only for the SQL/driver-side scoring and report.

`03_cluster_bench` waits until all `num_workers` executors have registered before timing.

## Cost drivers

Cost ≈ Σ over tasks of (nodes × DBU/hour per node × hours) + the VM cost of the same nodes. Main drivers:

* **Benchmark job**: one 20-vCPU cluster (5 × 4-vCPU nodes): ~7 min of start-up, then three passes of
  extraction and three inference runs each. Trim `runs`, `task_caps` or `max_clips` to cut time.
* **Setup job**: driver + 1 worker (8 vCPUs). Downloading D2-City and copying 1,000 × ~15 MB clips
  (~15 GB) dominates; use a smaller `n_clips` for a quicker first run.
* **Scoring/report**: serverless jobs (performance-optimized) for about a minute.
* Storage: ~15 GB of replicated clips (plus the D2-City ZIP and extract, if downloaded) and small Delta
  tables.

## Local tests

```bash
uv sync --group dev          # Python 3.12, versions aligned with DBR 18 ML
uv run pytest                # the PySpark tests need Java 17 (JAVA_HOME)
```

`test_preprocess.py` and `test_frames.py` cover `preprocess` and `extract_frames` on synthetic MP4s,
including seek accuracy. Other tests cover the speed traces and replication, ONNX export and int8
quantization, the UDF contract, the scoring SQL bands and window, the parallel-safe table helpers, and
(with PySpark) the full `mapInPandas` → pandas UDF pipeline for both backends.
`test_pickle_by_value.py` runs that pipeline with Spark workers that cannot import `video_cls`, as on a
cluster where only the notebook added `src/` to its path. `test_tracking.py` logs sample benchmark tables
to a temporary local MLflow store and checks the runs, metrics, charts and the skip on rerun.

## Licences

torchvision ResNet weights (BSD-3-Clause) or `microsoft/resnet-18` / `microsoft/resnet-50` on Hugging
Face (Apache 2.0). No Ultralytics/YOLO (AGPL). D2-City is downloaded from its publisher at setup time
and is not redistributed here.
