# ISAAC Instruments: CPU image classification at scale with Spark

A Databricks Asset Bundle that pulls ~10 frames from each dashcam clip with OpenCV, then classifies them
with ResNet-18/50. It compares **PyTorch, ONNX Runtime fp32 and ONNX int8**, measures how throughput
scales from 2 to 8 workers using **Spark only** (`mapInPandas` and pandas UDFs, no Ray), and scores
rolling-stop events by joining predictions with speed telemetry.

CPU only, on Azure Databricks (built and run in Canada Central). Setup and benchmarks use classic CPU job
clusters. Scoring and the report run on serverless. Everything environment-specific is a bundle
variable.

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

`<label>` is the benchmark task (`2w`, `4w`, `8w`, `4w_task_cpus_2`). Each benchmark cluster writes its own
tables, so the clusters can run in parallel.

## Known limitation

ImageNet-1k has **no "stop sign" class**, and a classifier returns one label per frame, not a location.

* That doesn't matter for the benchmark, which measures speed and scaling.
* The rolling-stop demo uses a clearly labelled **proxy class**: the `sign_class` parameter, default 919
  "street sign". On D2-City footage, 919 is almost never a frame's top-1 label. In the reference run,
  ResNet-18 had 0 of 3,992 frames and every event scored `none`. Class **920 "traffic light"** is
  top-1 on ~15% of frames and gives a usable demo: `--params sign_class=920`.
* The optional notebook 05 retrains the last layer as `stop_sign_visible: yes/no` on frames you label.
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

The sources are replicated to `n_clips` clip_ids (default 2,000) under `clips/` as plain byte copies,
with no re-encoding. Each clip gets a synthetic event JSON with a 2 Hz `speed_kmh` trace: about 35% full
stops, 40% rolling stops at 5–15 km/h, and 25% with no real slowdown. The benchmark uses the first
`max_clips` of them (default 400).

## Prerequisites

* Databricks CLI v0.250+ (`databricks bundle`) authenticated to your workspace (`-p <profile>`).
* [`uv`](https://docs.astral.sh/uv/) on the machine that deploys. The bundle builds the `video_cls`
  wheel with `uv build`.
* A workspace that allows **classic job clusters** (serverless-only workspaces reject the setup and
  benchmark jobs) and **serverless jobs** (scoring).
* An existing **catalog** you can `USE` and `CREATE SCHEMA` in, or an existing schema with `CREATE TABLE`
  and `CREATE VOLUME`.
* **vCPU quota** for the node family. The four benchmark clusters start together and need
  18 × `node_type_id` + 4 × `driver_node_type_id`: **320 Fsv2 vCPUs** with the defaults. With less quota,
  add `depends_on` between the benchmark tasks to run them one at a time (136 vCPUs at peak).
* Egress to `download.pytorch.org`, or to `huggingface.co` as the automatic fallback, for model weights.
  For the D2-City download only: egress to `download.scidb.cn`.

## Bundle variables

| Variable | Default | Notes |
|---|---|---|
| `catalog` | *(required)* | Never created by the bundle |
| `schema` | *(required)* | Created by `00_setup` if missing |
| `volume` | `video` | Created by `00_setup` if missing |
| `source_clips_path` | empty | Folder of source MP4s. Empty = download D2-City |
| `n_clips` | `2000` | clip_ids created by replication |
| `node_type_id` | `Standard_F16s_v2` | Workers, and the setup / single-node benchmark node |
| `driver_node_type_id` | `Standard_F8s_v2` | Driver of the multi-node benchmark clusters |
| `spark_version` | `18.x-cpu-ml-scala2.13` | 18 LTS ML (CPU). Check with `databricks clusters spark-versions` |

Every classic cluster uses **dedicated (single user) access mode**, so `/Volumes` paths work inside UDFs
on executors. Photon is off (`runtime_engine: STANDARD`), because it does not speed up Python UDFs and
costs more DBUs. Libraries are pinned against the DBR 18 ML stack (protobuf 5.29, numpy 2.1):
`onnx==1.21.0`, `onnxruntime==1.31.0`, `opencv-python-headless==4.11.0.86`. Torch and torchvision
come from the runtime.

## Deploy and run

```bash
databricks bundle validate -t dev --var catalog=my_catalog --var schema=video_demo -p <profile>
databricks bundle deploy   -t dev --var catalog=my_catalog --var schema=video_demo -p <profile>

# 1. setup -> export_models -> single_node_bench      (one classic single-node cluster)
databricks bundle run video_cls_setup     -t dev --var catalog=my_catalog --var schema=video_demo -p <profile>
# 2. 2 / 4 / 8 workers + 4 workers with spark.task.cpus=2, four classic clusters in parallel
databricks bundle run video_cls_benchmark -t dev --var catalog=my_catalog --var schema=video_demo -p <profile>
# 3. score_events -> report                            (serverless)
databricks bundle run video_cls_scoring   -t dev --var catalog=my_catalog --var schema=video_demo -p <profile> \
  --params sign_class=920
# optional: fine-tune stop_sign_visible, then rerun 3 with sign_source=finetuned
databricks bundle run video_cls_finetune_optional -t dev ... --params export_unlabeled=300
```

Use `-t prod` for a non-development deployment. Job parameters can be overridden per run with
`--params`, for example `--params runs=resnet18:int8,resnet50:int8,max_clips=800`.

| Notebook | Compute | What it does | Writes |
|---|---|---|---|
| `00_setup` | classic, 1 node | Schema/volume, source clips, replication, synthetic JSON, Auto Loader | `events` |
| `01_export_models` | classic, 1 node | torchvision weights (HF fallback) → `.pt`, ONNX fp32 opset 17, int8 static QDQ; top-1 agreement on ~100 calibration frames | `model_exports`, `models/` |
| `02_single_node_bench` | classic, 1 node | {resnet18, resnet50} × {torch, fp32, int8} × batch {1, 16, 64}, 500 frames, driver only | `bench_single_node` |
| `03_cluster_bench` | classic, 2–8 workers | Full Spark pipeline; extraction once per cluster, then one inference pass per `runs` entry | `frames_<label>`, `frame_predictions_<label>`, `bench_cluster` |
| `04_score_events` | serverless | Sign visible (`sign_class` proxy or fine-tuned) + min speed within `window_s` → severity | `event_scores` |
| `05_finetune_optional` | classic, 1 node | Frozen ResNet-18, new 2-class head from `labels/yes`, `labels/no`; ONNX + int8 export | `models/stopsign_resnet18*` |
| `06_report` | serverless | ms/frame per backend; frames/s vs cores with a linear-scaling line; extract vs inference share; conclusion | — |

`03_cluster_bench` job parameters: `runs` (comma list of `model:variant`, variant ∈ `torch`, `fp32`,
`int8`), `max_clips` (default 400; `0` = all), `n_partitions` (`0` = 3 × task slots), `batch_size` (Arrow
`maxRecordsPerBatch` = images per forward pass), `n_frames`. `ort_threads` is set per task: 1, or 2 on the
`spark.task.cpus=2` task. `bench_cluster` is created with a fixed schema and only receives appends,
which is safe from parallel clusters.

`04_score_events` parameters: `model` / `variant` (default resnet18 **fp32**; int8 disagrees with PyTorch
too often, see below), `sign_class` (default 919), `score_threshold` (0.3), `window_s` (3) and the
severity thresholds. It reuses a benchmark `frame_predictions_<label>` table and scores only the events
those predictions cover (the first `max_clips` clips). Without benchmark predictions, or with
`sign_source=finetuned`, it classifies frames itself with a UDF that loads the model from `/Volumes`.
That is guaranteed on dedicated classic clusters but not verified on serverless, so run the notebook on a
classic cluster if serverless rejects it.

## Point it at your own clips

1. Upload clips to any UC volume folder. Pass it as `--var source_clips_path=/Volumes/.../clips`,
   and set `--var n_clips=<count>` to replicate them for scale testing.
2. To use real telemetry instead of the synthetic JSON, write one JSON per clip into
   `/Volumes/<catalog>/<schema>/<volume>/json/` with the fields `event_id, clip_id, clip_path,
   vehicle_no, driver_no, event_type, start_ts, speed_kmh` (2 Hz array). Run `00_setup` with
   `n_clips=0` so it only runs Auto Loader. Use `reset=true` to rebuild `events` from scratch.

## Reference run (Canada Central, Standard_F16s_v2 workers)

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
* **Inference barely scales at 4,000 frames.** That is under a second of model compute on 128 cores,
  so per-task model loading, the shuffle and the Delta write dominate. Measure inference scaling with
  `max_clips=0` (20,000 frames).
* `spark.task.cpus=2` with 2 threads per task was about the same as 1 thread per vCPU.
* Wall-clock: setup job 20 min (15 of them D2-City download + copy). Benchmark 16 min (~7 min cluster
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

* **Benchmark job**: four clusters at once (2, 4, 8, 4 workers, plus a driver each), each paying ~7 min
  of start-up plus extraction and six inference passes. Running them in parallel cuts wall-clock time,
  not DBUs. Trim `runs` or `max_clips` to cut both.
* **Setup job**: one 16-vCPU node. Downloading and copying 2,000 × ~15 MB clips (~30 GB) dominates; use a
  smaller `n_clips` for a quicker first run.
* **Scoring/report**: serverless jobs (performance-optimized) for about a minute.
* Storage: ~30 GB of replicated clips (plus the D2-City ZIP and extract, if downloaded) and small Delta
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

## Licences

torchvision ResNet weights (BSD-3-Clause) or `microsoft/resnet-18` / `microsoft/resnet-50` on Hugging
Face (Apache 2.0). No Ultralytics/YOLO (AGPL). D2-City is downloaded from its publisher at setup time
and is not redistributed here.
