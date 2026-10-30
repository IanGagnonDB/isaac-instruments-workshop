# ISAAC Instruments video workshop

CPU video classification on **Databricks Serverless ML v6**: classify a video, benchmark three UDFs, register a model, and inspect successful and failed traces. No training or app deployment.

## Start here

1. Clone this repository as a **Databricks Git Folder**. Give both participants access; use separate signed-in identities. Keep the same Git revision during the session.
2. Open [00_configuration](notebooks/00_configuration.py) and enter **VOLUME_ROOT**, an existing Unity Catalog Volume path. Real D2-City data is selected by default. The other settings have small CPU defaults.
3. In each exercise notebook select **Serverless → ML base environment → version 6**. Run from the top: the visible `%pip` cell installs extras, then Python restarts before imports. No requirements file or manual dependency-panel editing.
4. If data/model files are missing, one designated participant opens [optional_prepare_dataset](notebooks/optional_prepare_dataset.py), selects **Download D2-City**, and runs it. Complete datasets and staged models are reused; the second person does not download them again.
5. Run **01 → 02 → 03 → 04**. Notebook 04 additionally needs **TRACE_WAREHOUSE_ID**, selected in 00_configuration; the first three exercises do not require a SQL warehouse.

| Notebook | What you do |
|---|---|
| [00 · Configuration](notebooks/00_configuration.py) | Enter one shared Volume location and choose the dataset |
| [Optional · Prepare data](notebooks/optional_prepare_dataset.py) | Stage real data and weights once; run a one-frame smoke test |
| [01 · Classify a video](notebooks/01_classify_video.py) | Decode, sample and classify frames; save traceable predictions |
| [02 · Benchmark UDFs](notebooks/02_benchmark_udfs.py) | Compare Python, scalar pandas and iterator pandas UDFs |
| [03 · Model lifecycle](notebooks/03_model_lifecycle.py) | Register versions; promote and roll back `champion` |
| [04 · Run and trace](notebooks/04_run_and_trace_champion.py) | Run the exact champion version; trace success and failure |

Setup and inference plumbing lives in `utils/`. Participants do not need to edit those files. Each `.py` file under `notebooks/` is a Databricks source notebook.

## Real data first, explicit fallback

D2-City videos remain the default. Preparation downloads the full source archive and creates one benchmark instance per source video. The scoring limit reduces the inference workload, not the initial download. No third-party videos are redistributed in Git.

A failed real-data download **stops and explains the error**. It never switches datasets automatically. If you explicitly want to continue with generated videos:

1. Set **DATASET = 'synthetic'** in 00_configuration.
2. In optional_prepare_dataset select **Use synthetic fallback**, then rerun it.
3. Continue through 01–04. All events/results identify the dataset as synthetic.

The fallback generates three short MP4s containing moving shapes, gradients and frame numbers. It verifies pipeline mechanics; ImageNet predictions on those images have no semantic meaning. Generated videos go to a separate folder and never replace real footage. Model weights still require Hugging Face access if they are not already staged.

```text
<Volume>/
  datasets/d2_city/       # raw/, benchmark/manifest.csv, dataset.json
  datasets/synthetic/     # separate fallback, same data contract
  workshop_models/       # pinned checkpoints, shared by both people
  workshop_outputs/      # participant / unique execution ID
```

## CPU environment and benchmark

Use the [Serverless v6 ML base environment](https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/six), which supplies the CPU ML stack, including Torch, torchvision and transformers. Every executable exercise visibly installs `opencv-python-headless==4.11.0.86`; lifecycle/tracing exercises additionally install `mlflow==3.17.2` for their model and trace APIs. The workshop does not pin or reinstall the base Torch, NumPy or pandas stack. Runtime package versions are saved with results.

The primary CPU starter is **microsoft/resnet-18**, pinned to checkpoint revision `65a5785d9156231087c481e0c7dd33a5ff6f7e3e`. For an optional architecture comparison, choose **microsoft/resnet-50**, revision `34c2154c194f829b11125337b98c8f5f9965ff19`, prepare its weights, then rerun. Model cards list Apache 2.0. Both models classify ImageNet objects, not driving-event severity.

Notebook 02 compares ordinary Python UDFs, scalar pandas UDFs and iterator pandas UDFs using shared decoding/preprocessing/inference. A driver-side sequential run is the correctness reference. Before comparing throughput, all UDFs must produce the same event/frame coverage and top-1 predictions with confidence differences <=1e-4.

Defaults are four events, three sampled frames per event, and small inference batches. The synthetic dataset has only three events. These defaults validate functionality; increase the workload for useful throughput measurements. First-use rounds and repeated rounds are reported separately, with model-load times, output-write completion, variation and dataset provenance. Serverless resource allocation/cache state is not fixed; results are observed performance, not guaranteed speedups. Arrow batch size is managed by the platform; tune Torch inference batch size separately.

## Access and troubleshooting

Both participants need USE CATALOG, USE SCHEMA and READ/WRITE VOLUME. Notebook 03 also needs CREATE MODEL, experiment creation/access and ownership of participant models for alias changes. Notebook 04 needs CREATE TABLE in the trace schema and CAN USE on its SQL warehouse. Create/select those resources with an administrator when necessary; the workshop does not create arbitrary infrastructure or grant permissions.

Models, experiments and output paths are derived from signed-in identity. On compute that shares a notebook execution session, take turns running cells; separate usernames alone do not isolate a shared Python process. Use independent notebook sessions for concurrent hands-on work, and change shared configuration only when both participants have stopped. Model alias changes are shared by sessions of the same identity; re-running 03 intentionally creates versions.

See the [facilitator runbook](docs/facilitator-runbook.md) for recovery and a fresh-workspace checklist, and [workshop plan](docs/workshop-plan.md) for scope. Local checks do not establish live ML v6/Spark/Unity Catalog compatibility: validate the complete flow in the customer workspace.
