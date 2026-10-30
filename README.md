# ISAAC Instruments Workshop

A one-day proof of concept for running a video frame classification pipeline on Databricks and benchmarking throughput and cost.

## Start in a new workspace — Git Folder

No DAB, CLI, or local Python installation is required. Two participants share one Git Folder and one dataset; each signs in with their own Databricks identity.

1. One person creates a **Git Folder** from this repository and grants the second participant access. If the repository is private, the person cloning needs Git provider credentials. Both participants need workspace permission to run notebooks.
2. Choose or create an existing Unity Catalog catalog, schema and Volume, plus a SQL warehouse. In the repository root, edit **`workshop_config.json`** once with those resource names and the warehouse ID. The Volume path is derived automatically. Do not put credentials in the file.
3. Open [00_setup_and_preflight](notebooks/00_setup_and_preflight.py). Follow its environment instructions in each notebook 00–04. The CPU starter uses Python 3.12 / Serverless environment version 4 and [requirements-cpu.txt](requirements-cpu.txt). Allow time for dependency installation. GPU needs a separately compatible environment; keep its runtime-provided Torch/torchvision.
4. Both participants run the initial checks. Designate **one participant** to set `prepare_dataset=true` and/or `prepare_models=true` in notebook 00 if those shared assets are missing. Rerun notebook 00 after changing widgets. Preparation is optional when validated assets already exist; data and model files themselves are required. Reset widgets to `false` afterward.
5. Both participants finish the one-event smoke test, then open **01 → 02 → 03 → 04**. Keep the shared config and Git revision unchanged while either person is running.

The starter scores only **two events** and creates **one benchmark instance per source video**. The dataset source is still a full ZIP download; download time and size are not reduced by the scoring limit. Increase the workload only after the smoke test works.

Shared: videos, manifest and pinned model snapshots. Isolated automatically by signed-in identity: experiment paths, registered model prefixes and output folders. Each execution also gets a unique output run ID. Use separate identities; two sessions of the same identity share model aliases. Re-running 03 intentionally adds model versions.

Resource access needed by both participants: USE CATALOG, USE SCHEMA, READ/WRITE VOLUME, CREATE MODEL, CREATE TABLE, and CAN USE on the selected SQL warehouse. Participants own the models they create and need permission to create experiments under their own user folders. Setup checks resource visibility and actual Volume read/write; notebooks 03/04 exercise registration, alias changes and trace provisioning/readback. Administrators must resolve missing grants.

**Compute gate:** Serverless CPU offers Sequential/Spark diagnostics, lifecycle and tracing, but not the complete matched Sequential/Ray/Spark benchmark. For that comparison use compatible classic single-node compute and pass notebook 02's probes. GPU availability is workspace/region dependent; selecting `device=cuda` asserts GPU access and fails clearly if unavailable. The earlier workspace's limitations are historical evidence, not a prediction for your new workspace.

The `classic_single_node` and `ray_on_spark` modes also require a compatible Ray installation. Use a DBR ML runtime that includes it or install Ray on the cluster; it is intentionally absent from the Serverless CPU starter requirements. Notebook 00 checks for it in those modes.

Use the [facilitator runbook](docs/facilitator-runbook.md) for fresh-workspace validation and recovery. Deployment automation with DABs can be added later; the workshop entry point is the Git Folder.

## Workshop scope

Use synthetic MP4 clips and public pretrained image classification models to stand in for ISAAC’s proprietary models and scripts. The intended pipeline is:

1. Ingest event archives containing MP4 video and JSON metadata into a Unity Catalog Volume and events table.
2. Extract frames with OpenCV using event-aware sampling (roughly 10 frames per 30-second clip; some event types may need only one).
3. Classify sampled frames in batches with PyTorch on Databricks Serverless GPU compute.
4. Track models with MLflow and compare inference throughput with full pipeline throughput.
5. Report events per minute, latency, and cost per event where compute cost can be attributed.

**Target compute:** Databricks Serverless GPU. This supersedes the CPU-only assumption in the initial workshop meeting notes. GPU access, supported dependencies, and the exact execution configuration still need validation in the workshop workspace.

## Selected models

Both models are pretrained CNNs for ImageNet-1k image classification. Their Hugging Face model cards list Apache 2.0 licensing.

| Role | Hugging Face model | Approximate parameters | Purpose |
| --- | --- | --- | --- |
| Primary | [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50) | 25.6 million | Main GPU batching and throughput benchmark |
| Lightweight baseline | [microsoft/resnet-18](https://huggingface.co/microsoft/resnet-18) | 11.7 million | Compare model compute cost within the same pipeline |

These models predict ImageNet object classes, not driving-event severity. They are substitutes for exercising the pipeline. Any severity scores shown in the workshop must be explicitly labeled as synthetic.

### Model staging

Use the optional model preparation cell in notebook 00. It downloads the exact checkpoint revisions used by the workshop into the shared Volume; notebooks 01–04 only load staged files. Weights stay outside Git.

## Benchmark approach

- Use the same videos, sampling rules, and preprocessing for both models.
- Batch frames from multiple clips and load the model once per inference worker.
- Measure several batch sizes and record GPU configuration and concurrency.
- Separate video decoding and preprocessing time from model inference time, then measure the full pipeline.
- Include warm-up and report steady-state performance separately from startup overhead.
- Compare available compute configurations after validating what Serverless GPU supports in the target workspace.

A small model may leave the GPU underused if decoding, data movement, or scheduling dominates. GPU speedups and scaling are benchmark results to measure, not assumptions.

## Documentation

See the [agreed workshop plan](docs/workshop-plan.md) for the four notebook scenarios, acceptance criteria, ownership, tracing demonstration, and optional extensions. Training and fine-tuning are outside the workshop scope.

This README is the entry point for workshop scope, model choices, and current assumptions. Environment setup and fresh-workspace instructions are above; recovery guidance is in the facilitator runbook.
