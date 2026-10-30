# Workshop plan

Agreed scope — 2026-10-29. Packaging updated for live setup in a shared Git Folder.

## Goal and audience

Give Ismael a runnable starting point for adapting ISAAC’s video classification pipeline to Databricks, and demonstrate how to benchmark and operate it. The one-day workshop uses notebooks in a shared Git Folder. Environment setup happens live; one designated participant prepares shared data and checkpoints if missing. Deliver notebooks only; no app.

ISAAC currently extracts frames from event videos and runs proprietary PyTorch/ONNX classification models. Public pretrained models substitute for that implementation during the workshop.

## Agreed decisions

- Target Databricks Serverless GPU; this supersedes the initial meeting’s CPU-only constraint. Validate workspace access and supported execution APIs before implementation. Fallback to CPU (if problem with GPU occurs).
- Use `microsoft/resnet-50` as the primary model and `microsoft/resnet-18` as the lightweight baseline.
- Dataset preparation is implemented in [`download_d2_city.py`](../download_d2_city.py).
- Demonstrate image classification on sampled video frames. ImageNet outputs are not driving-severity predictions; label any synthetic severity values explicitly.
- Compare three execution paths: Sequential (reference for the current processing pattern), Ray, and Spark + Pandas UDF. Use shared PyTorch inference logic and validate compute support for each path.
- Skip training and fine-tuning entirely. Cover retraining and evaluation only as a brief lifecycle talk track.
- Keep serving endpoints and real-time feature enrichment optional.
- Include a small manual MLflow tracing demonstration in notebook 04.

## Notebook 01 — Classify a video

**Story:** Turn one event video into traceable frame predictions.

Requirements:

- Consume the output of Ian’s dataset download notebook/script.
- Preserve event IDs from JSON metadata when provided alongside the MP4 files.
- Extract frames with OpenCV and make sampling configurable by event type. Use roughly 10 frames per 30-second clip as the representative case; some events may use one frame.
- Apply model-appropriate preprocessing and run pretrained ResNet-50 inference.
- Persist predictions with event and frame IDs.
- Explain where ISAAC substitutes its own sampling logic and model.

**Done when:** A representative clip produces stored predictions that can be linked back to its event and sampled frames.

## Notebook 02 — Batch and benchmark

**Story:** Compare Sequential (reference/current pattern), Ray, and Spark + Pandas UDF on the same video classification workload.

Requirements:

- Implement three paths with shared decoding, sampling, preprocessing, and PyTorch prediction logic: a sequential loop, Ray workers, and a Spark iterator Pandas UDF. Reuse the loaded model across batches in each worker.
- Run all three with ResNet-50 and ResNet-18 on the same dataset manifest. Compare output coverage and prediction consistency before comparing speed.
- Hold hardware type, CPU/memory/GPU budget, precision, model revision, inference batch size, input/cache state, and timing boundaries constant within each comparison. Record worker concurrency and Spark partitions separately from inference batch size.
- Include warm-up runs and repeated measured runs; separate startup/model loading, steady-state inference, and full-pipeline time. Synchronize GPU work for timing and force Spark execution through completed output writes. Disable tracing for timed runs.
- Record frames/second, events/minute, total runtime, speedup versus Sequential, run-to-run variation, and cost per event where attributable. Note that duplicated videos increase workload size, not dataset diversity.
- Render throughput, speedup, and runtime breakdown charts directly in the notebook, with resource configuration and variation visible.
- Validate compute compatibility first: Ray on Serverless GPU uses AI Runtime integration, while Ray-on-Spark uses `ray.util.spark.setup_ray_cluster`. Pandas UDF support does not establish GPU access inside UDF workers. If needed, use a compatible classic GPU cluster for the matched three-way comparison; retain Serverless GPU as a separate result. CPU fallback comparisons must run all three paths on matched CPU resources. Label unmatched configurations separately.

**Done when:** All three paths produce consistent outputs and repeatable, charted comparisons for both models and selected batch sizes. The Sequential result represents our mock reference, not a measured result from ISAAC’s current VM. No speedup is assumed.

API references: [Ray on AI Runtime](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/ray), [Ray-on-Spark setup](https://docs.databricks.com/aws/en/machine-learning/ray/ray-create), and [iterator Pandas UDFs](https://docs.databricks.com/aws/en/udf/pandas). Validate against the target Azure workspace and installed runtime before implementation.

## Notebook 03 — Manage the model lifecycle

**Story:** Make model selection reproducible and reversible.

Requirements:

- Log the pretrained models and relevant dependency/configuration information with MLflow.
- Register model versions and demonstrate a `champion` alias.
- Record which pretrained checkpoint revision is used.
- Demonstrate an illustrative alias change and rollback. Use compatible input/output contracts; register architectures separately if needed and demonstrate version changes within a registered model.
- Explain that selecting an architecture or changing an alias does not establish improved driving-severity accuracy.
- Give a brief talk track on where training, experiment tracking, held-out evaluation, promotion criteria, and automated retraining would fit.
- Explain that MLflow’s ONNX flavor provides an alternative integration path for ISAAC. Validate an ONNX example before presenting it as runnable functionality; an ONNX notebook is not required.

**Done when:** The champion alias resolves to the intended version, that version can be loaded, and the previous version can be restored.

**Excluded:** Training code, fine-tuning, training datasets, and automated retraining jobs.

## Notebook 04 — Run and inspect the champion

**Story:** Run the selected model and diagnose a slow or failed event.

Requirements:

- Load the registered champion for batch inference.
- Record the exact resolved model version with predictions so later alias changes do not obscure provenance.
- Inspect outputs, stage timing, and failures for a representative event.
- Manually instrument a few demonstration events or batches with MLflow traces.
- Use a trace hierarchy such as:

```text
Process video event
├── Decode video
├── Sample frames
├── Preprocess images
├── Run model inference
└── Aggregate predictions
```

- Record compact metadata: event ID, frame count, batch size, model version, and a prediction summary.
- Keep MP4 files and image tensors outside trace payloads.
- Demonstrate one successful execution and one deliberately failed input.
- Validate trace logging and display in the chosen Serverless GPU execution path before the workshop. Cross-worker trace propagation is not assumed.
- Explain that traces help diagnose execution paths; trace durations alone are not precise GPU compute measurements.

**Done when:** Participants can follow one event through the processing stages, identify a failure, and identify the model version that produced its predictions.

Tracing reference: [Manual and custom tracing](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/manual-tracing).

## Ownership and preparation

**Ian:** Dataset download notebook/script, model preparation, the four scenario notebooks, benchmark content, tracing demo, and workshop agenda.

**Packaging:** Git Folder is the supported workshop entry point; no DAB deployment is required. Two participants share the dataset and pinned checkpoints, while signed-in identity isolates models, experiments and outputs. Notebook 00 guides live setup, optional shared preparation and a smoke test. DAB automation is deferred.

Preparation checklist:

- [ ] Integrate Ian’s dataset script and document the data contract.
- [ ] Validate Sequential, Ray, and Spark + Pandas UDF compute compatibility; size matched CPU/GPU resources and check model memory, worker concurrency, and GPU assignment.
- [ ] Download and pin model checkpoints; keep weights outside Git.
- [ ] Prepare and rehearse all four notebooks in execution order.
- [ ] Validate model registration, alias resolution, and rollback permissions.
- [ ] Validate the tracing demonstration and deliberate failure case.
- [ ] Pre-stage outputs and benchmark results as fallback artifacts.
- [ ] Test the Git Folder flow in a fresh workspace using docs/facilitator-runbook.md.
- [ ] Share the workshop agenda with Dolapo and Ismael.

## Optional extensions

These are outside the core completion criteria:

- A model serving endpoint for a defined online inference use case.
- Real-time feature enrichment for a concrete prediction feature.
- Inference tables, if relevant to an optional serving scenario.
- Additional prediction and benchmark visualizations within the notebooks, with any severity data labeled synthetic.

Object detection and weather detection remain outside scope.

## Remaining implementation decisions

- Starter workload: one copy per source, two scored events; increase only after the setup smoke test. The source ZIP download remains full-sized.
- Matched compute configuration for the three execution paths, batch sizes, benchmark repetitions, and available cost attribution.
- Registered model naming and compatible versioning contracts.
- Trace storage configuration, metadata, and retention appropriate for the workshop.

These decisions refine implementation without adding training or optional extensions to the core scope.