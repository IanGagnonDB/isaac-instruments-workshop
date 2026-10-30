# Workshop plan — CPU Serverless

## Goal

Give ISAAC a runnable starting point for a video-frame classification pipeline, demonstrate CPU Spark UDF execution choices, and show model versioning and tracing. Two participants share a workspace; environment setup happens live through Git Folder notebooks. No app, training or fine-tuning.

## Agreed design

- Serverless CPU with the ML v6 base environment. No alternative compute-mode matrix.
- Visible `%pip` cells in every exercise, followed by a Python restart before configuration/imports. Keep the base ML stack and install only OpenCV plus the full MLflow package where needed.
- One editable configuration notebook with a Volume path. Trace warehouse selection is required only for exercise 04.
- Real D2-City videos by default. Prepare once when missing; reuse complete data. No redistribution of clips in Git without licensing review.
- Explicit generated-video fallback: change DATASET to synthetic and choose Use synthetic fallback. A download exception never selects it automatically. Separate storage and dataset labels prevent accidental mixing.
- One primary CPU model: ResNet-18. ResNet-50 is an optional architecture comparison. Both checkpoints are pinned. Object classifications are not driving-severity predictions.
- Shared dataset/checkpoints; participant-specific experiments, registered model prefixes and output paths. Keep configuration/revision stable and use independent notebook sessions for concurrent execution.
- Helpers live under utils/. Learning notebooks show the steps participants need to understand; no notebook executes another exercise as a library.

## Exercise 01 — Classify a video

Load one pinned model, decode a representative video, sample frames, preprocess, run CPU inference and save event/frame predictions. Preserve manifest instance IDs plus source-sidecar event IDs. Record source kind, model revision, dependencies and stage timing.

Done when a video produces saved predictions linked to event and frame identities. The default samples three frames; sampling is configurable, including one-frame snapshot events.

## Exercise 02 — Benchmark UDFs

Compare ordinary Python UDF, scalar pandas UDF and iterator pandas UDF on the same manifest/checkpoint/sampling/preprocessing. Cache the model per reused Python worker for all paths; iterator initialization happens before its loop. Scalar/iterator pandas paths batch frames across at most two events at a time. Control inference batch size independently of platform-managed Arrow batches.

Use a sequential driver run as a correctness reference. Require complete event/frame coverage, identical top-1 predictions and confidence agreement within 1e-4. Force inference through completed output writes. Exclude result readback from timed intervals. Record first-use wall time and model loading, then repeated runs in rotating path order. Save hardware reports, dependencies, manifest digest, dataset kind, settings, measurements and charts with variation. Warm-up does not guarantee reused workers; report observed worker model-load times.

Done when all three UDFs pass correctness and produce repeated, charted measurements. Defaults validate functionality; scale only after that. Serverless allocation/cache state is dynamic: do not claim identical physical hardware or attribute cost without evidence.

## Exercise 03 — Model lifecycle

Register two versions of the same pinned checkpoint, preserve preprocessing and the float32 NCHW → ImageNet logits contract, change champion, verify packaged inference, then restore its previous target. Save exact versions and dataset/dependency provenance. Re-runs intentionally add versions.

Done when promotion and rollback resolve/load the expected versions. Accuracy improvement is not implied. Training, held-out evaluation and retraining remain talk tracks. ONNX integration is an optional discussion, not a validated executable example.

## Exercise 04 — Champion inference and tracing

Resolve champion once, load its exact numeric version, score events and persist the version beside predictions. Trace a successful event and a deliberately missing video. Keep tensors/videos out of trace payloads. Store compact IDs, dataset kind, model version, frame counts and summaries. Verify stored success/ERROR traces and show viewer links.

Use the configured catalog/schema for governed UC trace storage, a participant trace experiment, and an explicitly selected SQL warehouse. Trace destination binding is permanent for an experiment. No cross-worker propagation is assumed. Trace spans diagnose processing stages; they are not a controlled compute benchmark.

Done when both traces can be read back and the model version producing the predictions is known.

## Preparation and validation

The designated preparer runs optional_prepare_dataset only when assets are missing, choosing real data or the explicitly selected fallback. Validate a one-frame smoke test before the core exercises. Do not delete partial data automatically. Directory locks discourage simultaneous preparation; they are not storage transactions.

Fresh-workspace acceptance: select ML v6, execute pip/restart cells, confirm Volume access, prepare/reuse assets from two identities, execute all three UDFs, then register/roll back models and export/read traces. Check worker memory, package availability and Volume access in the real workspace. The existing local test environment is not proof of ML v6 compatibility.

Optional extensions: larger real-data workloads, inference batch sizes, Spark partitions, the second architecture, and inference-only benchmarking with pre-extracted frames. Serving endpoints and feature enrichment remain outside the core workshop.
