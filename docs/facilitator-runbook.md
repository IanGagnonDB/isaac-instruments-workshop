# Facilitator runbook

## Fresh workspace checklist

- Clone the Git Folder, grant access and select independent execution sessions for concurrent hands-on use. If working in one shared Python session, take turns running cells.
- Create/select a UC Volume and grant both participants USE CATALOG/SCHEMA and READ/WRITE VOLUME. Set VOLUME_ROOT in 00_configuration; leave DATASET as d2_city.
- Select Serverless, ML base environment, version 6 in the exercise/preparation notebooks. Run from the first cell; the notebook installs extras and restarts Python before importing helpers.
- The designated preparer selects Download D2-City in optional_prepare_dataset. Validate the shared dataset, stage the selected model, and pass one-frame inference. The full source ZIP is downloaded; scoring limits do not shrink it.
- The second participant chooses Reuse existing data and confirms the same data/models are reused. Check distinct participant IDs and model prefixes in independent sessions.
- Run 01 and inspect event/frame identity, model revision, dataset labels and saved output.
- Run 02; all three UDF correctness probes must pass. Inspect first-use and repeated timings, worker model-load times and chart variation. Start small, then increase events for throughput testing.
- Run 03 with CREATE MODEL and experiment access. Verify promotion and rollback; each identity owns separate models.
- Before 04, select a SQL warehouse, set TRACE_WAREHOUSE_ID in 00_configuration and grant CAN USE plus CREATE TABLE in the trace schema. Verify success and ERROR traces, readback and viewer links.
- Re-run preparation and confirm reuse. Re-run 01 and confirm a new execution output folder. Re-run 03 and confirm additional versions are intentional.

## Explicit fallback drill

1. When real preparation fails, confirm the exception includes the underlying error and fallback instructions. It must not create or select synthetic data.
2. Set DATASET='synthetic' in 00_configuration while both participants are stopped.
3. Choose Use synthetic fallback in optional_prepare_dataset and rerun from the top.
4. Confirm three generated MP4s decode, predictions run and every output identifies synthetic data. Inspect both dataset folders to confirm real files were untouched.
5. Explain that generated shapes validate mechanics only. Model staging still requires network access if checkpoints are missing.
6. To return to real footage, select DATASET='d2_city' and reuse/prepare that dataset explicitly. Synthetic files can remain in their separate folder.

## Recovery

**Dependencies:** ensure ML v6, not Standard, is selected. Run the visible pip cell, restart cell and subsequent cells in order. Do not reinstall the old Torch/NumPy/pandas pins. If the overlay conflicts with the selected environment, retain the error and validate package compatibility before changing versions; versions are recorded with results.

**Missing data/model:** open optional_prepare_dataset. Reuse never starts video downloads. Real preparation uses one copy per source video; use a separate larger dataset root if later experimenting with duplication. Model preparation reuses complete local snapshots and can resume missing downloads.

**Old workshop data:** the new real-data location is <Volume>/datasets/d2_city/. Old root-level raw/ and benchmark/ are not moved automatically. Copy a complete, validated old dataset into the new real-data location if you want to reuse it, then choose Download D2-City: existing complete data is reused and labeled, not downloaded again. Do not copy synthetic data into that folder.

**Incomplete dataset:** preparation never overwrites populated raw/benchmark directories. Inspect the error or choose a fresh Volume. Nothing is deleted automatically. Manifest validation checks columns, IDs, paths and nonempty source/benchmark files; the smoke test decodes one event, not every video.

**Interrupted preparation:** ensure the designated preparer's session has stopped before removing a stale lock. Dataset lock: <Volume>/datasets/<source>/.dataset-preparation.lock. Model lock: <Volume>/workshop_models/.preparation.lock. Directory locks depend on storage semantics and are not distributed transactions; designate one preparer regardless.

**UDF failure or memory pressure:** inspect the saved correctness error. Workers need access to staged Volume videos/checkpoints and the notebook dependencies. The worker receives helper definitions by value and does not download files. ResNet-18 and inference batches 1–2 are the conservative starting point; limit sampled frames and avoid large decoded images. A successful local run is not proof of live Serverless worker compatibility.

**Slow benchmark:** a handful of events mostly measures scheduling/startup. Keep it for the functional check, then increase the event limit on real data. Worker replacement and dynamic resource allocation can change results. Record repeated rounds; do not claim guaranteed speedup or fixed hardware.

**Trace failure:** confirm warehouse ID, CAN USE, CREATE TABLE and warehouse startup. Notebook 04 does not silently change storage destinations. If a trace experiment is permanently bound elsewhere, choose a new trace_experiment widget value in 04.

**Configuration changes:** stop concurrent runs, edit 00_configuration together and rerun from the top. The dataset/model selection must be consistent between exercises; 04 resolves the selected architecture's registered model. Shared notebook state is not isolated solely by different usernames.
