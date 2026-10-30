# Fresh-workspace facilitator runbook

Start at README.md and notebook 00. Reserve workshop time for Git credentials, dependency installation, grants, data/model downloads and compute startup. No workspace configuration has been assumed complete in advance.

## Acceptance checklist

- Clone the repository once as a shared Git Folder. Confirm the other participant can open and run notebooks with their own identity.
- Set workshop_config.json to accessible resources in the new workspace; select CPU or compatible GPU compute explicitly.
- Configure dependencies for each notebook and participant session. Use requirements-cpu.txt for the CPU starter; do not replace GPU runtime Torch with CPU dependencies.
- Run 00 with preparation widgets false: check resource access and see missing data reported without starting a download.
- The designated preparer enables dataset/model preparation as needed and reruns 00. Confirm smoke inference succeeds. Reset preparation widgets to false.
- The second participant runs 00 with preparation disabled and reuses the same data/models. Compare displayed model prefixes and participant IDs: they must differ.
- Run 01–04 from each identity. Verify distinct output folders and model names, promotion/rollback, and successful/failed trace readback. SQL warehouse startup may add latency.
- In 02 inspect compatibility.json and output coverage before interpreting charts. Serverless CPU is a partial path; a skipped Ray path is not a successful three-way benchmark.
- Rerun 00 with preparation enabled: a complete dataset is reused. Rerun 01: output goes to a fresh run directory. Rerun 03: new model versions are intentional.

## Recovery

**Missing grants or resources:** create/select the resources with an authorized workspace administrator, grant both participants access, then rerun the failed step. The setup notebook does not create arbitrary catalogs, schemas, compute or permissions.

**Incomplete dataset:** preparation refuses to overwrite populated raw/benchmark directories. Inspect the reported problem or select a fresh Volume in the shared config. Do not delete data while either participant is using it. Validation checks manifest structure, unique IDs, path safety and nonempty referenced files; the smoke test checks decoding of one event, not every video.

**Preparation interrupted:** confirm the designated preparer's command/session has stopped. Inspect partial outputs. Only then manually remove a stale .dataset-preparation.lock or .model-preparation.lock directory from the Volume. These directory locks discourage overlapping setup but are not distributed transactions; always designate one preparer. The model downloader can resume missing downloads; partial datasets require inspection or a fresh location.

**External download unavailable:** notebook 00 needs network access to the dataset source and Hugging Face when assets are missing. Place the required data/model layout into the chosen Volume through an available route, or resolve outbound access. Scoring notebooks do not initiate downloads.

**Dependency/import errors:** confirm the selected environment and Python version, install the documented dependencies and restart the notebook's Python session. Both participants must select an environment in each notebook; settings in one notebook do not configure all the others. The setup smoke test imports the real inference dependencies and fails visibly on incompatible installations.

**GPU/Ray unavailable:** choose an explicit compatible CPU path in workshop_config.json. Run the partial workshop and label unavailable comparisons. Full matched three-way results require compatible compute; do not present partial diagnostics as a matched benchmark.

**Trace failure:** verify trace schema, CREATE TABLE and SQL warehouse CAN USE access; start the selected warehouse if necessary. Notebook 04 uses a separate trace experiment. If an existing experiment is permanently bound to another trace location, select a new trace_experiment widget value in 04.

**Config edits:** stop concurrent runs, edit together, then rerun notebooks from their first cell. Model identity and dataset root must remain consistent between 03 and 04. Notebook-specific benchmark widgets remain available in 02; shared paths/device settings come from workshop_config.json.
