# CPU Serverless workshop redesign

Accepted design: real D2-City data by default; explicit synthetic fallback only. No bundled third-party clips. CPU Serverless ML v6, visible notebook %pip cells, helpers in utils/, three Spark UDF benchmarks, one model in the core workshop. Shared data/model assets and separate participant outputs/models remain.

Implementation sequence:
1. Move setup helpers under utils/. Add strict configuration, dataset provenance and explicit fallback selection. Test failed real preparation, incomplete data, dataset isolation, reuse and synthetic video decoding.
2. Extract CPU inference helpers; replace nested execution of notebook 01 with imports. Add regular Python, scalar pandas and iterator pandas UDF examples with equal sampling and correctness checks. Time completed output writes; label first and repeated rounds without claiming fixed hardware or guaranteed worker reuse.
3. Add 00_configuration, optional preparation and small visible pip overlays in 01–04. Keep warehouse setup in 04. Pin extras without replacing the base ML Torch/pandas/NumPy stack.
4. Rewrite README/runbook/scope, remove obsolete configuration and requirements. Run all tests, a real local checkpoint smoke test and notebook structure checks; independently review the finished change.

Verification boundary: local CPU video/model tests do not establish ML v6 package compatibility, worker access to Volume files or live Spark/UC tracing behavior. Fresh-workspace validation remains required.
