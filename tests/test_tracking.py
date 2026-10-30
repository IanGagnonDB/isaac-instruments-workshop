import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")
from video_cls.tracking import log_cluster, log_single_node  # noqa: E402

OLD, NEW = pd.Timestamp("2026-10-29 10:00"), pd.Timestamp("2026-10-30 10:00")


@pytest.fixture()
def experiment_id(tmp_path, monkeypatch):
    # Throwaway local store; on Databricks the tracking server is the workspace itself.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    yield mlflow.create_experiment("bench")
    mlflow.set_tracking_uri(None)


def single_rows():
    rows = []
    for ts in (OLD, NEW):                                    # the OLD run must be ignored
        for model in ("resnet18", "resnet50"):
            for variant, ms in (("torch", 11.0), ("fp32", 5.0), ("int8", 3.0)):
                for bs in (1, 64):
                    rows.append(dict(run_ts=ts, node_type="Standard_F16s_v2", model=model, backend="b",
                                     variant=variant, batch_size=bs, threads=16, frames=500,
                                     seconds=0.5, ms_per_frame=ms, frames_per_s=1000 / ms))
    return pd.DataFrame(rows)


def exports_rows():
    return pd.DataFrame([dict(model=m, variant=v, size_mb=10.0, top1_agreement_vs_torch=a)
                         for m in ("resnet18", "resnet50") for v, a in (("torch", 1.0), ("fp32", 1.0), ("int8", 0.77))])


def cluster_rows():
    rows = []
    for label, workers, ts in (("2w", 2, OLD), ("2w", 2, NEW), ("8w", 8, NEW)):   # 2w OLD is superseded
        for variant in ("fp32", "int8"):
            rows.append(dict(run_ts=ts, run_label=label, num_workers=workers, total_cores=16 * workers,
                             task_cpus=1, node_type="Standard_F16s_v2", model="resnet18", variant=variant,
                             backend="onnxruntime", n_partitions=48 * workers, batch_size=64, ort_threads=1,
                             clips=400, frames=3992, extract_s=100.0 / workers, inference_s=20.0,
                             inference_frames_per_s=200.0 * workers, end_to_end_frames_per_s=30.0))
    return pd.DataFrame(rows)


def children(experiment_id, parent_id):
    return mlflow.search_runs([experiment_id], filter_string=f"tags.mlflow.parentRunId = '{parent_id}'")


def test_single_node_runs(experiment_id):
    parent_id, logged = log_single_node(experiment_id, single_rows(), exports_rows(), "c.s.bench_single_node")
    assert logged
    kids = children(experiment_id, parent_id)
    assert len(kids) == 12                                   # NEW run only: 2 models x 3 variants x 2 batch sizes
    int8 = kids[(kids["params.variant"] == "int8") & (kids["params.model"] == "resnet18")
                & (kids["params.batch_size"] == "64")].iloc[0]
    assert int8["metrics.ms_per_frame"] == 3.0
    assert int8["metrics.top1_agreement_vs_torch"] == 0.77
    assert int8["tags.source_run_ts"] == NEW.isoformat()
    parent = mlflow.get_run(parent_id)
    assert parent.data.tags["source_table"] == "c.s.bench_single_node"
    assert [a.path for a in mlflow.MlflowClient().list_artifacts(parent_id)] == ["single_node.png"]


def test_cluster_runs_latest_per_label(experiment_id):
    parent_id, logged = log_cluster(experiment_id, cluster_rows(), "c.s.bench_cluster")
    assert logged
    kids = children(experiment_id, parent_id)
    assert sorted(kids["tags.mlflow.runName"]) == ["2w resnet18 fp32", "2w resnet18 int8",
                                                   "8w resnet18 fp32", "8w resnet18 int8"]
    assert set(kids["metrics.inference_frames_per_s"]) == {400.0, 1600.0}


def test_rerun_is_skipped(experiment_id):
    first, _ = log_cluster(experiment_id, cluster_rows(), "c.s.bench_cluster")
    again, logged = log_cluster(experiment_id, cluster_rows(), "c.s.bench_cluster")
    assert (again, logged) == (first, False)
    assert len(mlflow.search_runs([experiment_id])) == 1 + 4  # one parent, four children
