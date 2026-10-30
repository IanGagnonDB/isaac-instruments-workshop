"""The 03_cluster_bench pipeline on local Spark: repartition -> mapInPandas -> pandas UDF."""
import functools

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyspark")

from conftest import write_clip
from video_cls.frames import FRAME_SCHEMA, extract_frames
from video_cls.infer import make_classify, make_classify_torch
from video_cls.models import _torchvision_arch, artifact_path, export_onnx, save_checkpoint


@pytest.fixture(scope="module")
def model_paths(tmp_path_factory):
    out = tmp_path_factory.mktemp("pipeline_models")
    torch.manual_seed(0)
    model = _torchvision_arch("resnet18", pretrained=False).eval()
    save_checkpoint(model, "torchvision", "resnet18", artifact_path(out, "resnet18", "torch"))
    export_onnx(model, artifact_path(out, "resnet18", "fp32"))
    return out


def test_pipeline_end_to_end(spark, tmp_path, model_paths):
    clips = [(f"clip_{i}", write_clip(tmp_path / f"clip_{i}.mp4", seconds=2 + i % 2)) for i in range(5)]
    frames = (spark.createDataFrame(clips, "clip_id string, clip_path string")
              .repartition(4)
              .mapInPandas(functools.partial(extract_frames, n_frames=6), FRAME_SCHEMA)
              .repartition(4)
              .cache())
    assert frames.count() == 30

    results = {}
    for name, factory, variant in [("onnx", make_classify, "fp32"), ("torch", make_classify_torch, "torch")]:
        udf = factory(artifact_path(model_paths, "resnet18", variant), threads=1)
        scored = frames.withColumn("pred", udf("jpeg")).select("clip_id", "frame_idx", "pred.*")
        results[name] = {(r.clip_id, r.frame_idx): (r.label, r.score) for r in scored.collect()}
    assert len(results["onnx"]) == 30
    assert results["onnx"].keys() == results["torch"].keys()
    labels = [(results["onnx"][k][0], results["torch"][k][0]) for k in results["onnx"]]
    assert all(a == b for a, b in labels)
    scores = np.array([[results["onnx"][k][1], results["torch"][k][1]] for k in results["onnx"]])
    np.testing.assert_allclose(scores[:, 0], scores[:, 1], atol=1e-3)
