"""Export -> int8 quantization -> predictors, on a randomly initialised ResNet-18 (no downloads)."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from video_cls.infer import classify_batches
from video_cls.models import (artifact_path, export_onnx, load_predictor, quantize_int8, replace_head,
                              save_checkpoint, top1_agreement, _torchvision_arch)
from video_cls.preprocess import preprocess_batch


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    torch.manual_seed(0)
    model = _torchvision_arch("resnet18", pretrained=False).eval()
    save_checkpoint(model, "torchvision", "resnet18", artifact_path(out, "resnet18", "torch"))
    export_onnx(model, artifact_path(out, "resnet18", "fp32"))
    calibration = np.random.default_rng(0).standard_normal((8, 3, 224, 224)).astype(np.float32)
    quantize_int8(artifact_path(out, "resnet18", "fp32"), artifact_path(out, "resnet18", "int8"), calibration)
    return out, calibration


def test_onnx_fp32_matches_torch(artifacts):
    out, x = artifacts
    ref = load_predictor(artifact_path(out, "resnet18", "torch"))(x)
    onnx = load_predictor(artifact_path(out, "resnet18", "fp32"))(x)
    assert ref.shape == onnx.shape == (8, 1000)
    np.testing.assert_allclose(ref, onnx, rtol=1e-3, atol=1e-3)
    assert top1_agreement(ref, onnx) == 1.0


def test_int8_runs_with_dynamic_batch(artifacts):
    out, x = artifacts
    predictor = load_predictor(artifact_path(out, "resnet18", "int8"))
    assert predictor(x[:1]).shape == (1, 1000)
    assert predictor(x).shape == (8, 1000)


def test_classify_batches_contract(artifacts, jpeg):
    out, _ = artifacts
    predictor = load_predictor(artifact_path(out, "resnet18", "fp32"))
    batches = [pd.Series([jpeg] * 3), pd.Series([], dtype=object)]
    frames = list(classify_batches(iter(batches), predictor))
    assert list(frames[0].columns) == ["label", "score"] and len(frames[0]) == 3
    assert frames[0].label.between(0, 999).all() and frames[0].score.between(0, 1).all()
    assert frames[1].empty
    logits = predictor(preprocess_batch([jpeg]))
    assert frames[0].label.iloc[0] == logits.argmax()


def test_two_class_head_roundtrip(tmp_path):
    model = _torchvision_arch("resnet18", pretrained=False).eval()
    replace_head(model, 2)
    path = artifact_path(tmp_path, "stopsign_resnet18", "torch")
    save_checkpoint(model.eval(), "torchvision", "resnet18", path)
    x = np.zeros((2, 3, 224, 224), np.float32)
    assert load_predictor(path)(x).shape == (2, 2)
