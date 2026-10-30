"""Load ResNet weights, export to ONNX, static int8 quantization, and backend-neutral predictors.

Artifacts in <models_dir>:
  <name>.pt          PyTorch checkpoint (eager model; torchvision or Hugging Face source)
  <name>.onnx        ONNX fp32, opset 17, dynamic batch axis
  <name>_int8.onnx   ONNX Runtime static quantization, QDQ format
"""
import os
from pathlib import Path

import numpy as np

VARIANTS = ("torch", "fp32", "int8")
OPSET = 17
HF_IDS = {"resnet18": "microsoft/resnet-18", "resnet50": "microsoft/resnet-50"}


def artifact_path(models_dir, name, variant):
    suffix = {"torch": ".pt", "fp32": ".onnx", "int8": "_int8.onnx"}
    if variant not in suffix:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    return str(Path(models_dir) / f"{name}{suffix[variant]}")


# ---------------------------------------------------------------- PyTorch side

def _torchvision_arch(arch, pretrained):
    import torchvision
    weights = "DEFAULT" if pretrained else None
    return getattr(torchvision.models, arch)(weights=weights)


def _hf_logits(hf_model):
    """Wrap a Hugging Face ResNet so forward(x) returns plain logits like torchvision."""
    import torch

    class HFLogits(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(pixel_values=x).logits

    return HFLogits(hf_model)


def load_pretrained(arch):
    """Return (eval-mode torch model, source). torchvision first, Hugging Face as fallback.

    The fallback covers workspaces whose egress blocks download.pytorch.org.
    """
    try:
        return _torchvision_arch(arch, pretrained=True).eval(), "torchvision"
    except Exception as exc:                            # network/egress failures vary by platform
        print(f"torchvision weights for {arch} unavailable ({type(exc).__name__}: {exc}); "
              f"falling back to Hugging Face {HF_IDS[arch]}")
    from transformers import ResNetForImageClassification
    hf = ResNetForImageClassification.from_pretrained(HF_IDS[arch])
    return _hf_logits(hf).eval(), "huggingface"


def replace_head(model, num_classes):
    """Swap the final linear layer for a new num_classes head; returns the new layer."""
    import torch
    if hasattr(model, "fc"):                             # torchvision
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model.fc
    head = model.inner.classifier                        # Hugging Face: Sequential(Flatten, Linear)
    head[-1] = torch.nn.Linear(head[-1].in_features, num_classes)
    model.inner.config.num_labels = num_classes
    return head[-1]


def save_checkpoint(model, source, arch, path):
    import torch
    payload = {"source": source, "arch": arch, "state_dict": model.state_dict()}
    if source == "huggingface":
        payload["config"] = model.inner.config.to_dict()
    else:
        payload["num_classes"] = model.fc.out_features
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path):
    """Rebuild an eval-mode eager model from save_checkpoint() output without any network access."""
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["source"] == "huggingface":
        from transformers import ResNetConfig, ResNetForImageClassification
        model = _hf_logits(ResNetForImageClassification(ResNetConfig.from_dict(payload["config"])))
    else:
        model = _torchvision_arch(payload["arch"], pretrained=False)
        if payload["num_classes"] != model.fc.out_features:
            replace_head(model, payload["num_classes"])
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def export_onnx(model, path):
    import torch
    dummy = torch.zeros(1, 3, 224, 224)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False: the TorchScript exporter needs no onnxscript and honours dynamic_axes.
    torch.onnx.export(model, dummy, path, opset_version=OPSET, input_names=["input"],
                      output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)
    import onnx
    onnx.checker.check_model(path)
    return path


# ------------------------------------------------------------ quantization

def quantize_int8(fp32_path, int8_path, calibration):
    """ONNX Runtime static quantization (QDQ, per-channel int8 weights, uint8 activations).

    calibration: float32 array (N, 3, 224, 224) of preprocessed frames from the real clips.
    """
    import tempfile
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, data):
            self._it = iter([{"input": x[None]} for x in data])

        def get_next(self):
            return next(self._it, None)

    with tempfile.TemporaryDirectory() as tmp:
        prepped = os.path.join(tmp, "prepped.onnx")
        quant_pre_process(fp32_path, prepped)            # recommended pass: fuse/shape-infer first
        quantize_static(prepped, int8_path, Reader(np.asarray(calibration, np.float32)),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    return int8_path


# ----------------------------------------------------------------- predictors

class OnnxPredictor:
    def __init__(self, path, threads=1):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads              # Spark already runs one task per core
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.threads = threads

    def __call__(self, batch):
        return self.session.run(None, {self.input_name: batch})[0]


class TorchPredictor:
    def __init__(self, path, threads=1):
        import torch
        torch.set_num_threads(threads)
        self._torch = torch
        self.model = load_checkpoint(path)
        self.threads = threads

    def __call__(self, batch):
        with self._torch.inference_mode():
            return self.model(self._torch.from_numpy(batch)).numpy()


def load_predictor(path, threads=1):
    """Pick the backend from the artifact suffix: .onnx -> ONNX Runtime, .pt -> PyTorch."""
    return OnnxPredictor(path, threads) if str(path).endswith(".onnx") else TorchPredictor(path, threads)


def top1_agreement(reference_logits, other_logits):
    return float(np.mean(np.argmax(reference_logits, 1) == np.argmax(other_logits, 1)))
