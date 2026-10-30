"""JPEG bytes -> normalized CHW float32, identical for PyTorch and ONNX Runtime."""
import cv2
import numpy as np

SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(jpeg):
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode JPEG bytes")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SIZE, SIZE)).astype(np.float32) / 255
    return ((img - MEAN) / STD).transpose(2, 0, 1)            # HWC -> CHW


def preprocess_batch(jpegs):
    """Stack a sequence of JPEG bytes into an (N, 3, 224, 224) float32 array."""
    return np.stack([preprocess(b) for b in jpegs]).astype(np.float32, copy=False)


def softmax(logits):
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)
