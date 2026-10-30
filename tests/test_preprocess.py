import cv2
import numpy as np
import pytest

from video_cls.preprocess import MEAN, STD, preprocess, preprocess_batch, softmax


def test_shape_and_dtype(jpeg):
    x = preprocess(jpeg)
    assert x.shape == (3, 224, 224)
    assert x.dtype == np.float32


def test_normalization_matches_imagenet_stats():
    # A flat image with value v in every channel normalises to (v/255 - mean) / std per channel.
    ok, buf = cv2.imencode(".png", np.full((50, 60, 3), 128, np.uint8))
    x = preprocess(buf.tobytes())
    expected = (128 / 255 - MEAN) / STD
    np.testing.assert_allclose(x[:, 0, 0], expected, atol=1e-5)
    np.testing.assert_allclose(x[:, 111, 87], expected, atol=1e-5)


def test_bgr_to_rgb_channel_order():
    img = np.zeros((32, 32, 3), np.uint8)
    img[..., 2] = 255                                    # pure red in OpenCV's BGR layout
    ok, buf = cv2.imencode(".png", img)
    x = preprocess(buf.tobytes())
    raw = x * STD[:, None, None] + MEAN[:, None, None]   # undo normalisation
    np.testing.assert_allclose(raw[:, 16, 16], [1.0, 0.0, 0.0], atol=1e-5)


def test_batch_stacks(jpeg):
    batch = preprocess_batch([jpeg, jpeg, jpeg])
    assert batch.shape == (3, 3, 224, 224)
    assert batch.dtype == np.float32
    np.testing.assert_array_equal(batch[0], batch[2])


def test_invalid_bytes_raise():
    with pytest.raises(ValueError):
        preprocess(b"not an image")


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, -1000.0]], np.float32))
    np.testing.assert_allclose(p.sum(1), 1.0, rtol=1e-6)
    assert p[1].argmax() == 0
