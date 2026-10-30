"""Frame extraction for mapInPandas: seek to a few timestamps, never decode the whole clip."""
from typing import Iterator

import cv2
import numpy as np
import pandas as pd

FRAME_SCHEMA = "clip_id string, frame_idx int, ts_s double, jpeg binary"
COLUMNS = ["clip_id", "frame_idx", "ts_s", "jpeg"]


def clip_duration_s(path):
    """Clip length in seconds from container metadata; 0.0 if the file cannot be opened."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        return cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    finally:
        cap.release()


def _shrink(frame, max_width):
    # 1080p JPEGs are ~300 KB each. Downscaling before the shuffle keeps rows small;
    # preprocess() resizes to 224x224 anyway, so predictions are effectively unchanged.
    if not max_width or frame.shape[1] <= max_width:
        return frame
    height = round(frame.shape[0] * max_width / frame.shape[1])
    return cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)


def extract_clip(clip_id, path, n_frames=10, max_width=640, jpeg_quality=90):
    """Return [(clip_id, frame_idx, ts_s, jpeg_bytes)] for n_frames evenly spaced timestamps."""
    rows = []
    cap = cv2.VideoCapture(path)                       # /Volumes/... path, works on executors
    try:
        if not cap.isOpened():
            return rows
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        for i, t in enumerate(np.linspace(0, dur, n_frames, endpoint=False)):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)   # seek, don't decode every frame
            ok, frame = cap.read()
            if ok:
                ok, buf = cv2.imencode(".jpg", _shrink(frame, max_width),
                                       [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                if ok:
                    rows.append((clip_id, i, float(t), buf.tobytes()))
    finally:
        cap.release()
    return rows


def extract_frames(batches: Iterator[pd.DataFrame], n_frames=10, max_width=640) -> Iterator[pd.DataFrame]:
    """mapInPandas function: (clip_id, clip_path) rows in, one row per sampled frame out.

    Bind parameters with functools.partial(extract_frames, n_frames=..., max_width=...).
    Unreadable clips produce no rows rather than failing the task.
    """
    for pdf in batches:
        rows = []
        for clip_id, path in zip(pdf.clip_id, pdf.clip_path):
            rows.extend(extract_clip(clip_id, path, n_frames, max_width))
        yield pd.DataFrame(rows, columns=COLUMNS)
