import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Local Spark workers are separate Python processes: give them src/ (on a cluster the wheel does this).
SRC = str(Path(__file__).resolve().parents[1] / "src")
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [SRC, os.environ.get("PYTHONPATH")]))
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

FPS = 10
SECONDS = 3


def write_clip(path, seconds=SECONDS, fps=FPS, size=(320, 240)):
    """Synthetic MP4 whose frame n is a flat grey of value 8 * n, so seeks can be checked."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened(), "OpenCV cannot write mp4v on this platform"
    for n in range(seconds * fps):
        writer.write(np.full((size[1], size[0], 3), 8 * n, np.uint8))
    writer.release()
    return str(path)


@pytest.fixture
def clip(tmp_path):
    return write_clip(tmp_path / "clip.mp4")


@pytest.fixture
def jpeg():
    rng = np.random.default_rng(0)
    ok, buf = cv2.imencode(".jpg", rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="session")
def spark():
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession
    session = (SparkSession.builder.master("local[2]").appName("video_cls-tests")
               .config("spark.ui.enabled", "false")
               .config("spark.sql.execution.arrow.maxRecordsPerBatch", "4")
               .getOrCreate())
    yield session
    session.stop()
