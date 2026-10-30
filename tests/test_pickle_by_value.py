"""Git Folder mode: no wheel. Executors cannot import video_cls; the driver ships it by value.

Runs the 03 pipeline in a subprocess whose Spark workers have no src/ on their path, exactly like a
cluster where the notebooks only did sys.path.insert(0, "../src") on the driver.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyspark")

from conftest import write_clip
from video_cls.models import _torchvision_arch, artifact_path, export_onnx, save_checkpoint

SRC = str(Path(__file__).resolve().parents[1] / "src")

SCRIPT = textwrap.dedent("""
    import functools, importlib.util, sys
    sys.path.insert(0, sys.argv[1])                  # driver only, like the notebook bootstrap
    import video_cls
    from pyspark import cloudpickle
    cloudpickle.register_pickle_by_value(video_cls)

    from pyspark.sql import SparkSession
    from video_cls.frames import FRAME_SCHEMA, extract_frames
    from video_cls.infer import make_classify, make_classify_torch

    spark = SparkSession.builder.master("local[2]").config("spark.ui.enabled", "false").getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "4")

    def worker_can_import(batches):
        import pandas as pd
        for _ in batches:
            yield pd.DataFrame({"found": [importlib.util.find_spec("video_cls") is not None]})
    found = spark.range(1).mapInPandas(worker_can_import, "found boolean").first().found
    assert not found, "workers must not see video_cls for this test to mean anything"

    clips = [("a", sys.argv[2]), ("b", sys.argv[2])]
    frames = (spark.createDataFrame(clips, "clip_id string, clip_path string").repartition(2)
              .mapInPandas(functools.partial(extract_frames, n_frames=3), FRAME_SCHEMA).cache())
    assert frames.count() == 6
    for factory, path in ((make_classify, sys.argv[3]), (make_classify_torch, sys.argv[4])):
        rows = frames.withColumn("p", factory(path)("jpeg")).select("p.*").collect()
        assert len(rows) == 6 and all(0 <= r.label < 1000 for r in rows)
    print("OK")
""")


def test_executors_without_package_on_path(tmp_path):
    model = _torchvision_arch("resnet18", pretrained=False).eval()
    save_checkpoint(model, "torchvision", "resnet18", artifact_path(tmp_path, "resnet18", "torch"))
    export_onnx(model, artifact_path(tmp_path, "resnet18", "fp32"))
    clip = write_clip(tmp_path / "clip.mp4")
    script = tmp_path / "run.py"
    script.write_text(SCRIPT)

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYSPARK_PYTHON"] = sys.executable
    result = subprocess.run(
        [sys.executable, str(script), SRC, clip, artifact_path(tmp_path, "resnet18", "fp32"),
         artifact_path(tmp_path, "resnet18", "torch")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-3000:]
