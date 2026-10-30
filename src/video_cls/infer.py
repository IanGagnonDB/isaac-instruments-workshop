"""pandas UDF factories for frame classification. Model loaded once per task, batched by Arrow.

Executors never import video_cls: the notebooks register the package with
cloudpickle.register_pickle_by_value, so everything a UDF uses must be a module-level name
(shipped by value with the UDF), not an import inside the UDF body.
"""
from typing import Iterator

import numpy as np
import pandas as pd

from video_cls.models import OnnxPredictor, TorchPredictor
from video_cls.preprocess import preprocess_batch, softmax

PRED_SCHEMA = "label int, score float"


def classify_batches(batches, predictor):
    """Core loop shared by both UDFs and by tests: JPEG Series in, (label, score) frames out."""
    for jpegs in batches:
        if len(jpegs) == 0:
            yield pd.DataFrame({"label": pd.Series(dtype=np.int32), "score": pd.Series(dtype=np.float32)})
            continue
        p = softmax(predictor(preprocess_batch(jpegs)))
        yield pd.DataFrame({"label": p.argmax(1).astype(np.int32), "score": p.max(1).astype(np.float32)})


def make_classify(model_path, threads=1):
    """ONNX Runtime classify UDF. Batch size = spark.sql.execution.arrow.maxRecordsPerBatch."""
    from pyspark.sql.functions import pandas_udf

    @pandas_udf(PRED_SCHEMA)
    def classify(batches: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        yield from classify_batches(batches, OnnxPredictor(model_path, threads))

    return classify


def make_classify_torch(model_path, threads=1):
    """PyTorch variant with the same signature: torch.set_num_threads(threads), inference_mode()."""
    from pyspark.sql.functions import pandas_udf

    @pandas_udf(PRED_SCHEMA)
    def classify(batches: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        yield from classify_batches(batches, TorchPredictor(model_path, threads))

    return classify


def make_classifier(model_path, threads=1):
    """Pick the UDF from the artifact suffix: .onnx -> ONNX Runtime, .pt -> PyTorch."""
    factory = make_classify if str(model_path).endswith(".onnx") else make_classify_torch
    return factory(model_path, threads)
