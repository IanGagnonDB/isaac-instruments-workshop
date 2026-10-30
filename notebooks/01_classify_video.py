# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Classify a video
# MAGIC Turn one video event into stored frame predictions. ImageNet object labels are not driving-severity scores.
# MAGIC Select **Serverless → ML base environment → version 6**. Then **Run All**.
# MAGIC The install cell adds notebook dependencies; the ML base supplies Torch, torchvision, transformers,
# MAGIC NumPy and pandas. The next cell restarts Python before imports. No manual dependency-panel editing.

# COMMAND ----------
# MAGIC %pip install "opencv-python-headless==4.11.0.86"

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_configuration

# COMMAND ----------
from utils import inference as inf
from utils.session import create_session

ctx = create_session(WORKSHOP)
print(f'Dataset: {ctx.dataset}; output: {ctx.output}')

# COMMAND ----------
# MAGIC %md
# MAGIC ## Decode → sample → preprocess → predict
# MAGIC One model is loaded and reused. Sampling, preprocessing and inference are shared with every UDF in 02.
# MAGIC The helper implementation is in utils/inference.py; replace its model/sampling logic for ISAAC's pipeline.

# COMMAND ----------
import pandas as pd
engine, load_seconds = inf.load_engine(ctx.model_path)
inf.warm_engine(engine, ctx.batch_size)
result = inf.run_events(ctx.manifest[:1], engine, ctx.model_id, ctx.revision, ctx.batch_size, ctx.sampling)
inf.save_predictions(f'{ctx.output}/01_predictions.jsonl', result['events'])
inf.write_json(f'{ctx.output}/01_provenance.json', {
    'dataset': ctx.dataset, 'manifest_sha256': inf.manifest_digest(ctx.manifest[:1]),
    'model_id': ctx.model_id, 'revision': ctx.revision, 'dependencies': inf.versions(),
    'sampling': ctx.sampling, 'load_seconds': load_seconds, 'hardware': result['hardware'],
    'stages': result['stages'],
})
display(pd.DataFrame(result['events'][0]['predictions']))
print(f'Saved predictions: {ctx.output}/01_predictions.jsonl')

# COMMAND ----------
# MAGIC %md
# MAGIC ## Sequential reference
# MAGIC The loop above runs in this notebook process. Notebook 02 compares three Spark UDF implementations.
# MAGIC Results from synthetic videos verify mechanics only. No training, detection or severity model is involved.
