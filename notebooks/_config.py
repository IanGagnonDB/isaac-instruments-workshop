# Databricks notebook source
# Shared configuration. Run through 00–04; no edits needed here.
from pathlib import Path
import sys

# Git Folder notebook working directory is notebooks/; root is also accepted.
REPOSITORY = Path.cwd()
if not (REPOSITORY / 'workshop_config.json').is_file():
    REPOSITORY = REPOSITORY.parent
if not (REPOSITORY / 'workshop_config.json').is_file():
    raise RuntimeError('Open the notebooks directly inside the cloned Git Folder.')
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
from workshop_support import load_config

WORKSHOP = load_config(REPOSITORY, spark.sql('SELECT current_user()').first()[0])
