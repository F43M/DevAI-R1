from __future__ import annotations

import os

try:  # Airflow is optional when running tests
    from airflow.models import Variable
except Exception:  # pragma: no cover - optional dependency
    Variable = None  # type: ignore

from training.airflow_pipeline import create_dag

# Read configuration from Airflow Variables if available
if Variable is not None:
    langs = Variable.get("languages", default_var=None)
    if langs:
        os.environ["LANGUAGES"] = langs
    cats = Variable.get("categories", default_var=None)
    if cats:
        os.environ["CATEGORIES"] = cats
    storage = Variable.get("storage_backend", default_var=None)
    if storage:
        os.environ["STORAGE_BACKEND"] = storage

dag = create_dag()
