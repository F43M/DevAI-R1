"""Airflow DAGs to automate dataset creation and publishing."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, List


try:  # pragma: no cover - optional dependency
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except Exception:  # pragma: no cover - optional dependency
    DAG = None  # type: ignore
    PythonOperator = None  # type: ignore


def _output_path(name: str) -> str:
    from scraper_wiki import Config

    return os.path.join(Config.OUTPUT_DIR, name)


def _parse_env_list(name: str, default: List[str]) -> List[str]:
    value = os.environ.get(name)
    if not value:
        return default
    return [v.strip() for v in value.split(",") if v.strip()]


def scrape_data() -> str:
    """Collect pages using :class:`DatasetBuilder`.``

    Returns
    -------
    str
        Path to the raw dataset file saved on disk.
    """
    from scraper_wiki import Config, DatasetBuilder, normalize_category

    langs = _parse_env_list("LANGUAGES", Config.LANGUAGES)
    categories = _parse_env_list("CATEGORIES", list(Config.CATEGORIES))
    builder = DatasetBuilder()
    pages: List[dict] = []
    for lang in langs:
        wiki = DatasetBuilder.get_wikipedia_scraper(lang)
        for cat in categories:
            cat = normalize_category(cat) or cat
            pages.extend(
                wiki.get_category_members(cat)[: Config.MAX_PAGES_PER_CATEGORY]
            )
    builder.build_from_pages(pages, "Coletando paginas")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    raw_path = _output_path("wikipedia_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(builder.dataset, f, ensure_ascii=False, indent=2)
    return raw_path


def postprocess_dataset(ti) -> str:  # pragma: no cover - executed by Airflow
    """Deduplicate and clean the dataset."""
    path = ti.xcom_pull(task_ids="scrape_data")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(path)
    from dq import (
        deduplicate_by_hash,
        deduplicate_by_embedding,
        remove_pii,
        strip_credentials,
    )
    from training.postprocessing import filter_by_complexity

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data, _ = deduplicate_by_hash(data)
    data, _ = deduplicate_by_embedding(data)
    cleaned = []
    for rec in data:
        if "content" in rec:
            rec["content"] = remove_pii(strip_credentials(rec["content"]))
        cleaned.append(rec)
    cleaned = filter_by_complexity(cleaned)
    processed_path = _output_path("wikipedia_processed.json")
    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    return processed_path


def publish_dataset(ti) -> None:  # pragma: no cover - executed by Airflow
    """Publish the processed dataset and optionally fine-tune a model."""
    import mlflow
    from pathlib import Path
    from training import pretrained_utils

    path = ti.xcom_pull(task_ids="postprocess_dataset")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    from scraper_wiki import Config, DatasetBuilder

    builder = DatasetBuilder()
    builder.dataset = data
    builder.save_dataset(format=os.environ.get("DATASET_FORMAT", "hf"))

    dataset_info = Path(Config.OUTPUT_DIR) / "dataset_info.json"
    dataset_version = None
    if dataset_info.exists():
        try:
            dataset_version = json.loads(dataset_info.read_text(encoding="utf-8"))[
                "version"
            ]
        except Exception:
            dataset_version = None

    fine_tune_flag = os.environ.get("FINE_TUNE_MODEL")
    model_name = os.environ.get("MODEL_NAME", "bert-base-uncased")

    with mlflow.start_run():
        if dataset_version:
            mlflow.log_param("dataset_version", dataset_version)
        if fine_tune_flag:
            model_version = pretrained_utils.fine_tune_model(
                Path(path), model_name=model_name
            )
            mlflow.log_param("model_version", model_version)


def create_dag() -> Any:
    """Create the Airflow DAG for dataset creation."""
    if DAG is None or PythonOperator is None:
        raise RuntimeError("Airflow is not installed")

    schedule = os.environ.get("AIRFLOW_SCHEDULE")
    with DAG(
        dag_id="dataset_pipeline",
        start_date=datetime(2024, 1, 1),
        schedule_interval=schedule,
        catchup=False,
    ) as dag:
        scrape = PythonOperator(
            task_id="scrape_data",
            python_callable=scrape_data,
        )
        postprocess = PythonOperator(
            task_id="postprocess_dataset",
            python_callable=postprocess_dataset,
        )
        publish = PythonOperator(
            task_id="publish_dataset",
            python_callable=publish_dataset,
        )
        scrape >> postprocess >> publish
    return dag


try:  # pragma: no cover - optional dependency
    dag = create_dag()
except Exception:
    dag = None
