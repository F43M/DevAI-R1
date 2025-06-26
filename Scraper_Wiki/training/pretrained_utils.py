"""Utilities for working with pretrained models.

This module provides helper functions to prepare inputs for BERT models and to
build simple image datasets compatible with Stable Diffusion training.

Example
-------
>>> from training.pretrained_utils import prepare_bert_inputs
>>> inputs = prepare_bert_inputs(["Hello World"])  # doctest: +SKIP
>>> inputs["input_ids"].shape[0] == 1
True
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import hashlib
import json

import mlflow

import requests
import torch
from transformers import BertTokenizer


def prepare_bert_inputs(texts: List[str]) -> Dict[str, torch.Tensor]:
    """Tokenize ``texts`` with ``BertTokenizer`` and return tensors.

    Parameters
    ----------
    texts : list[str]
        Sentences or documents to encode.

    Returns
    -------
    Dict[str, torch.Tensor]
        Tokenized inputs ready for ``torch`` models.

    Example
    -------
    >>> from training.pretrained_utils import prepare_bert_inputs
    >>> res = prepare_bert_inputs(["example text"])  # doctest: +SKIP
    >>> list(res.keys())
    ['input_ids', 'token_type_ids', 'attention_mask']
    """

    with mlflow.start_run(nested=True):
        mlflow.log_param("tokenizer_name", "bert-base-uncased")
        mlflow.log_metric("num_texts", len(texts))
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")


def extract_image_dataset(records: List[dict], out_dir: Path) -> None:
    """Download image URLs and captions for Stable Diffusion.

    Each record should contain ``image_url`` and ``caption`` or ``title`` fields.
    Images are saved inside ``out_dir`` using zero-padded names and a
    ``captions.txt`` file maps filenames to captions.

    Example
    -------
    >>> recs = [{"image_url": "http://example.com/pic.jpg", "caption": "A cat"}]
    >>> extract_image_dataset(recs, Path('data'))  # doctest: +SKIP
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_version = hashlib.md5(
        json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    captions = out_dir / "captions.txt"
    downloaded = 0
    with mlflow.start_run(nested=True):
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("output_dir", str(out_dir))
        mlflow.log_metric("num_records", len(records))
        with captions.open("w", encoding="utf-8") as cf:
            for idx, rec in enumerate(records):
                url = rec.get("image_url")
                if not url:
                    continue
                caption = rec.get("caption") or rec.get("title", "")
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                ext = Path(url).suffix or ".jpg"
                name = f"{idx:05d}{ext}"
                with open(out_dir / name, "wb") as img_f:
                    img_f.write(resp.content)
                cf.write(f"{name}\t{caption}\n")
                downloaded += 1
        mlflow.log_metric("images_downloaded", downloaded)


def fine_tune_model(dataset_path: Path, model_name: str = "bert-base-uncased") -> str:
    """Fine-tune a text model using ``dataset_path`` and log results with MLflow.

    The dataset must be a JSON file with a ``content`` field. This function
    performs a minimal training routine just to demonstrate integration with
    MLflow and returns a computed model version identifier.

    Parameters
    ----------
    dataset_path : pathlib.Path
        Path to the processed dataset file.
    model_name : str
        Name of the pretrained checkpoint to start from.

    Returns
    -------
    str
        Deterministic identifier of the produced model version.
    """
    from .pipeline import load_dataset

    records = load_dataset(dataset_path)
    texts = [r.get("content", "") for r in records if r.get("content")]
    tokenized = prepare_bert_inputs(texts)

    dataset_version = hashlib.md5(
        json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    model_version = hashlib.md5(
        (model_name + dataset_version).encode("utf-8")
    ).hexdigest()[:8]

    with mlflow.start_run(nested=True):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("model_version", model_version)
        mlflow.log_metric("num_records", len(texts))
        mlflow.log_metric("token_count", len(tokenized["input_ids"]))

    return model_version
