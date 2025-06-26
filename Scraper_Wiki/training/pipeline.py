import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

import mlflow

from .formats import (
    save_arrow_dataset,
    save_arrow_table,
    save_delta_table,
    save_hf_dataset,
    save_tfrecord_dataset,
)


def load_dataset(path: str | Path) -> List[Dict]:
    """Load dataset from JSON file with optional compression."""
    from utils.compression import load_json_file

    return load_json_file(path)


def save_json(data, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def convert_to_conversation_pairs(records: List[Dict]) -> List[Dict]:
    """Return conversation pairs from dataset."""
    pairs = []
    for rec in records:
        qs = rec.get("questions", [])
        ans = rec.get("answers", [])
        for q, a in zip(qs, ans):
            q_text = q["text"] if isinstance(q, dict) else q
            a_text = a["text"] if isinstance(a, dict) else a
            pairs.append({"question": q_text, "answer": a_text})
    return pairs


def convert_to_embeddings(records: List[Dict]) -> List[Dict]:
    """Extract embeddings from dataset."""
    emb = []
    for rec in records:
        emb.append({"id": rec.get("id"), "embedding": rec.get("content_embedding")})
    return emb


def convert_to_triples(records: List[Dict]) -> List[Dict]:
    """Generate knowledge triples using keywords."""
    triples = []
    for rec in records:
        subj = rec.get("title")
        for kw in rec.get("keywords", []):
            triples.append({"subject": subj, "relation": "related_to", "object": kw})
    return triples


def _hash_file(path: Path) -> str:
    """Return MD5 hash of ``path`` for dataset versioning."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run_pipeline(
    dataset_path: str | Path, model_params: Optional[Dict] | None = None
) -> None:
    """Run full conversion pipeline for training and log results with MLflow."""
    dataset_path = Path(dataset_path)
    records = load_dataset(dataset_path)
    base = dataset_path.with_suffix("")

    dataset_version = _hash_file(dataset_path)

    with mlflow.start_run():
        mlflow.log_param("dataset_path", str(dataset_path))
        mlflow.log_param("dataset_version", dataset_version)
        if model_params:
            mlflow.log_params(model_params)

        pairs = convert_to_conversation_pairs(records)
        mlflow.log_metric("num_pairs", len(pairs))
        save_json(pairs, base.with_name(base.name + "_pairs.json"))

        emb = convert_to_embeddings(records)
        mlflow.log_metric("num_embeddings", len(emb))
        save_json(emb, base.with_name(base.name + "_embeddings.json"))

        triples = convert_to_triples(records)
        mlflow.log_metric("num_triples", len(triples))
        save_json(triples, base.with_name(base.name + "_triples.json"))

        save_hf_dataset(records, base.with_name(base.name + "_hf"))
        save_tfrecord_dataset(records, base.with_suffix(".tfrecord"))
        save_arrow_table(records, base.with_suffix(".arrow"))
        save_arrow_dataset(records, base.with_name(base.name + "_arrow_dataset"))
        save_delta_table(records, base.with_name(base.name + "_delta"))
