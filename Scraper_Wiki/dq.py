"""Data quality utilities for deduplication and validation."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
from simhash import Simhash, SimhashIndex


def deduplicate_by_hash(records: List[Dict]) -> Tuple[List[Dict], int]:
    """Remove duplicate records using a hash of content.

    Parameters
    ----------
    records: List[Dict]
        Dataset records.

    Returns
    -------
    Tuple[List[Dict], int]
        Unique records and count of removed items.
    """
    seen: set[str] = set()
    unique: List[Dict] = []
    removed = 0

    for rec in records:
        base = rec.get("id") or f"{rec.get('title','')}_{rec.get('language','')}"
        if "content" in rec:
            base += rec["content"][:50]
        h = hashlib.md5(base.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(rec)
        else:
            removed += 1
    return unique, removed


def deduplicate_by_embedding(
    records: List[Dict], threshold: float = 0.95
) -> Tuple[List[Dict], int]:
    """Remove semantically duplicated records using cosine similarity."""
    if not records:
        return records, 0

    embeddings = [rec.get("content_embedding", []) for rec in records]
    if not embeddings or not isinstance(embeddings[0], list):
        return records, 0

    emb = np.array(embeddings, dtype=float)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_emb = emb / norms

    to_remove = set()
    for i in range(len(records)):
        if i in to_remove:
            continue
        sims = np.dot(norm_emb[i], norm_emb[i + 1 :].T)
        for j, sim in enumerate(sims, start=i + 1):
            if sim >= threshold:
                to_remove.add(j)

    unique = [rec for idx, rec in enumerate(records) if idx not in to_remove]
    return unique, len(to_remove)


def deduplicate_by_simhash(
    records: List[Dict], distance: int = 3
) -> Tuple[List[Dict], int]:
    """Remove near-duplicate records using Simhash with LSH.

    Parameters
    ----------
    records: List[Dict]
        Dataset records.
    distance: int, optional
        Maximum Hamming distance to consider records duplicates. Defaults to ``3``.

    Returns
    -------
    Tuple[List[Dict], int]
        Unique records and count of removed items.
    """
    if not records:
        return records, 0

    objs = [(str(i), Simhash(rec.get("content", ""))) for i, rec in enumerate(records)]
    index = SimhashIndex(objs, k=distance)

    to_remove: set[int] = set()
    for i, (_, h) in enumerate(objs):
        if i in to_remove:
            continue
        for dup_id in index.get_near_dups(h):
            j = int(dup_id)
            if j != i and j > i:
                to_remove.add(j)

    unique = [rec for idx, rec in enumerate(records) if idx not in to_remove]
    return unique, len(to_remove)


def detect_leaks_by_hash(records: List[Dict], reference: List[Dict]) -> List[Dict]:
    """Detect records that also appear in the reference dataset using hashing."""
    ref_hashes = set()
    for rec in reference:
        base = rec.get("content", "")[:50] + rec.get("language", "")
        ref_hashes.add(hashlib.md5(base.encode("utf-8")).hexdigest())

    leaks = []
    for rec in records:
        base = rec.get("content", "")[:50] + rec.get("language", "")
        if hashlib.md5(base.encode("utf-8")).hexdigest() in ref_hashes:
            leaks.append(rec)
    return leaks


def detect_leaks_by_embedding(
    records: List[Dict], reference: List[Dict], threshold: float = 0.95
) -> List[Dict]:
    """Detect semantic leaks comparing embeddings with a reference dataset."""
    if not records or not reference:
        return []

    rec_emb = [r.get("content_embedding", []) for r in records]
    ref_emb = [r.get("content_embedding", []) for r in reference]
    if not rec_emb or not ref_emb or not isinstance(rec_emb[0], list):
        return []

    rec_arr = np.array(rec_emb, dtype=float)
    ref_arr = np.array(ref_emb, dtype=float)
    rec_arr = rec_arr / np.linalg.norm(rec_arr, axis=1, keepdims=True)
    ref_arr = ref_arr / np.linalg.norm(ref_arr, axis=1, keepdims=True)

    leaks = []
    for idx, emb in enumerate(rec_arr):
        sims = np.dot(ref_arr, emb)
        if np.any(sims >= threshold):
            leaks.append(records[idx])
    return leaks


def load_sensitive_hashes(
    path: str = "reference/sensitive_embeddings.json",
) -> List[Simhash]:
    """Load hashed embedding simhashes from ``path``."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        hashes = json.load(f)
    return [Simhash(int(h, 16)) for h in hashes]


def _embedding_to_simhash(embedding: List[float]) -> Simhash:
    """Return a Simhash computed from ``embedding``."""
    feats = [f"{i}:{int(v * 100)}" for i, v in enumerate(embedding)]
    return Simhash(feats)


def check_sensitive_embeddings(
    records: List[Dict], reference_hashes: List[Simhash], distance: int = 3
) -> List[Dict]:
    """Return records whose embeddings are near a sensitive reference."""
    if not records or not reference_hashes:
        return []

    flagged: List[Dict] = []
    for rec in records:
        emb = rec.get("content_embedding")
        if not isinstance(emb, list):
            continue
        sh = _embedding_to_simhash(emb)
        for ref in reference_hashes:
            if sh.distance(ref) <= distance:
                flagged.append(rec)
                break
    return flagged


def validate_semantics(records: List[Dict]) -> Tuple[List[Dict], int]:
    """Validate semantic integrity of records."""
    valid: List[Dict] = []
    invalid = 0
    for rec in records:
        ok = True
        if not rec.get("content") or not rec.get("summary"):
            ok = False
        if not rec.get("questions") or not rec.get("answers"):
            ok = False
        if not np.all(np.isfinite(rec.get("content_embedding", []))):
            ok = False
        if not np.all(np.isfinite(rec.get("summary_embedding", []))):
            ok = False
        if ok:
            valid.append(rec)
        else:
            invalid += 1
    return valid, invalid


def complete_missing_fields(records: List[Dict], extra: List[Dict]) -> List[Dict]:
    """Fill empty fields using extra plugin data."""
    lookup = {(item.get("title"), item.get("language")): item for item in extra}
    for rec in records:
        key = (rec.get("title"), rec.get("language"))
        more = lookup.get(key)
        if not more:
            continue
        for k, v in more.items():
            if k == "metadata" and isinstance(v, dict):
                rec_meta = rec.setdefault("metadata", {})
                for mk, mv in v.items():
                    if mk not in rec_meta or rec_meta[mk] in (None, "", []):
                        rec_meta[mk] = mv
            elif k not in rec or rec[k] in (None, "", []):
                rec[k] = v
    return records


def strip_credentials(code: str) -> str:
    """Return ``code`` with tokens and credentials redacted."""
    patterns = [
        r"ghp_[A-Za-z0-9]{36}",
        r"github_pat_[A-Za-z0-9_]{80,}",
        r"sk-[A-Za-z0-9]{16,}",
        r"(?i)(?:api|secret|token|key)[^\n]{0,20}[\'\"]?[A-Za-z0-9_-]{16,}[\'\"]?",
    ]
    text = code
    for pat in patterns:
        text = re.sub(pat, "<REDACTED>", text)
    return text


_PII_PATTERNS = [
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    r"\b(?:\d[ -]?){13,16}\b",  # credit card numbers
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN format
    r"\b\+?\d{1,3}[ -]?\(?\d{1,4}\)?[ -]?\d{1,4}[ -]?\d{1,9}\b",  # phone
]


def remove_pii(text: str) -> str:
    """Return ``text`` with personal data redacted."""

    cleaned = text
    for pat in _PII_PATTERNS:
        cleaned = re.sub(pat, "<PII>", cleaned)
    return cleaned


def detect_code_plagiarism(records: List[Dict], threshold: float = 0.95) -> List[Dict]:
    """Return records whose ``content`` is very similar to others."""
    plagiarized: List[Dict] = []
    for i, rec in enumerate(records):
        code_i = rec.get("content", "")
        for other in records[i + 1 :]:
            code_j = other.get("content", "")
            if not code_i or not code_j:
                continue
            sim = difflib.SequenceMatcher(None, code_i, code_j).ratio()
            if sim >= threshold:
                plagiarized.append(other)
    return plagiarized
