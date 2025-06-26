"""Preprocessing utilities for training datasets."""

from __future__ import annotations

from typing import List, Dict


def tokenize_texts(texts: List[str], tokenizer_name: str = "bert-base-uncased") -> Dict:
    """Tokenize a list of texts using a Hugging Face tokenizer.

    Parameters
    ----------
    texts : list[str]
        Text samples to tokenize.
    tokenizer_name : str, optional
        Name of the tokenizer to load, by default ``"bert-base-uncased"``.

    Returns
    -------
    dict
        Tokenization output from the tokenizer.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        return tok(texts, padding=True, truncation=True)
    except Exception:  # pragma: no cover - optional deps
        return {"input_ids": [t.split() for t in texts]}


def chunk_text(text: str, max_length: int, overlap: int = 0) -> List[str]:
    """Split ``text`` into word chunks of ``max_length`` tokens.

    Parameters
    ----------
    text : str
        Input text.
    max_length : int
        Maximum number of words per chunk.
    overlap : int, optional
        Number of overlapping words between consecutive chunks.

    Returns
    -------
    list[str]
        List of text chunks.
    """
    words = text.split()
    chunks: List[str] = []
    start = 0
    step = max_length - overlap if overlap < max_length else max_length
    while start < len(words):
        end = start + max_length
        chunks.append(" ".join(words[start:end]))
        start += step
    return chunks
