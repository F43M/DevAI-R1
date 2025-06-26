"""Active learning utilities for prioritizing scraping tasks."""

from __future__ import annotations

import heapq
from typing import Iterable, List, Dict, Tuple

import numpy as np

from scraper_wiki import NLPProcessor
from task_queue import publish


class ActiveLearner:
    """Score pages by diversity and publish high-value tasks first."""

    def __init__(self) -> None:
        self.embedding_model = NLPProcessor.get_embedding_model()
        self.embeddings: List[np.ndarray] = []

    def _compute_embedding(self, text: str) -> np.ndarray:
        emb = self.embedding_model.encode(text, show_progress_bar=False)
        arr = np.asarray(emb)
        return arr[0] if arr.ndim > 1 else arr

    def _diversity_score(self, emb: np.ndarray) -> float:
        if not self.embeddings:
            return 1.0
        matrix = np.vstack(self.embeddings)
        dots = matrix @ emb
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(emb)
        sim = (dots / (norms + 1e-9)).max()
        return 1.0 - float(sim)

    def update_from_record(self, record: Dict) -> None:
        text = record.get("content")
        emb = record.get("content_embedding")
        if emb is None and text:
            emb = self._compute_embedding(text)
        else:
            emb = np.asarray(emb, dtype=float)
        if emb is not None:
            self.embeddings.append(emb)

    def score_pages(
        self, pages: Iterable[Dict[str, str]]
    ) -> List[Tuple[float, Dict[str, str]]]:
        scored = []
        for page in pages:
            text = page.get("title") or page.get("url", "")
            emb = self._compute_embedding(text)
            score = self._diversity_score(emb)
            scored.append((score, page))
        return scored

    def enqueue(
        self, pages: Iterable[Dict[str, str]], queue: str = "scrape_tasks"
    ) -> None:
        scored = self.score_pages(pages)
        heap = [(-score, page) for score, page in scored]
        heapq.heapify(heap)
        while heap:
            _, page = heapq.heappop(heap)
            publish(queue, page)


__all__ = ["ActiveLearner"]
