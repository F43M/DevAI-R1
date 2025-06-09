from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover - optional dependency
    LogisticRegression = None

from .config import config, logger
from .memory import MemoryManager

MODEL_FILE = Path("intent_model.pkl")

_memory: MemoryManager | None = None
_model: LogisticRegression | None = None


def _get_memory() -> MemoryManager:
    global _memory
    if _memory is None:
        _memory = MemoryManager(config.MEMORY_DB, config.EMBEDDING_MODEL, model=None, index=None)
    return _memory


def _load_model() -> LogisticRegression | None:
    global _model
    if _model is None and MODEL_FILE.exists():
        with MODEL_FILE.open("rb") as f:
            _model = pickle.load(f)
    return _model


def train_intent_model(samples: List[Tuple[str, str]]) -> None:
    """Treina classificador de intenção e salva em intent_model.pkl."""
    if LogisticRegression is None or np is None:
        raise RuntimeError("sklearn/numpy não instalados")
    mem = _get_memory()
    X = [mem._get_embedding(text) for text, _ in samples]
    y = [intent for _, intent in samples]
    X = np.array(X)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    with MODEL_FILE.open("wb") as f:
        pickle.dump(clf, f)
    logger.info("intent_model_trained", samples=len(samples))
    global _model
    _model = clf


def predict_intent(text: str) -> str:
    """Retorna a intenção prevista utilizando o modelo treinado."""
    if np is None:
        raise RuntimeError("numpy não instalado")
    model = _load_model()
    if model is None:
        raise FileNotFoundError("intent_model.pkl not found")
    mem = _get_memory()
    vec = np.array([mem._get_embedding(text)])
    return model.predict(vec)[0]
