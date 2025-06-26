"""Training utilities and preprocessing helpers."""

try:
    from .pipeline import run_pipeline
except Exception:  # pragma: no cover - optional dependency

    def run_pipeline(*_a, **_k):  # type: ignore
        raise RuntimeError("mlflow not available")


from .preprocessing import chunk_text, tokenize_texts
from .postprocessing import analyze_code_ast, filter_by_complexity

__all__ = [
    "run_pipeline",
    "chunk_text",
    "tokenize_texts",
    "analyze_code_ast",
    "filter_by_complexity",
]
