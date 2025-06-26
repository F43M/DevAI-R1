"""Utilities for dataset enrichment."""

from __future__ import annotations

import ast
import os
import uuid
from typing import Dict, List

from graphviz import Digraph


class _CallGraphVisitor(ast.NodeVisitor):
    """AST visitor building a function call graph."""

    def __init__(self) -> None:
        self.current: str | None = None
        self.calls: Dict[str, set[str]] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[override]
        prev = self.current
        self.current = node.name
        self.calls.setdefault(node.name, set())
        self.generic_visit(node)
        self.current = prev

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore

    def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
        if self.current and isinstance(node.func, ast.Name):
            self.calls.setdefault(self.current, set()).add(node.func.id)
        self.generic_visit(node)


def generate_diagram(code: str, out_dir: str, name: str | None = None) -> str:
    """Return path to a DOT file representing function calls in ``code``."""
    os.makedirs(out_dir, exist_ok=True)
    visitor = _CallGraphVisitor()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = ast.parse("")
    visitor.visit(tree)

    dot = Digraph(comment="Call Graph")
    for func in visitor.calls:
        dot.node(func)
    for caller, callees in visitor.calls.items():
        for callee in callees:
            dot.edge(caller, callee)

    filename = f"{name or uuid.uuid4().hex}.gv"
    path = os.path.join(out_dir, filename)
    dot.save(path)
    return path


def generate_explanations(text: str, lang: str = "en") -> Dict[str, str]:
    """Return explanations at high, medium and low abstraction levels."""
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    joined = " ".join(sentences)
    high = sentences[0] if sentences else joined[:80]
    medium = " ".join(sentences[:3]) if sentences else joined[:200]
    low = joined if len(joined) <= 500 else joined[:500] + "..."
    return {"high": high, "medium": medium, "low": low}


_THEORY_MAP = {
    "sort": "Cormen et al., Introduction to Algorithms",
    "graph": "Sedgewick and Wayne, Algorithms",
    "neural": "Goodfellow et al., Deep Learning",
    "database": "Silberschatz et al., Database System Concepts",
    "compiler": "Aho et al., Compilers: Principles, Techniques, and Tools",
}


def link_theory(text: str) -> List[str]:
    """Return academic references related to ``text`` keywords."""
    matches: List[str] = []
    lower = text.lower()
    for key, ref in _THEORY_MAP.items():
        if key in lower:
            matches.append(ref)
    # remove duplicates preserving order
    seen = set()
    result = []
    for ref in matches:
        if ref not in seen:
            result.append(ref)
            seen.add(ref)
    return result


__all__ = ["generate_diagram", "generate_explanations", "link_theory"]


def generate_synthetic_qa(topic: str, lang: str = "en", n: int = 1) -> List[dict]:
    """Return ``n`` synthetic question/answer pairs for ``topic``.

    The function attempts to use ``transformers`` text generation models if
    available. When the dependency or model loading fails, simple heuristic
    answers based on :func:`generate_explanations` are returned instead.

    Parameters
    ----------
    topic : str
        Topic used to craft the prompt for the language model.
    lang : str
        Language of the desired output.
    n : int
        Number of question/answer pairs to generate.

    Returns
    -------
    List[dict]
        List of dictionaries with ``question`` and ``answer`` keys.
    """

    try:  # pragma: no cover - optional dependency
        from transformers import pipeline

        generator = pipeline("text-generation", model="distilgpt2")
    except Exception:  # pragma: no cover - fallback when transformers missing
        generator = None

    pairs: List[dict] = []
    for _ in range(max(1, n)):
        question = f"What is {topic}?"
        if generator:
            try:  # pragma: no cover - model inference
                prompt = f"{question} Answer:"
                res = generator(prompt, max_new_tokens=40, num_return_sequences=1)[0][
                    "generated_text"
                ]
                answer = res.split("Answer:", 1)[-1].strip()
            except Exception:  # pragma: no cover - inference failure
                answer = generate_explanations(topic, lang)["medium"]
        else:
            answer = generate_explanations(topic, lang)["medium"]

        pairs.append({"question": question, "answer": answer})

    return pairs


__all__.append("generate_synthetic_qa")
