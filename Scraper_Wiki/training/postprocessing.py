"""Dataset post-processing utilities."""

from __future__ import annotations

import ast
from typing import Any, Dict, List

from utils.code import (
    detect_programming_language,
    normalize_indentation,
    remove_comments,
    docstring_to_google,
    parse_function_signature,
)
from utils.ast_tools import get_functions_complexity


def analyze_code_ast(code: str, language: str | None = None) -> Dict[str, Any]:
    """Analyze ``code`` and return AST-based metrics.

    Args:
        code: Source code string.
        language: Optional language name. If ``None`` the language is guessed.

    Returns:
        dict: Dictionary with the cleaned ``content`` string, detected
        ``language`` and the ``complexities`` mapping. For Python code the
        ``docstring`` and ``signature`` fields are also provided.
    """
    lang = language or detect_programming_language(code)
    cleaned = normalize_indentation(remove_comments(code, lang))
    result: Dict[str, Any] = {
        "content": cleaned,
        "language": lang,
        "complexities": get_functions_complexity(cleaned, lang),
    }
    if lang in {"python", "py"}:
        result["signature"] = parse_function_signature(code)
        doc = ""
        try:
            tree = ast.parse(code)
            func = next(
                (
                    n
                    for n in tree.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ),
                None,
            )
            if func:
                ds = ast.get_docstring(func)
                if ds:
                    doc = docstring_to_google(ds)
        except Exception:
            pass
        if doc:
            result["docstring"] = doc
    return result


def filter_by_complexity(
    records: List[Dict],
    min_complexity: int | None = None,
    max_complexity: int | None = None,
) -> List[Dict]:
    """Filter ``records`` by cyclomatic complexity.

    Each record must contain a ``content`` field with source code. Complexity is
    computed using :func:`analyze_code_ast` and records outside the range are
    discarded. Detected metrics are added back to the records that are kept.

    Args:
        records: Dataset records to filter.
        min_complexity: Minimum allowed complexity.
        max_complexity: Maximum allowed complexity.

    Returns:
        list[dict]: Filtered dataset.
    """
    if min_complexity is None and max_complexity is None:
        return records

    filtered: List[Dict] = []
    for rec in records:
        code = rec.get("content", "")
        analysis = analyze_code_ast(code, rec.get("metadata", {}).get("code_language"))
        complexities = analysis.get("complexities", {})
        max_c = max(complexities.values()) if complexities else 0
        if min_complexity is not None and max_c < min_complexity:
            continue
        if max_complexity is not None and max_c > max_complexity:
            continue
        rec.setdefault("metadata", {})
        rec["metadata"]["code_language"] = analysis["language"]
        if complexities:
            rec["metadata"]["complexities"] = complexities
        if analysis.get("docstring"):
            rec["docstring"] = analysis["docstring"]
        if analysis.get("signature"):
            rec["signature"] = analysis["signature"]
        rec["content"] = analysis["content"]
        filtered.append(rec)
    return filtered


__all__ = ["analyze_code_ast", "filter_by_complexity"]
