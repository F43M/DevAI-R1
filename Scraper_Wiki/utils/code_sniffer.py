import ast
from typing import List, Tuple


def scan(code: str) -> Tuple[List[str], str]:
    """Analyze ``code`` and return detected problems and a fixed version."""
    problems: List[str] = []
    lines = code.splitlines()
    try:
        tree = ast.parse(code)
    except Exception:
        return problems, code

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in {"eval", "exec"}:
                problems.append(f"Use of {name} on line {node.lineno}")
                idx = node.lineno - 1
                if 0 <= idx < len(lines):
                    if not lines[idx].lstrip().startswith("#"):
                        lines[idx] = "# " + lines[idx]
        if isinstance(node, ast.Compare):
            if len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
                left = node.left
                right = node.comparators[0]
                bool_val = None
                if isinstance(right, ast.Constant) and isinstance(right.value, bool):
                    bool_val = right.value
                elif isinstance(left, ast.Constant) and isinstance(left.value, bool):
                    bool_val = left.value
                if bool_val is not None:
                    problems.append(f"Comparison to {bool_val} on line {node.lineno}")
                    idx = node.lineno - 1
                    if 0 <= idx < len(lines):
                        target = "== True" if bool_val else "== False"
                        replacement = "is True" if bool_val else "is False"
                        lines[idx] = lines[idx].replace(target, replacement)

    fixed_version = "\n".join(lines)
    return problems, fixed_version


__all__ = ["scan"]
