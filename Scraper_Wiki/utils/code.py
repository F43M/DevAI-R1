import re
import textwrap
import inspect
import ast

from parsers import get_parser


def normalize_indentation(code: str) -> str:
    """Return ``code`` with common indentation removed."""
    dedented = textwrap.dedent(code)
    lines = [line.rstrip() for line in dedented.splitlines()]
    return "\n".join(lines).strip()


def remove_comments(code: str, language: str) -> str:
    """Return ``code`` without comments for ``language``."""
    patterns = []
    lang = language.lower()
    if lang in {"python", "py"}:
        patterns = [r"#.*$", r'""".*?"""', r"'''(.|\n)*?'''"]
    elif lang in {"javascript", "js", "java", "c", "cpp", "go", "php", "rust"}:
        patterns = [r"//.*", r"/\*.*?\*/"]
    else:
        patterns = [r"#.*", r"//.*", r"/\*.*?\*/"]
    text = code
    for pat in patterns:
        flags = re.DOTALL
        if pat == r"#.*$":
            flags = re.MULTILINE
        text = re.sub(pat, "", text, flags=flags)
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def detect_programming_language(code: str) -> str:
    """Guess the programming language of ``code`` using simple heuristics."""
    heuristics = {
        "python": ["def ", "import ", "print("],
        "javascript": ["function ", "console.log", "var ", "let ", "const "],
        "java": ["public class", "System.out.println"],
        "c": ["#include", "printf("],
        "cpp": ["std::", "#include <iostream"],
        "php": ["<?php", "echo "],
        "ruby": ["end", "puts "],
        "go": ["package main", "fmt."],
        "rust": ["fn ", "::"],
    }
    lowered = code.lower()
    for lang, keys in heuristics.items():
        if any(k.lower() in lowered for k in keys):
            return lang
    return "unknown"


def docstring_to_google(docstring: str) -> str:
    """Convert ``docstring`` written in reST style to Google style."""
    lines = textwrap.dedent(docstring).strip().splitlines()
    params: list[tuple[str, str, str | None]] = []
    types: dict[str, str] = {}
    returns = ""
    rtype = ""
    other = []
    for line in lines:
        m = re.match(r":param\s+(\w+)\s*:\s*(.*)", line)
        if m:
            params.append((m.group(1), m.group(2), None))
            continue
        m = re.match(r":type\s+(\w+)\s*:\s*(.*)", line)
        if m:
            types[m.group(1)] = m.group(2)
            continue
        m = re.match(r":returns?\s*:\s*(.*)", line)
        if m:
            returns = m.group(1)
            continue
        m = re.match(r":rtype\s*:\s*(.*)", line)
        if m:
            rtype = m.group(1)
            continue
        other.append(line)

    params = [(n, d, types.get(n)) for n, d, _ in params]

    out_lines = [l for l in other if l.strip()]
    if params:
        out_lines += ["", "Args:"]
        for name, desc, typ in params:
            t = f" ({typ})" if typ else ""
            out_lines.append(f"    {name}{t}: {desc}".rstrip())
    if returns:
        out_lines += ["", "Returns:"]
        t = f" ({rtype})" if rtype else ""
        out_lines.append(f"    {returns}{t}".rstrip())
    return "\n".join(out_lines).strip()


def parse_function_signature(obj: object | str) -> str:
    """Return the function signature from ``obj``."""
    if callable(obj):
        try:
            sig = inspect.signature(obj)
            return f"{obj.__name__}{sig}"
        except Exception:
            return ""
    if isinstance(obj, str):
        try:
            tree = ast.parse(obj)
        except Exception:
            return ""
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                if node.args.vararg:
                    args.append("*" + node.args.vararg.arg)
                if node.args.kwonlyargs:
                    args.extend(a.arg for a in node.args.kwonlyargs)
                if node.args.kwarg:
                    args.append("**" + node.args.kwarg.arg)
                return f"{node.name}({', '.join(args)})"
    return ""


def parse_with_language(code: str, language: str) -> bool:
    """Parse ``code`` using the parser for ``language``."""
    parser = get_parser(language)
    if parser:
        return parser.parse(code)
    return False


def extract_context_from_code(code: str, language: str) -> str:
    """Return context extracted from ``code`` using ``language`` parser."""
    parser = get_parser(language)
    if parser:
        return parser.extract_context(code)
    return ""


def parse_with_detected_language(code: str) -> tuple[str, bool]:
    """Detect language of ``code`` and parse it.

    Returns the detected language and whether parsing succeeded.
    """
    lang = detect_programming_language(code)
    return lang, parse_with_language(code, lang)
