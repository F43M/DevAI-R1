from __future__ import annotations

import re

from .base import BaseParser


class JavaParser(BaseParser):
    """Very small Java parser using heuristics."""

    def parse(self, code: str) -> bool:
        braces = 0
        for ch in code:
            if ch == "{":
                braces += 1
            elif ch == "}":
                braces -= 1
                if braces < 0:
                    return False
        return braces == 0

    def extract_context(self, code: str) -> str:
        m = re.search(r"/\*(.*?)\*/", code, re.DOTALL)
        if m:
            return m.group(1).strip().splitlines()[0]
        m = re.search(r"//(.*)", code)
        if m:
            return m.group(1).strip()
        return ""
