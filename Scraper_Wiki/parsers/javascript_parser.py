from __future__ import annotations

import re

from .base import BaseParser


class JavaScriptParser(BaseParser):
    """Very small JavaScript parser using heuristics."""

    def parse(self, code: str) -> bool:
        # Extremely small syntax check: balanced braces
        stack = []
        pairs = {"{": "}", "(": ")"}
        for ch in code:
            if ch in pairs:
                stack.append(pairs[ch])
            elif ch in pairs.values():
                if not stack or stack.pop() != ch:
                    return False
        return not stack

    def extract_context(self, code: str) -> str:
        m = re.search(r"/\*(.*?)\*/", code, re.DOTALL)
        if m:
            return m.group(1).strip().splitlines()[0]
        m = re.search(r"//(.*)", code)
        if m:
            return m.group(1).strip()
        return ""
