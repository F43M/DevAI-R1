from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class CodeContext:
    """Container for incremental code generation context."""

    history: List[str] = field(default_factory=list)

    def append(self, text: str) -> None:
        """Add generated chunk to the internal history."""
        self.history.append(text)

    def text(self) -> str:
        """Return full context text."""
        return "\n".join(self.history)
