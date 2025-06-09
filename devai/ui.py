from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterable, List

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory
except Exception:  # pragma: no cover - optional dependency
    PromptSession = None  # type: ignore


class CLIUI:
    """Interactive command line UI using Rich and prompt_toolkit."""

    def __init__(self, plain: bool = False, commands: Iterable[str] | None = None) -> None:
        self.plain = plain
        self.console = Console()
        self.history: List[str] = []
        if not plain and PromptSession is not None:
            history_file = Path.home() / ".devai_history"
            completer = WordCompleter(list(commands or []), ignore_case=True)
            self.session = PromptSession(history=FileHistory(str(history_file)), completer=completer)
        else:
            self.session = None

    async def read_command(self, prompt: str = ">>> ") -> str:
        if self.session is not None:
            return (await self.session.prompt_async(prompt)).strip()
        return input(prompt).strip()

    def show_history(self) -> None:
        if self.plain:
            for line in self.history:
                print(line)
        else:
            panel = Panel("\n".join(self.history[-20:]), title="Histórico", height=20)
            self.console.print(panel)

    def add_history(self, line: str) -> None:
        self.history.append(line)

    def render_diff(self, diff: str) -> None:
        if self.plain:
            print(diff)
        else:
            self.console.print(Syntax(diff, "diff"))

    @asynccontextmanager
    async def loading(self, message: str = "Gerando..."):
        if self.plain:
            print(message)
            yield
        else:
            with self.console.status(message):
                yield

