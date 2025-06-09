from __future__ import annotations

import re
from typing import Iterable

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, TextLog

from .core import CodeMemoryAI
from .ui import CLIUI


class TUIApp(App):
    """Simple Textual-based UI for DevAI."""

    BINDINGS = [("enter", "submit", "Enviar"), ("ctrl+c", "quit", "Sair")]

    def __init__(
        self,
        ai: CodeMemoryAI | None = None,
        cli_ui: CLIUI | None = None,
        *,
        log: bool = True,
    ) -> None:
        super().__init__()
        self.ai = ai or CodeMemoryAI()
        self.cli = cli_ui or CLIUI(plain=False, commands=None, log=log)
        # Reuse the console from Textual
        self.cli.console = self.console
        self.history_panel: TextLog
        self.diff_panel: TextLog
        self.progress_panel: TextLog
        self.input: Input

    def compose(self) -> ComposeResult:
        self.history_panel = TextLog(highlight=False, name="history")
        self.progress_panel = TextLog(highlight=False, name="progress", height=3)
        self.input = Input(placeholder="Digite um comando...", name="input")
        left = Vertical(self.history_panel, self.progress_panel, self.input)
        self.diff_panel = TextLog(highlight=True, name="diff")
        self.cli.diff_panel = self.diff_panel
        self.cli.progress_handler = self._progress_update
        yield Horizontal(left, self.diff_panel)

    def _progress_update(self, message: str) -> None:
        try:
            self.progress_panel.write(message)
        except Exception:
            pass

    async def on_mount(self) -> None:
        self.cli.load_history()
        for line in self.cli.history:
            self.history_panel.write(line)

    async def action_submit(self) -> None:
        text = self.input.value.strip()
        if not text:
            return
        self.input.value = ""
        self.cli.add_history(f">>> {text}")
        self.history_panel.write(f">>> {text}")
        if text.lower() == "/sair":
            await self.action_quit()
            return
        async with self.cli.loading("Gerando resposta..."):
            response = await self.ai.generate_response(
                text, double_check=self.ai.double_check
            )
        self.cli.add_history(response)
        self.history_panel.write(response)
        is_patch = bool(
            re.search(r"\ndiff --git", response)
            or re.search(r"^[+-](?![+-])", response, re.MULTILINE)
        )
        if is_patch:
            self.cli.render_diff(response)
        else:
            self.diff_panel.clear()
