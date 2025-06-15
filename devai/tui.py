from __future__ import annotations

import inspect
from typing import Iterable

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, TextLog

from .core import CodeMemoryAI
from .ui import CLIUI
from .feedback import FeedbackDB
from .command_router import COMMANDS, handle_default

"""Simple Textual user interface for interacting with DevAI."""


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
        """Initialize the TUI and underlying CLI components."""
        super().__init__()
        self.ai = ai or CodeMemoryAI()
        cmds = [f"/{c}" for c in COMMANDS]
        self.cli = cli_ui or CLIUI(plain=False, commands=cmds, log=log)
        self.feedback_db = FeedbackDB()
        # Reuse the console from Textual
        self.cli.console = self.console
        self.history_panel: TextLog
        self.diff_panel: TextLog
        self.progress_panel: TextLog
        self.input: Input
        self.command_history: list[str] = []
        self.history_index = 0
        self.commands: list[str] = cmds
        self.completion_matches: list[str] = []
        self.completion_index = 0

    def compose(self) -> ComposeResult:
        """Create application layout with history, progress and diff panels."""
        self.history_panel = TextLog(highlight=False, name="history")
        self.progress_panel = TextLog(highlight=False, name="progress", height=3)
        self.input = Input(placeholder="Digite um comando...", name="input")
        left = Vertical(self.history_panel, self.progress_panel, self.input)
        self.diff_panel = TextLog(highlight=True, name="diff", wrap=False)
        self.cli.diff_panel = self.diff_panel
        self.cli.progress_handler = self._progress_update
        yield Horizontal(left, self.diff_panel)

    def _progress_update(self, message: str) -> None:
        """Write progress messages to the UI panel."""
        try:
            if message == "done":
                self.progress_panel.clear()
            else:
                self.progress_panel.write(message)
        except Exception:
            pass

    async def on_mount(self) -> None:
        """Populate command history and prepare completion list."""
        self.cli.load_history()
        if self.cli.session and getattr(self.cli.session, "completer", None):
            words = getattr(self.cli.session.completer, "words", [])
            self.commands = list(words)
        for line in self.cli.history:
            self.history_panel.write(line)
            if line.startswith(">>> "):
                self.command_history.append(line[4:])
        self.history_index = len(self.command_history)

    async def action_submit(self) -> None:
        """Handle user input submission from the prompt."""
        text = self.input.value.strip()
        if not text:
            return
        self.input.value = ""
        self.cli.add_history(f">>> {text}")
        self.history_panel.write(f">>> {text}")
        self.command_history.append(text)
        self.history_index = len(self.command_history)
        self.history_panel.write("", scroll_end=False)
        if text.startswith("/"):
            parts = text[1:].split(" ", 1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            handler = COMMANDS.get(cmd)
            if handler:
                should_exit = await handler(
                    self.ai, self.cli, args, plain=False, feedback_db=self.feedback_db
                )
                if should_exit:
                    await self.action_quit()
                return
        await handle_default(
            self.ai,
            self.cli,
            text,
            plain=False,
            feedback_db=self.feedback_db,
            side_by_side=True,
        )

    async def on_key(self, event) -> None:
        """Provide basic history navigation and completion."""
        if self.focused is not self.input:
            return
        key = getattr(event, "key", "")
        if key == "up":
            if self.command_history and self.history_index > 0:
                self.history_index -= 1
                self.input.value = self.command_history[self.history_index]
            event.stop()
        elif key == "down":
            if (
                self.command_history
                and self.history_index < len(self.command_history) - 1
            ):
                self.history_index += 1
                self.input.value = self.command_history[self.history_index]
            else:
                self.history_index = len(self.command_history)
                self.input.value = ""
            event.stop()
        elif key == "tab":
            prefix = self.input.value
            if not self.completion_matches or not all(
                m.startswith(prefix) for m in self.completion_matches
            ):
                self.completion_matches = [
                    c for c in self.commands if c.startswith(prefix)
                ]
                self.completion_index = 0
            if self.completion_matches:
                self.input.value = self.completion_matches[self.completion_index]
                self.completion_index = (self.completion_index + 1) % len(
                    self.completion_matches
                )
                event.stop()

    async def action_quit(self) -> None:
        """Clear progress panel and quit the app."""
        self.progress_panel.clear()
        method = super().action_quit
        result = method()
        if inspect.isawaitable(result):
            await result
