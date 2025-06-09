from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.table import Table

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter, PathCompleter, Completer
    from prompt_toolkit.history import FileHistory
    try:  # pragma: no cover - optional dependency
        from prompt_toolkit.completion import merge_completers
    except Exception:  # pragma: no cover - older versions
        merge_completers = None  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PromptSession = None  # type: ignore


class CLIUI:
    """Interactive command line UI using Rich and prompt_toolkit."""

    def __init__(
        self,
        plain: bool = False,
        commands: Iterable[str] | None = None,
        *,
        log: bool = True,
    ) -> None:
        self.plain = plain
        self.console = Console()
        self.history: List[str] = []
        self.log = log
        self.log_path = Path.home() / ".devai_chat.log"
        self.diff_panel = None
        self.progress_handler = None
        self.remember_choice: bool = False
        self.remember_expires: str | None = None
        if not plain and PromptSession is not None:
            history_file = Path.home() / ".devai_history"
            cmd_completer = WordCompleter(list(commands or []), ignore_case=True)
            path_completer = PathCompleter(expanduser=True)

            if merge_completers is not None:
                completer = merge_completers([cmd_completer, path_completer])
            else:
                class _MergedCompleter(Completer):
                    def __init__(self, comps: list[Completer]):
                        self.completers = comps

                    def get_completions(self, document, complete_event):
                        for c in self.completers:
                            yield from c.get_completions(document, complete_event)

                completer = _MergedCompleter([cmd_completer, path_completer])

            self.session = PromptSession(
                history=FileHistory(str(history_file)),
                completer=completer,
            )
        else:
            self.session = None

    def load_history(self, lines: int | None = 20) -> None:
        """Load lines from the log file.

        When ``lines`` is ``None`` the entire file is loaded.
        """
        if not self.log:
            return
        if self.log_path.exists():
            try:
                content = self.log_path.read_text().splitlines()
            except Exception:
                return
            if lines is None:
                self.history.extend(content)
            else:
                self.history.extend(content[-lines:])

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
        if self.log:
            try:
                with self.log_path.open("a") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def get_log(self) -> list[str]:
        """Return the entire CLI log as a list of lines."""
        if not self.log:
            return []
        if not self.log_path.exists():
            return []
        try:
            return self.log_path.read_text().splitlines()
        except Exception:
            return []

    def render_diff(
        self,
        diff: str,
        *,
        side_by_side: bool = False,
        collapse: bool = True,
        scroll: bool = True,
    ) -> None:
        if self.plain:
            print(diff)
            return

        def _collapse(text: str, context: int = 3) -> str:
            lines = text.splitlines()
            important = [
                i
                for i, l in enumerate(lines)
                if l.startswith("+") or l.startswith("-") or l.startswith("@@")
            ]
            if not important:
                return text
            result: list[str] = []
            last = 0
            for idx in important:
                start = max(idx - context, last)
                if start > last:
                    result.append("...")
                result.extend(lines[start : idx + context + 1])
                last = idx + context + 1
            if last < len(lines):
                result.append("...")
            return "\n".join(result)

        collapsed = _collapse(diff) if collapse else diff

        if side_by_side:
            lines = collapsed.splitlines()
            table = Table.grid(expand=True)
            table.add_column("Original")
            table.add_column("Novo")
            for line in lines:
                if line == "...":
                    row = Text("...", style="dim")
                    table.add_row(row, row)
                elif line.startswith("-") and not line.startswith("---"):
                    table.add_row(Text(line[1:], style="red"), Text(""))
                elif line.startswith("+") and not line.startswith("+++"):
                    table.add_row(Text(""), Text(line[1:], style="green"))
                else:
                    style = "bold" if line.startswith("@@") or line.startswith("diff") or line.startswith("index") or line.startswith("---") or line.startswith("+++") else None
                    t = Text(line, style=style)
                    table.add_row(t, t)
            renderable = table
        else:
            renderable = Syntax(collapsed, "diff")

        if self.diff_panel is not None:
            try:
                self.diff_panel.clear()
                self.diff_panel.write(renderable, scroll_end=scroll)
            except Exception:
                pass
        else:
            panel = Panel(renderable, height=20, title="Diff")
            self.console.print(panel)

    @asynccontextmanager
    async def loading(self, message: str = "Gerando..."):
        if self.plain:
            print(message)
            yield
        else:
            if self.progress_handler:
                try:
                    self.progress_handler(message)
                except Exception:
                    pass
            with self.console.status(message):
                yield
            if self.progress_handler:
                try:
                    self.progress_handler("done")
                except Exception:
                    pass

    @asynccontextmanager
    async def progress(self, message: str = "Processando..."):
        """Display a progress bar for long running tasks."""
        if self.plain:
            print(message)

            def _update(_pct: float | None = None, stage: str | None = None) -> None:
                if stage:
                    print(stage)
                if self.progress_handler and stage:
                    try:
                        self.progress_handler(stage)
                    except Exception:
                        pass

            try:
                yield _update
            finally:
                if self.progress_handler:
                    try:
                        self.progress_handler("done")
                    except Exception:
                        pass
        else:
            from rich.progress import (
                Progress,
                SpinnerColumn,
                BarColumn,
                TextColumn,
                TimeElapsedColumn,
            )

            progress = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console,
            )
            with progress:
                task_id = progress.add_task(message, total=100)

                def _update(pct: float | None = None, stage: str | None = None) -> None:
                    kwargs = {}
                    if pct is not None:
                        kwargs["completed"] = pct
                    if stage:
                        kwargs["description"] = stage
                    progress.update(task_id, **kwargs)
                    if self.progress_handler and stage:
                        try:
                            self.progress_handler(stage)
                        except Exception:
                            pass

                try:
                    yield _update
                finally:
                    if self.progress_handler:
                        try:
                            self.progress_handler("done")
                        except Exception:
                            pass

    async def confirm(self, message: str) -> bool:
        """Ask the user to confirm an action."""
        result: bool | None = None
        if not self.plain:
            try:
                from textual.app import App, ComposeResult
                from textual.widgets import Label, Button
                from textual.containers import Horizontal, Vertical

                class ConfirmApp(App):
                    CSS = "Screen {align: center middle}"

                    def __init__(self, msg: str) -> None:
                        super().__init__()
                        self.msg = msg
                        self.result: bool | None = None

                    def compose(self) -> ComposeResult:
                        yield Vertical(
                            Label(self.msg),
                            Horizontal(Button("Sim", id="yes"), Button("Não", id="no")),
                        )

                    async def on_button_pressed(self, event: Button.Pressed) -> None:
                        self.result = event.button.id == "yes"
                        await self.action_quit()

                app = ConfirmApp(message)
                await app.run_async()
                result = app.result
            except Exception:
                try:
                    from rich.prompt import Confirm

                    result = Confirm.ask(Text(message), default=False, console=self.console)
                except Exception:
                    pass

        if result is None:
            resp = await self.read_command(f"{message} [s/N] ")
            result = resp.strip().lower() in {"s", "sim", "y", "yes"}

        self.remember_expires = None
        if result:
            r = await self.read_command("Lembrar esta decisão? [s/N] ")
            self.remember_choice = r.strip().lower() in {"s", "sim", "y", "yes"}
            if self.remember_choice:
                d = await self.read_command("Lembrar por quantos dias? ")
                try:
                    days = int(d)
                    if days > 0:
                        self.remember_expires = (
                            datetime.now() + timedelta(days=days)
                        ).isoformat()
                except Exception:
                    self.remember_expires = None
        else:
            self.remember_choice = False
            self.remember_expires = None
        return result
