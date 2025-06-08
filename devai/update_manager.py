from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Iterable, List
import asyncio

from .ai_model import AIModel
from .learning_engine import registrar_licao_negativa
from .feedback import registrar_feedback_negativo
from .test_runner import run_pytest

from .config import logger


class UpdateManager:
    """Safely apply changes to a file and verify using tests."""

    def __init__(self, tests_cmd: Iterable[str] | None = None) -> None:
        self.tests_cmd: List[str] = list(tests_cmd or ["pytest", "-q"])

    def _backup(self, path: Path) -> Path:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        logger.info("Backup criado", file=str(path), backup=str(backup))
        return backup

    def _restore(self, backup: Path, original: Path) -> None:
        shutil.move(str(backup), str(original))
        logger.warning(
            "Arquivo restaurado devido a falha nos testes", file=str(original)
        )

    def run_tests(self, capture_output: bool = False):
        logger.info("Executando testes para validacao")
        ok, output = run_pytest(Path.cwd())
        logger.info("Testes finalizados", success=ok)
        if capture_output:
            return ok, output
        return ok

    def safe_apply(
        self,
        file_path: str | Path,
        apply_func: Callable[[Path], None],
        max_attempts: int = 1,
        capture_output: bool = False,
    ):
        """Apply modifications with automatic rollback if tests fail."""
        path = Path(file_path)
        original_lines = path.read_text().splitlines()
        protected: List[tuple[int, int]] = []
        start = None
        for i, line in enumerate(original_lines):
            if "<protect>" in line:
                start = i
            elif "</protect>" in line and start is not None:
                protected.append((start, i))
                start = None
        attempt = 0
        result_output = ""
        while attempt < max_attempts:
            attempt += 1
            backup = self._backup(path)
            try:
                apply_func(path)
                new_lines = path.read_text().splitlines()
                for s, e in protected:
                    if original_lines[s : e + 1] != new_lines[s : e + 1]:
                        logger.error("Tentativa de modificar area protegida", file=str(path))
                        raise RuntimeError("protected block modified")
                try:
                    success, out = self.run_tests(capture_output=True)
                except TypeError:
                    success = self.run_tests()
                    out = ""
                if success:
                    backup.unlink(missing_ok=True)
                    logger.info(
                        "Atualizacao aplicada com sucesso", file=str(path)
                    )
                    return (True, out) if capture_output else True

                registrar_licao_negativa(str(path), out)
                self._restore(backup, path)
                backup = self._backup(path)

                async def _retry() -> tuple[bool, str]:
                    ai = AIModel()
                    try:
                        prompt = (
                            "A refatoração anterior falhou nos testes com o seguinte erro:\n"
                            f"{out}\nProponha uma nova solução corrigida, mantendo o objetivo anterior."
                        )
                        suggestion = await ai.safe_api_call(
                            prompt,
                            len(path.read_text()) + 200,
                            prompt,
                        )
                    finally:
                        await ai.close()
                    path.write_text(suggestion)
                    return self.run_tests(capture_output=True)

                success2, out2 = asyncio.run(_retry())
                if success2:
                    backup.unlink(missing_ok=True)
                    logger.info(
                        "Atualizacao aplicada com sucesso", file=str(path)
                    )
                    return (True, out2) if capture_output else True

                registrar_feedback_negativo(str(path), out2)
                self._restore(backup, path)
                if capture_output:
                    result_output = out2
            except Exception as e:  # pragma: no cover - unexpected errors
                logger.error("Erro na atualizacao", file=str(path), error=str(e))
                self._restore(backup, path)
                raise
        logger.error(
            "Falha em validar atualizacao apos tentativas", file=str(path)
        )
        return (False, result_output) if capture_output else False
