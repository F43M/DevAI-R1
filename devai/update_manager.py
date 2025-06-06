from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable, List

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

    def run_tests(self) -> bool:
        logger.info("Executando testes para validacao")
        proc = subprocess.run(
            self.tests_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        logger.info("Testes finalizados", returncode=proc.returncode)
        return proc.returncode == 0

    def safe_apply(
        self, file_path: str | Path, apply_func: Callable[[Path], None], max_attempts: int = 1
    ) -> bool:
        """Apply modifications with automatic rollback if tests fail."""
        path = Path(file_path)
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            backup = self._backup(path)
            try:
                apply_func(path)
                if self.run_tests():
                    backup.unlink(missing_ok=True)
                    logger.info(
                        "Atualizacao aplicada com sucesso", file=str(path)
                    )
                    return True
                self._restore(backup, path)
            except Exception as e:  # pragma: no cover - unexpected errors
                logger.error("Erro na atualizacao", file=str(path), error=str(e))
                self._restore(backup, path)
                raise
        logger.error(
            "Falha em validar atualizacao apos tentativas", file=str(path)
        )
        return False
