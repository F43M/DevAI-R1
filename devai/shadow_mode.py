import difflib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from typing import Tuple
from threading import Thread

from .ai_model import AIModel
from .config import config, logger


SHADOW_BASE = Path("/tmp/devai_shadow")


def simulate_update(file_path: str, suggested_code: str) -> Tuple[str, str, str]:
    """Create a temporary project copy with the suggested change applied."""
    original_path = Path(file_path)
    original_lines = original_path.read_text().splitlines(keepends=True)
    new_lines = suggested_code.splitlines(keepends=True)
    diff = "".join(
        difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile=str(original_path),
            tofile="suggested",
        )
    )
    sim_id = uuid4().hex
    SHADOW_BASE.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f"shadow_{sim_id}_", dir=SHADOW_BASE))
    project_root = Path(config.CODE_ROOT).resolve()
    shutil.copytree(project_root, temp_root / project_root.name, dirs_exist_ok=True)
    temp_file = temp_root / project_root.name / original_path.relative_to(project_root)
    assert not os.path.samefile(original_path, temp_file)
    temp_file.write_text(suggested_code)
    return diff, str(temp_root), sim_id


async def evaluate_change_with_ia(diff_text: str) -> dict:
    """Ask the external AI to evaluate the diff."""
    prompt = (
        "Avalie o seguinte diff e diga:\n"
        "- Essa mudança é segura?\n"
        "- Há riscos estruturais?\n"
        "- Que impacto ela pode ter?\n"
        "- Você recomendaria aplicar?\n\n"
        f"{diff_text}"
    )
    ai = AIModel()
    try:
        analysis = await ai.safe_api_call(prompt, 800)
    finally:
        await ai.close()
    return {"analysis": analysis, "confidence": "Desconhecida"}


def log_simulation(
    sim_id: str, file_path: str, tests_passed: bool, evaluation: str, action: str
) -> None:
    """Append an entry to the simulation history log."""
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(exist_ok=True)
    log = log_dir / "simulation_history.md"
    with log.open("a", encoding="utf-8") as f:
        f.write(f"## Simulação {sim_id}\n")
        f.write(f"Arquivo: {file_path}\n")
        f.write(f"Testes passaram: {tests_passed}\n")
        f.write(f"Ação: {action}\n")
        f.write("### Avaliação IA\n")
        f.write(evaluation + "\n\n")
    logger.info(
        "Simulação registrada", file=file_path, action=action, id=sim_id, tests=tests_passed
    )


def run_test_isolated(path: str | Path, timeout: int = 30) -> Tuple[bool, str]:
    """Run pytest in a subprocess with a timeout to avoid freezes."""
    cwd = Path(path)
    try:
        proc = subprocess.run(
            ["pytest", "-q"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            timeout=timeout,
        )
        return proc.returncode == 0, proc.stdout.decode()
    except subprocess.TimeoutExpired:
        return False, f"\U0001F6D1 Tempo excedido: os testes demoraram mais de {timeout}s e foram cancelados."
    except Exception as e:  # pragma: no cover - unexpected errors
        return False, f"\u26A0\uFE0F Erro ao executar testes: {e}"


def run_tests_in_temp(temp_dir: str, timeout: int = 30) -> Tuple[bool, str]:
    """Execute pytest in a temporary directory copy of the project using isolation."""
    project_subdir = Path(temp_dir) / Path(config.CODE_ROOT).name
    cwd = project_subdir if project_subdir.exists() else Path(temp_dir)

    result: Tuple[bool, str] | None = None

    def _target() -> None:
        nonlocal result
        result = run_test_isolated(cwd, timeout=timeout)

    t = Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout + 5)
    if result is None:
        return False, f"\U0001F6D1 Tempo excedido: os testes demoraram mais de {timeout}s e foram cancelados."
    return result


def run_tests_async(path: str, timeout: int = 30) -> None:
    """Run tests asynchronously in a daemon thread."""
    Thread(target=lambda: run_test_isolated(path, timeout), daemon=True).start()



