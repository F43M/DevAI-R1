import difflib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import hashlib
from uuid import uuid4
from datetime import datetime
from typing import Tuple, Callable, Optional
from threading import Thread

from .ai_model import AIModel
from .config import config, logger
from .test_runner import run_pytest


SHADOW_BASE = Path("/tmp/devai_shadow")


def simulate_update(
    file_path: str,
    suggested_code: str,
    cleanup_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[str, bool, str, str, str]:
    """Create a temporary project copy, run tests and cleanup."""
    original_path = Path(file_path).resolve()
    original_lines = original_path.read_text().splitlines(keepends=True)
    new_lines = suggested_code.splitlines(keepends=True)
    project_root = Path(config.CODE_ROOT).resolve()
    rel = str(original_path.relative_to(project_root))
    diff = "".join(
        difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile=rel,
            tofile=rel,
        )
    )

    patch_file = tempfile.NamedTemporaryFile(delete=False, prefix="change_", suffix=".patch")
    patch_file.write(diff.encode("utf-8"))
    patch_file.close()

    proc = subprocess.run(
        ["git", "apply", "--check", patch_file.name],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        Path(patch_file.name).unlink(missing_ok=True)
        raise RuntimeError(f"Patch conflict: {proc.stderr.strip()}")

    sim_id = uuid4().hex
    SHADOW_BASE.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f"shadow_{sim_id}_", dir=SHADOW_BASE))
    shutil.copytree(project_root, temp_root / project_root.name, dirs_exist_ok=True)
    temp_file = temp_root / project_root.name / rel
    assert not os.path.samefile(original_path, temp_file)
    temp_file.write_text(suggested_code)

    tests_ok, test_output = run_tests_in_temp(temp_root)

    if cleanup_cb:
        cleanup_cb(str(temp_root))
    else:
        shutil.rmtree(temp_root, ignore_errors=True)

    return diff, tests_ok, test_output, sim_id, patch_file.name


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
    sim_id: str,
    file_path: str,
    tests_passed: bool,
    evaluation: str,
    action: str,
    patch_hash: str = "",
    test_output: str = "",
) -> None:
    """Append an entry to the simulation history log."""
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(exist_ok=True)
    log = log_dir / "simulation_history.md"
    with log.open("a", encoding="utf-8") as f:
        f.write(f"## Simulação {sim_id}\n")
        f.write(f"Arquivo: {file_path}\n")
        if patch_hash:
            f.write(f"Patch: {patch_hash}\n")
        f.write(f"Testes passaram: {tests_passed}\n")
        if test_output:
            f.write("Resultado dos testes:\n")
            f.write(f"```\n{test_output}\n```\n")
        f.write(f"Ação: {action}\n")
        f.write("### Avaliação IA\n")
        f.write(evaluation + "\n\n")
    logger.info(
        "Simulação registrada",
        file=file_path,
        action=action,
        id=sim_id,
        tests=tests_passed,
    )


def run_test_isolated(path: str | Path, timeout: int = 30) -> Tuple[bool, str]:
    """Run pytest in an isolated process."""
    return run_pytest(path, timeout)


def run_tests_in_temp(temp_dir: str | Path, timeout: int = 30) -> Tuple[bool, str]:
    """Execute pytest in a temporary directory copy of the project using isolation."""
    temp_path = Path(temp_dir)
    if not temp_path.is_dir():
        raise ValueError("Diretório temporário inválido para execução de testes.")
    project_subdir = temp_path / Path(config.CODE_ROOT).name
    cwd = project_subdir if project_subdir.exists() else temp_path

    result: Tuple[bool, str] | None = None

    def _target() -> None:
        nonlocal result
        result = run_test_isolated(cwd, timeout=timeout)

    t = Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout + 5)
    if result is None:
        return (
            False,
            f"\U0001f6d1 Tempo excedido: os testes demoraram mais de {timeout}s e foram cancelados.",
        )
    return result


def run_tests_async(path: str, timeout: int = 30) -> None:
    """Run tests asynchronously in a daemon thread."""
    Thread(target=lambda: run_test_isolated(path, timeout), daemon=True).start()
