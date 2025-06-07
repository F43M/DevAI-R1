import difflib
import shutil
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from typing import Tuple

from .ai_model import AIModel
from .config import config, logger


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
    temp_root = Path(tempfile.mkdtemp(prefix=f"shadow_{sim_id}_"))
    project_root = Path(config.CODE_ROOT).resolve()
    shutil.copytree(project_root, temp_root / project_root.name, dirs_exist_ok=True)
    temp_file = temp_root / project_root.name / original_path.relative_to(project_root)
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


def log_simulation(file_path: str, evaluation: str, action: str) -> None:
    """Append an entry to the simulation history log."""
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(exist_ok=True)
    log = log_dir / "simulation_history.md"
    with log.open("a", encoding="utf-8") as f:
        f.write(
            f"[{datetime.now().isoformat()}] Simulação de alteração em {file_path}\n"
        )
        f.write(f"Resultado IA: {evaluation}\n")
        f.write(f"Ação: {action}\n\n")
    logger.info("Simulação registrada", file=file_path, action=action)


def run_tests_in_temp(temp_dir: str) -> Tuple[bool, str]:
    """Execute pytest in a temporary directory."""
    proc = subprocess.run(
        ["pytest", "-q"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=temp_dir
    )
    return proc.returncode == 0, proc.stdout.decode()

