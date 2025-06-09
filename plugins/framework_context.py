from pathlib import Path
import json
import xml.etree.ElementTree as ET
from typing import List, Dict


PLUGIN_INFO = {
    "name": "Framework Context",
    "version": "1.0",
    "description": "Extrai contexto de frameworks e dependências",
}


def register(task_manager):
    async def _perform_framework_context_task(self, task, *args):
        root = Path(self.code_analyzer.code_root).parent
        entries: List[Dict[str, List[str]]] = []
        files = [root / "package.json", root / "requirements.txt", root / "pom.xml"]
        for f in files:
            if not f.exists():
                continue
            try:
                if f.name == "package.json":
                    data = json.loads(f.read_text())
                    deps = list(data.get("dependencies", {}).keys())
                elif f.suffix == ".xml":
                    tree = ET.parse(f)
                    deps = [d.text for d in tree.findall(".//dependency/artifactId")]
                else:
                    deps = [line.strip().split("==")[0] for line in f.read_text().splitlines() if line.strip() and not line.startswith("#")]
                entries.append({f.name: deps})
                self.memory.save(
                    {
                        "type": "framework_info",
                        "content": f"Dependências em {f.name}",
                        "metadata": {"file": f.name, "deps": deps},
                        "tags": ["framework", "dependency"],
                    }
                )
            except Exception:
                continue
        return entries or ["Nenhuma configuração detectada"]

    task_manager.tasks["framework_context"] = {
        "name": "Extrair Contexto de Frameworks",
        "type": "framework_context",
        "description": "Lê arquivos de config de frameworks para a memória",
    }
    setattr(task_manager, "_perform_framework_context_task", _perform_framework_context_task.__get__(task_manager))


def unregister(task_manager):
    task_manager.tasks.pop("framework_context", None)
    if hasattr(task_manager, "_perform_framework_context_task"):
        delattr(task_manager, "_perform_framework_context_task")
