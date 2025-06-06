import os
import yaml
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
import subprocess
import asyncio
from pathlib import Path

import networkx as nx
from asteval import Interpreter

from .config import logger
from .analyzer import CodeAnalyzer
from .memory import MemoryManager
from .ai_model import AIModel
from .plugin_manager import PluginManager
from .notifier import Notifier


class TaskManager:
    def __init__(self, task_file: str, code_analyzer: CodeAnalyzer, memory: MemoryManager, ai_model: Optional[AIModel] = None):
        self.tasks = self._load_tasks(task_file)
        self.code_analyzer = code_analyzer
        self.memory = memory
        self.ai_model = ai_model
        self.aeval = Interpreter()
        self.history: List[Dict[str, Any]] = []
        self._setup_default_tasks()
        self.plugins = PluginManager(self)
        self.notifier = Notifier()

    def get_history(self) -> List[Dict[str, Any]]:
        """Return list of executed tasks."""
        return list(self.history)

    def _load_tasks(self, task_file: str) -> Dict:
        if os.path.exists(task_file):
            with open(task_file, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _setup_default_tasks(self):
        if "impact_analysis" not in self.tasks:
            self.tasks["impact_analysis"] = {
                "name": "Análise de Impacto",
                "type": "analysis",
                "scope": "dependents",
                "condition": "True",
                "description": "Analisa o impacto de mudanças em funções",
            }
        if "code_review" not in self.tasks:
            self.tasks["code_review"] = {
                "name": "Revisão de Código",
                "type": "verification",
                "scope": "all",
                "condition": "'🚩' in findings or '⚠️' in findings",
                "description": "Revisão geral de qualidade de código",
            }
        if "lint" not in self.tasks:
            self.tasks["lint"] = {
                "name": "Lint",
                "type": "lint",
                "description": "Verifica TODOs no código",
            }
        if "pylint" not in self.tasks:
            self.tasks["pylint"] = {
                "name": "Pylint",
                "type": "pylint",
                "description": "Roda pylint para checar estilo",
            }
        if "type_check" not in self.tasks:
            self.tasks["type_check"] = {
                "name": "Type Check",
                "type": "type_check",
                "description": "Executa mypy para verificação de tipos",
            }
        if "run_tests" not in self.tasks:
            self.tasks["run_tests"] = {
                "name": "Testes Automatizados",
                "type": "test",
                "description": "Executa a suíte de testes com pytest",
            }
        if "static_analysis" not in self.tasks:
            self.tasks["static_analysis"] = {
                "name": "Análise Estática",
                "type": "static_analysis",
                "description": "Roda flake8 para encontrar problemas",
            }
        if "security_analysis" not in self.tasks:
            self.tasks["security_analysis"] = {
                "name": "Análise de Segurança",
                "type": "security_analysis",
                "description": "Roda bandit para detectar vulnerabilidades",
            }
        if "coverage" not in self.tasks:
            self.tasks["coverage"] = {
                "name": "Cobertura de Testes",
                "type": "coverage",
                "description": "Gera relatório de cobertura com coverage.py",
            }
        if "auto_refactor" not in self.tasks:
            self.tasks["auto_refactor"] = {
                "name": "Refatoração Automática",
                "type": "auto_refactor",
                "description": "Usa IA para refatorar um arquivo e valida com testes",
            }

    async def run_task(self, task_name: str, *args) -> Any:
        if task_name not in self.tasks:
            logger.error("Tarefa não encontrada", task=task_name)
            return {"error": f"Tarefa '{task_name}' não encontrada"}
        task = self.tasks[task_name]
        logger.info("Executando tarefa", task=task_name)
        if task["type"] == "analysis":
            result = await self._perform_analysis_task(task, *args)
        elif task["type"] == "verification":
            result = await self._perform_verification_task(task, *args)
        elif task["type"] == "learning":
            result = await self._perform_learning_task(task, *args)
        elif task["type"] == "lint":
            result = await self._perform_lint_task(task, *args)
        elif task["type"] == "test":
            result = await self._perform_test_task(task, *args)
        elif task["type"] == "static_analysis":
            result = await self._perform_static_analysis_task(task, *args)
        elif task["type"] == "security_analysis":
            result = await self._perform_security_analysis_task(task, *args)
        elif task["type"] == "pylint":
            result = await self._perform_pylint_task(task, *args)
        elif task["type"] == "type_check":
            result = await self._perform_type_check_task(task, *args)
        elif task["type"] == "auto_refactor":
            result = await self._perform_auto_refactor_task(task, *args)
        else:
            handler = getattr(self, f"_perform_{task['type']}_task", None)
            if handler:
                result = await handler(task, *args)
            else:
                logger.error("Tipo de tarefa inválido", task_type=task["type"])
                result = {"error": f"Tipo de tarefa '{task['type']}' não suportado"}
        self.memory.save(
            {
                "type": "task_result",
                "content": f"Resultado da tarefa {task_name}",
                "metadata": {
                    "task": task_name,
                    "args": args,
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                },
                "tags": ["task", task_name.split("_")[0]],
                "context_level": "short",
            }
        )
        self.history.append({
            "task": task_name,
            "args": args,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })
        if hasattr(self.notifier, 'send'):
            self.notifier.send(
                f"Tarefa {task_name} concluída",
                f"Resultado: {str(result)[:200]}"
            )
        return result

    async def _perform_analysis_task(self, task: Dict, *args) -> List[Dict]:
        target = args[0] if args else None
        findings = []
        if task["scope"] == "all":
            chunks = list(self.code_analyzer.code_chunks.values())
        elif task["scope"] == "dependents":
            chunks = []
            for name in args:
                if name in self.code_analyzer.code_graph.nodes:
                    for dep in nx.descendants(self.code_analyzer.code_graph, name):
                        if dep in self.code_analyzer.code_chunks:
                            chunks.append(self.code_analyzer.code_chunks[dep])
        else:
            chunks = [self.code_analyzer.code_chunks[target]] if target in self.code_analyzer.code_chunks else []
        for chunk in chunks:
            try:
                analysis = {
                    "chunk": chunk["name"],
                    "file": chunk["file"],
                    "issues": self._check_dependencies(chunk["name"]),
                    "rule_findings": self._apply_learned_rules(chunk),
                    "custom_findings": self._evaluate_custom_condition(task["condition"], chunk, args),
                }
                if self.ai_model:
                    try:
                        from .prompt_utils import build_analysis_prompt
                        prompt = build_analysis_prompt(
                            chunk["code"], analysis["issues"] + analysis["rule_findings"]
                        )
                        suggestion = await self.ai_model.generate(prompt, max_length=200)
                        analysis["ai_suggestion"] = suggestion.strip()
                    except Exception as e:
                        logger.error("Erro ao gerar sugestão da IA", chunk=chunk["name"], error=str(e))
                findings.append(analysis)
            except Exception as e:
                logger.error("Erro na análise", chunk=chunk["name"], error=str(e))
                findings.append({"chunk": chunk["name"], "error": str(e)})
        return findings

    def _evaluate_custom_condition(self, condition: str, chunk: Dict, args: Tuple) -> List[str]:
        findings = []
        try:
            if self.aeval(condition, chunk=chunk, args=args, graph=self.code_analyzer.code_graph):
                findings.append("✅ Condição personalizada atendida")
            else:
                findings.append("❌ Condição personalizada não atendida")
        except Exception as e:
            findings.append(f"⚠️ Erro ao avaliar condição: {str(e)}")
        return findings

    async def _perform_verification_task(self, task: Dict, *args) -> List[str]:
        findings = []
        if task["scope"] == "all":
            chunks = list(self.code_analyzer.code_chunks.values())
        else:
            target = args[0] if args else None
            chunks = [self.code_analyzer.code_chunks[target]] if target in self.code_analyzer.code_chunks else []
        for chunk in chunks:
            issues = self._check_dependencies(chunk["name"])
            rule_findings = self._apply_learned_rules(chunk)
            if issues or rule_findings:
                findings.append(f"Verificação {task['name']} em {chunk['name']}:")
                findings.extend(issues)
                findings.extend(rule_findings)
        return findings if findings else ["✅ Nenhum problema encontrado"]

    async def _perform_learning_task(self, task: Dict, *args) -> Dict:
        if task["operation"] == "add_rule":
            rule_name = args[0]
            condition = args[1]
            self.code_analyzer.learned_rules[rule_name] = condition
            self.memory.save(
                {
                    "type": "learned_rule",
                    "content": f"Nova regra aprendida: {rule_name}",
                    "metadata": {
                        "rule": rule_name,
                        "condition": condition,
                        "learned_at": datetime.now().isoformat(),
                    },
                    "tags": ["learning", "rule"],
                }
            )
            return {"status": "success", "rule_added": rule_name}
        return {"status": "error", "message": "Operação de aprendizado não reconhecida"}

    async def _perform_lint_task(self, task: Dict, *args) -> List[Dict]:
        from .lint import Linter
        linter = Linter(self.code_analyzer.code_root)
        results = linter.lint_all()
        findings = []
        for file, issues in results.items():
            findings.append({"file": file, "issues": issues})
        return findings if findings else ["✅ Nenhum TODO encontrado"]

    async def _perform_test_task(self, task: Dict, *args) -> List[str]:
        cmd = ["pytest", "-q"]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await proc.communicate()
            output = out.decode().splitlines()
            logger.info("Testes executados", returncode=proc.returncode)
            return output
        except FileNotFoundError:
            logger.error("pytest não encontrado")
            return ["pytest não disponível"]
        except Exception as e:
            logger.error("Erro ao executar testes", error=str(e))
            return [f"Erro ao executar testes: {e}"]

    async def _perform_static_analysis_task(self, task: Dict, *args) -> List[str]:
        cmd = ["flake8", str(self.code_analyzer.code_root)]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await proc.communicate()
            output = out.decode().splitlines()
            logger.info("Análise estática executada", returncode=proc.returncode)
            return output or ["✅ Nenhum problema encontrado"]
        except FileNotFoundError:
            logger.error("flake8 não encontrado")
            return ["flake8 não disponível"]
        except Exception as e:
            logger.error("Erro na análise estática", error=str(e))
            return [f"Erro na análise estática: {e}"]

    async def _perform_security_analysis_task(self, task: Dict, *args) -> List[str]:
        cmd = ["bandit", "-r", str(self.code_analyzer.code_root)]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await proc.communicate()
            output = out.decode().splitlines()
            logger.info("Análise de segurança executada", returncode=proc.returncode)
            return output or ["✅ Nenhuma vulnerabilidade encontrada"]
        except FileNotFoundError:
            logger.error("bandit não encontrado")
            return ["bandit não disponível"]
        except Exception as e:
            logger.error("Erro na análise de segurança", error=str(e))
            return [f"Erro na análise de segurança: {e}"]

    async def _perform_pylint_task(self, task: Dict, *args) -> List[str]:
        cmd = ["pylint", str(self.code_analyzer.code_root)]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await proc.communicate()
            output = out.decode().splitlines()
            logger.info("Pylint executado", returncode=proc.returncode)
            return output or ["✅ Nenhum problema encontrado"]
        except FileNotFoundError:
            logger.error("pylint não encontrado")
            return ["pylint não disponível"]
        except Exception as e:
            logger.error("Erro no pylint", error=str(e))
            return [f"Erro no pylint: {e}"]

    async def _perform_type_check_task(self, task: Dict, *args) -> List[str]:
        cmd = ["mypy", str(self.code_analyzer.code_root)]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await proc.communicate()
            output = out.decode().splitlines()
            logger.info("Type check executado", returncode=proc.returncode)
            return output or ["✅ Tipagem ok"]
        except FileNotFoundError:
            logger.error("mypy não encontrado")
            return ["mypy não disponível"]
        except Exception as e:
            logger.error("Erro na verificação de tipos", error=str(e))
            return [f"Erro na verificação de tipos: {e}"]

    async def _perform_coverage_task(self, task: Dict, *args) -> List[str]:
        cmd = ["coverage", "run", "-m", "pytest", "-q"]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await proc.communicate()
            report_proc = await asyncio.create_subprocess_exec(
                "coverage",
                "report",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            rep_out, _ = await report_proc.communicate()
            output = out.decode().splitlines() + rep_out.decode().splitlines()
            logger.info("Cobertura executada", returncode=proc.returncode)
            return output
        except FileNotFoundError:
            logger.error("coverage.py não encontrado")
            return ["coverage não disponível"]
        except Exception as e:
            logger.error("Erro na cobertura", error=str(e))
            return [f"Erro na cobertura: {e}"]

    async def _perform_auto_refactor_task(self, task: Dict, *args) -> Dict:
        if not self.ai_model:
            return {"error": "Modelo de IA não configurado"}
        file_path = args[0] if args else None
        if not file_path or not os.path.exists(file_path):
            return {"error": "Arquivo não encontrado"}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                original = f.read()
        except Exception as e:
            logger.error("Erro ao ler arquivo", file=file_path, error=str(e))
            return {"error": str(e)}

        from .prompt_utils import build_refactor_prompt
        prompt = build_refactor_prompt(original)
        try:
            suggestion = await self.ai_model.generate(prompt, max_length=len(original) + 200)
        except Exception as e:
            logger.error("Erro ao gerar refatoração", error=str(e))
            return {"error": str(e)}

        from .update_manager import UpdateManager
        updater = UpdateManager()

        def apply(p: Path) -> None:
            p.write_text(suggestion)

        success = updater.safe_apply(file_path, apply)
        return {"success": success, "new_code": suggestion[:200]}

    def _check_dependencies(self, chunk_name: str) -> List[str]:
        issues = []
        if chunk_name not in self.code_analyzer.code_graph:
            return issues
        for dep in self.code_analyzer.code_graph.successors(chunk_name):
            if dep not in self.code_analyzer.code_chunks:
                issues.append(f"⚠️ {chunk_name} depende de {dep} que não foi encontrado")
            else:
                chunk = self.code_analyzer.code_chunks[dep]
                if "last_modified" in chunk:
                    if (datetime.now() - datetime.fromisoformat(chunk["last_modified"])).days < 7:
                        issues.append(f"🚩 {chunk_name} depende de {dep} que foi modificado recentemente")
        return issues

    def _apply_learned_rules(self, chunk: Dict) -> List[str]:
        findings = []
        for rule, condition in self.code_analyzer.learned_rules.items():
            try:
                if self.aeval(condition, chunk=chunk, graph=self.code_analyzer.code_graph):
                    findings.append(f"🧠 {rule}")
            except Exception as e:
                logger.error("Erro ao aplicar regra", rule=rule, error=str(e))
                findings.append(f"⚠️ Erro ao aplicar regra {rule}: {str(e)}")
        return findings
