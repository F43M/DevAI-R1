import os
import yaml  # type: ignore
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
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
from .test_runner import run_pytest
from .sandbox import run_in_sandbox
from .approval import requires_approval, request_approval
from .decision_log import log_decision
from .patch_utils import split_diff_by_file, apply_patch
import re

"""Task execution utilities used by DevAI.

This module defines :class:`TaskManager`, responsible for loading task
definitions and executing them. It also exposes several helper methods used
internally when running quality checks such as lint, tests and static
analysis.
"""


class TaskManager:
    """Manage task execution and persistence for DevAI."""

    def __init__(
        self,
        task_file: str,
        code_analyzer: CodeAnalyzer,
        memory: MemoryManager,
        ai_model: Optional[AIModel] = None,
    ):
        """Initialize the manager and load built-in tasks."""
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

    def last_actions(self, n: int = 3) -> List[Dict[str, Any]]:
        """Return last n executed actions."""
        return self.history[-n:]

    def _load_tasks(self, task_file: str) -> Dict:
        """Load task definitions from ``task_file`` if it exists."""
        if os.path.exists(task_file):
            with open(task_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _setup_default_tasks(self):
        """Ensure core tasks are present in ``self.tasks`` dictionary."""
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
        if "quality_suite" not in self.tasks:
            self.tasks["quality_suite"] = {
                "name": "Quality Suite",
                "type": "quality_suite",
                "description": "Roda lint, análise e testes em paralelo",
            }

    async def run_task(self, task_name: str, *args, progress=None, ui=None) -> Any:
        """Execute a named task with optional UI confirmations."""
        if task_name not in self.tasks:
            logger.error("Tarefa não encontrada", task=task_name)
            return {"error": f"Tarefa '{task_name}' não encontrada"}
        task = self.tasks[task_name]
        logger.info("Executando tarefa", task=task_name)
        action_map = {
            "test": "shell",
            "static_analysis": "shell_safe",
            "security_analysis": "shell_safe",
            "pylint": "shell_safe",
            "type_check": "shell_safe",
            "coverage": "shell",
            "auto_refactor": "edit",
        }
        action = action_map.get(task["type"])
        if action and requires_approval(action):
            if ui:
                approved = await ui.confirm(f"Executar tarefa {task_name}?")
                model = "cli"
            else:
                approved = await request_approval(f"Executar tarefa {task_name}?")
                model = "web"
            log_decision(
                action, task_name, "execucao", model, "ok" if approved else "nao"
            )
            if not approved:
                return {"canceled": True}

        if task["type"] == "analysis":
            result = await self._perform_analysis_task(task, *args)
        elif task["type"] == "verification":
            result = await self._perform_verification_task(task, *args)
        elif task["type"] == "learning":
            result = await self._perform_learning_task(task, *args)
        elif task["type"] == "lint":
            result = await self._perform_lint_task(task, *args)
        elif task["type"] == "test":
            try:
                result = await self._perform_test_task(
                    task, *args, progress_cb=progress, ui=ui
                )
            except TypeError:
                result = await self._perform_test_task(
                    task, *args, progress_cb=progress
                )
        elif task["type"] == "static_analysis":
            try:
                result = await self._perform_static_analysis_task(task, *args, ui=ui)
            except TypeError:
                result = await self._perform_static_analysis_task(task, *args)
        elif task["type"] == "security_analysis":
            try:
                result = await self._perform_security_analysis_task(task, *args, ui=ui)
            except TypeError:
                result = await self._perform_security_analysis_task(task, *args)
        elif task["type"] == "pylint":
            try:
                result = await self._perform_pylint_task(task, *args, ui=ui)
            except TypeError:
                result = await self._perform_pylint_task(task, *args)
        elif task["type"] == "type_check":
            try:
                result = await self._perform_type_check_task(task, *args, ui=ui)
            except TypeError:
                result = await self._perform_type_check_task(task, *args)
        elif task["type"] == "auto_refactor":
            try:
                result = await self._perform_auto_refactor_task(task, *args, ui=ui)
            except TypeError:
                result = await self._perform_auto_refactor_task(task, *args)
        elif task["type"] == "quality_suite":
            try:
                result = await self._perform_quality_suite_task(task, *args, ui=ui)
            except TypeError:
                result = await self._perform_quality_suite_task(task, *args)
        elif task["type"] == "coverage":
            try:
                result = await self._perform_coverage_task(
                    task, *args, progress_cb=progress, ui=ui
                )
            except TypeError:
                result = await self._perform_coverage_task(
                    task, *args, progress_cb=progress
                )
        else:
            handler = getattr(self, f"_perform_{task['type']}_task", None)
            if handler:
                try:
                    result = await handler(task, *args, ui=ui)
                except TypeError:
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
        self.history.append(
            {
                "task": task_name,
                "args": args,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }
        )
        if hasattr(self.notifier, "send"):
            self.notifier.send(
                f"Tarefa {task_name} concluída", f"Resultado: {str(result)[:200]}"
            )
        return result

    async def _perform_analysis_task(self, task: Dict, *args) -> List[Dict]:
        """Run a static impact analysis on the given code chunks."""
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
            chunks = (
                [self.code_analyzer.code_chunks[target]]
                if target in self.code_analyzer.code_chunks
                else []
            )
        for chunk in chunks:
            try:
                analysis = {
                    "chunk": chunk["name"],
                    "file": chunk["file"],
                    "issues": self._check_dependencies(chunk["name"]),
                    "rule_findings": self._apply_learned_rules(chunk),
                    "custom_findings": self._evaluate_custom_condition(
                        task["condition"], chunk, args
                    ),
                }
                if self.ai_model:
                    try:
                        from .prompt_utils import build_analysis_prompt

                        prompt = build_analysis_prompt(
                            chunk["code"],
                            analysis["issues"] + analysis["rule_findings"],
                        )
                        suggestion = await self.ai_model.safe_api_call(
                            prompt, 200, prompt, self.memory
                        )
                        analysis["ai_suggestion"] = suggestion.strip()
                    except Exception as e:
                        logger.error(
                            "Erro ao gerar sugestão da IA",
                            chunk=chunk["name"],
                            error=str(e),
                        )
                findings.append(analysis)
            except Exception as e:
                logger.error("Erro na análise", chunk=chunk["name"], error=str(e))
                findings.append({"chunk": chunk["name"], "error": str(e)})
        return findings

    def _evaluate_custom_condition(
        self, condition: str, chunk: Dict, args: Tuple
    ) -> List[str]:
        """Evaluate a user-defined condition using ``asteval``."""
        findings = []
        try:
            if self.aeval(
                condition, chunk=chunk, args=args, graph=self.code_analyzer.code_graph
            ):
                findings.append("✅ Condição personalizada atendida")
            else:
                findings.append("❌ Condição personalizada não atendida")
        except Exception as e:
            findings.append(f"⚠️ Erro ao avaliar condição: {str(e)}")
        return findings

    async def _perform_verification_task(self, task: Dict, *args) -> List[str]:
        """Check project code against learned rules and dependencies."""
        findings = []
        if task["scope"] == "all":
            chunks = list(self.code_analyzer.code_chunks.values())
        else:
            target = args[0] if args else None
            chunks = (
                [self.code_analyzer.code_chunks[target]]
                if target in self.code_analyzer.code_chunks
                else []
            )
        for chunk in chunks:
            issues = self._check_dependencies(chunk["name"])
            rule_findings = self._apply_learned_rules(chunk)
            if issues or rule_findings:
                findings.append(f"Verificação {task['name']} em {chunk['name']}:")
                findings.extend(issues)
                findings.extend(rule_findings)
        return findings if findings else ["✅ Nenhum problema encontrado"]

    async def _perform_learning_task(self, task: Dict, *args) -> Dict:
        """Handle symbolic learning operations for custom rules."""
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
        """Run the basic project linter and return any TODO findings."""
        from .lint import Linter

        linter = Linter(self.code_analyzer.code_root)
        results = linter.lint_all()
        findings = []
        for file, issues in results.items():
            findings.append({"file": file, "issues": issues})
        return findings if findings else ["✅ Nenhum TODO encontrado"]

    async def _perform_test_task(
        self, task: Dict, *args, progress_cb=None, ui=None
    ) -> List[str]:
        """Execute the project's test suite using ``pytest``."""
        if requires_approval("shell"):
            if ui:
                approved = await ui.confirm("Executar testes?")
                model = "cli"
            else:
                approved = await request_approval(
                    "Executar testes?",
                    details="pytest -q",
                )
                model = "web"
            log_decision(
                "shell",
                task.get("type", "test"),
                "execucao",
                model,
                "ok" if approved else "nao",
            )
            if not approved:
                return ["cancelado"]
        if progress_cb:
            progress_cb(0, "running tests")
        ok, out = run_pytest(self.code_analyzer.code_root)
        if progress_cb:
            progress_cb(100, "tests done")
        logger.info("Testes executados", success=ok)
        return out.splitlines()

    async def _perform_static_analysis_task(
        self, task: Dict, *args, ui=None
    ) -> List[str]:
        """Run flake8 on the code base inside a sandbox."""
        if requires_approval("shell_safe"):
            if ui:
                approved = await ui.confirm("Executar análise estática?")
                model = "cli"
            else:
                approved = await request_approval(
                    "Executar análise estática?",
                    details="flake8 " + str(self.code_analyzer.code_root),
                )
                model = "web"
            log_decision(
                "shell_safe",
                task.get("type", "static"),
                "execucao",
                model,
                "ok" if approved else "nao",
            )
            if not approved:
                return ["cancelado"]
        cmd = ["flake8", str(self.code_analyzer.code_root)]
        try:
            out = await asyncio.to_thread(run_in_sandbox, cmd, 30)
            output = out.splitlines()
            logger.info("Análise estática executada")
            return output or ["✅ Nenhum problema encontrado"]
        except FileNotFoundError:
            logger.error("flake8 não encontrado")
            return ["flake8 não disponível"]
        except Exception as e:
            logger.error("Erro na análise estática", error=str(e))
            return [f"Erro na análise estática: {e}"]

    async def _perform_security_analysis_task(
        self, task: Dict, *args, ui=None
    ) -> List[str]:
        """Execute bandit to look for security issues."""
        if requires_approval("shell_safe"):
            if ui:
                approved = await ui.confirm("Executar análise de segurança?")
                model = "cli"
            else:
                approved = await request_approval(
                    "Executar análise de segurança?",
                    details="bandit -r " + str(self.code_analyzer.code_root),
                )
                model = "web"
            log_decision(
                "shell_safe",
                task.get("type", "security"),
                "execucao",
                model,
                "ok" if approved else "nao",
            )
            if not approved:
                return ["cancelado"]
        cmd = ["bandit", "-r", str(self.code_analyzer.code_root)]
        try:
            out = await asyncio.to_thread(run_in_sandbox, cmd, 30)
            output = out.splitlines()
            logger.info("Análise de segurança executada")
            return output or ["✅ Nenhuma vulnerabilidade encontrada"]
        except FileNotFoundError:
            logger.error("bandit não encontrado")
            return ["bandit não disponível"]
        except Exception as e:
            logger.error("Erro na análise de segurança", error=str(e))
            return [f"Erro na análise de segurança: {e}"]

    async def _perform_pylint_task(self, task: Dict, *args, ui=None) -> List[str]:
        """Run pylint over the project in a sandbox environment."""
        if requires_approval("shell_safe"):
            if ui:
                approved = await ui.confirm("Executar pylint?")
                model = "cli"
            else:
                approved = await request_approval(
                    "Executar pylint?",
                    details="pylint " + str(self.code_analyzer.code_root),
                )
                model = "web"
            log_decision(
                "shell_safe",
                task.get("type", "pylint"),
                "execucao",
                model,
                "ok" if approved else "nao",
            )
            if not approved:
                return ["cancelado"]
        cmd = ["pylint", str(self.code_analyzer.code_root)]
        try:
            out = await asyncio.to_thread(run_in_sandbox, cmd, 30)
            output = out.splitlines()
            logger.info("Pylint executado")
            return output or ["✅ Nenhum problema encontrado"]
        except FileNotFoundError:
            logger.error("pylint não encontrado")
            return ["pylint não disponível"]
        except Exception as e:
            logger.error("Erro no pylint", error=str(e))
            return [f"Erro no pylint: {e}"]

    async def _perform_type_check_task(self, task: Dict, *args, ui=None) -> List[str]:
        """Execute ``mypy`` to ensure static type correctness."""
        if requires_approval("shell_safe"):
            if ui:
                approved = await ui.confirm("Executar verificação de tipos?")
                model = "cli"
            else:
                approved = await request_approval(
                    "Executar verificação de tipos?",
                    details="mypy " + str(self.code_analyzer.code_root),
                )
                model = "web"
            log_decision(
                "shell_safe",
                task.get("type", "type_check"),
                "execucao",
                model,
                "ok" if approved else "nao",
            )
            if not approved:
                return ["cancelado"]
        cmd = ["mypy", str(self.code_analyzer.code_root)]
        try:
            out = await asyncio.to_thread(run_in_sandbox, cmd, 30)
            output = out.splitlines()
            logger.info("Type check executado")
            return output or ["✅ Tipagem ok"]
        except FileNotFoundError:
            logger.error("mypy não encontrado")
            return ["mypy não disponível"]
        except Exception as e:
            logger.error("Erro na verificação de tipos", error=str(e))
            return [f"Erro na verificação de tipos: {e}"]

    async def _perform_coverage_task(
        self, task: Dict, *args, progress_cb=None, ui=None
    ) -> List[str]:
        """Run ``coverage.py`` to measure test coverage."""
        if requires_approval("shell"):
            if ui:
                approved = await ui.confirm("Executar cobertura de testes?")
                model = "cli"
            else:
                approved = await request_approval(
                    "Executar cobertura de testes?",
                    details="coverage run -m pytest -q",
                )
                model = "web"
            log_decision(
                "shell",
                task.get("type", "coverage"),
                "execucao",
                model,
                "ok" if approved else "nao",
            )
            if not approved:
                return ["cancelado"]
        cmd = ["coverage", "run", "-m", "pytest", "-q"]
        try:
            if progress_cb:
                progress_cb(0, "running tests")
            out = await asyncio.to_thread(run_in_sandbox, cmd, 30)
            if progress_cb:
                progress_cb(70, "analyzing coverage")
            rep = await asyncio.to_thread(run_in_sandbox, ["coverage", "report"], 30)
            output = out.splitlines() + rep.splitlines()
            if progress_cb:
                progress_cb(100, "done")
            logger.info("Cobertura executada")
            return output
        except FileNotFoundError:
            logger.error("coverage.py não encontrado")
            return ["coverage não disponível"]
        except Exception as e:
            logger.error("Erro na cobertura", error=str(e))
            return [f"Erro na cobertura: {e}"]

    async def _perform_auto_refactor_task(self, task: Dict, *args, ui=None) -> Dict:
        """Use the AI model to suggest and apply a code refactor."""
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

        prompt = build_refactor_prompt(original, file_path)
        try:
            suggestion = await self.ai_model.safe_api_call(
                prompt,
                len(original) + 200,
                prompt,
                self.memory,
            )
        except Exception as e:
            logger.error("Erro ao gerar refatoração", error=str(e))
            return {"error": str(e)}

        from .update_manager import UpdateManager

        updater = UpdateManager()

        diff_match = re.search(r"```diff\n(.*?)```", suggestion, re.DOTALL)
        diff_text = diff_match.group(1).strip() if diff_match else suggestion.strip()

        patches = split_diff_by_file(diff_text)
        patch = (
            patches.get(file_path)
            or patches.get(os.path.relpath(file_path))
            or (next(iter(patches.values())) if len(patches) == 1 else None)
        )
        if not patch:
            logger.error("Patch inválido", file=file_path)
            return {"error": "Patch inválido"}

        def apply(p: Path, d=patch) -> None:
            apply_patch(d)

        if requires_approval("edit"):
            if ui:
                approved = await ui.confirm(f"Aplicar refatoração em {file_path}?")
                model = "cli"
            else:
                approved = await request_approval(
                    f"Aplicar refatoração em {file_path}?",
                    details=diff_text,
                )
                model = "web"
            log_decision(
                "edit",
                task.get("type", "auto_refactor"),
                "execucao",
                model,
                "ok" if approved else "nao",
            )
            if not approved:
                return {"canceled": True}

        try:
            success, out = updater.safe_apply(file_path, apply, capture_output=True)
        except TypeError:
            success = updater.safe_apply(file_path, apply)
            out = ""
        if not success:
            from .prompt_engine import build_debug_prompt

            debug = build_debug_prompt(out, "", original[:200])
            try:
                suggestion = await self.ai_model.safe_api_call(
                    debug,
                    len(original) + 200,
                    debug,
                    self.memory,
                )
            except Exception as e:
                logger.error("Erro no fallback de refatoracao", error=str(e))
                return {"error": str(e)}

            diff_match_retry = re.search(r"```diff\n(.*?)```", suggestion, re.DOTALL)
            diff_text_retry = (
                diff_match_retry.group(1).strip()
                if diff_match_retry
                else suggestion.strip()
            )
            patches_retry = split_diff_by_file(diff_text_retry)
            patch_retry = (
                patches_retry.get(file_path)
                or patches_retry.get(os.path.relpath(file_path))
                or (
                    next(iter(patches_retry.values()))
                    if len(patches_retry) == 1
                    else None
                )
            )
            if not patch_retry:
                logger.error("Patch inválido", file=file_path)
                return {"error": "Patch inválido"}

            def apply_retry(p: Path, d=patch_retry) -> None:
                apply_patch(d)

            if requires_approval("edit"):
                if ui:
                    approved = await ui.confirm(
                        f"Aplicar refatoração em {file_path} (retry)?"
                    )
                    model = "cli"
                else:
                    approved = await request_approval(
                        f"Aplicar refatoração em {file_path} (retry)?",
                        details=diff_text_retry,
                    )
                    model = "web"
                log_decision(
                    "edit",
                    task.get("type", "auto_refactor"),
                    "retry",
                    model,
                    "ok" if approved else "nao",
                )
                if not approved:
                    return {"canceled": True}
            try:
                success, _ = updater.safe_apply(
                    file_path, apply_retry, capture_output=True
                )
            except TypeError:
                success = updater.safe_apply(file_path, apply_retry)
        return {"success": success, "patch": diff_text[:200]}

    async def _perform_quality_suite_task(
        self, task: Dict, *args, ui=None
    ) -> Dict[str, List[str]]:
        """Run lint, static analysis, security checks and tests in parallel."""
        lint_t = self._perform_lint_task(self.tasks["lint"])
        static_t = self._perform_static_analysis_task(
            self.tasks["static_analysis"], ui=ui
        )
        sec_t = self._perform_security_analysis_task(
            self.tasks["security_analysis"], ui=ui
        )
        tests_t = self._perform_test_task(self.tasks["run_tests"], ui=ui)
        results = await asyncio.gather(
            lint_t, static_t, sec_t, tests_t, return_exceptions=True
        )
        names = ["lint", "static", "security", "tests"]
        out: Dict[str, List[str]] = {}
        for name, res in zip(names, results):
            if isinstance(res, Exception):
                out[name] = [str(res)]
            else:
                out[name] = res
        return out

    def _check_dependencies(self, chunk_name: str) -> List[str]:
        """Return dependency issues for the specified code chunk."""
        issues = []
        if chunk_name not in self.code_analyzer.code_graph:
            return issues
        for dep in self.code_analyzer.code_graph.successors(chunk_name):
            if dep not in self.code_analyzer.code_chunks:
                issues.append(f"⚠️ {chunk_name} depende de {dep} que não foi encontrado")
            else:
                chunk = self.code_analyzer.code_chunks[dep]
                if "last_modified" in chunk:
                    if (
                        datetime.now() - datetime.fromisoformat(chunk["last_modified"])
                    ).days < 7:
                        issues.append(
                            f"🚩 {chunk_name} depende de {dep} que foi modificado recentemente"
                        )
        return issues

    def _apply_learned_rules(self, chunk: Dict) -> List[str]:
        """Apply learned symbolic rules to a code chunk."""
        findings = []
        for rule, condition in self.code_analyzer.learned_rules.items():
            try:
                if self.aeval(
                    condition, chunk=chunk, graph=self.code_analyzer.code_graph
                ):
                    findings.append(f"🧠 {rule}")
            except Exception as e:
                logger.error("Erro ao aplicar regra", rule=rule, error=str(e))
                findings.append(f"⚠️ Erro ao aplicar regra {rule}: {str(e)}")
        return findings
