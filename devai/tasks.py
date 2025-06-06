import os
import yaml
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import networkx as nx
from asteval import Interpreter

from .config import logger
from .analyzer import CodeAnalyzer
from .memory import MemoryManager
from .ai_model import AIModel


class TaskManager:
    def __init__(self, task_file: str, code_analyzer: CodeAnalyzer, memory: MemoryManager, ai_model: Optional[AIModel] = None):
        self.tasks = self._load_tasks(task_file)
        self.code_analyzer = code_analyzer
        self.memory = memory
        self.ai_model = ai_model
        self.aeval = Interpreter()
        self._setup_default_tasks()

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
            }
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
                        prompt = (
                            f"Analise o código a seguir e sugira melhorias de forma breve:\n{chunk['code']}\n"
                            f"Problemas detectados: {', '.join(analysis['issues'] + analysis['rule_findings'])}"
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
