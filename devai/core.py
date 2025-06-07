import asyncio
import os
import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional
import ast
from pathlib import Path

import networkx as nx
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import config, logger, metrics
from .complexity_tracker import ComplexityTracker
from .memory import MemoryManager
from .analyzer import CodeAnalyzer
from .file_history import FileHistory
from .tasks import TaskManager
from .log_monitor import LogMonitor
from .ai_model import AIModel


class CodeMemoryAI:
    def __init__(self):
        self.memory = MemoryManager(config.MEMORY_DB, config.EMBEDDING_MODEL)
        self.history = FileHistory(config.FILE_HISTORY)
        self.analyzer = CodeAnalyzer(config.CODE_ROOT, self.memory, self.history)
        self.ai_model = AIModel()
        self.project_identity = self._load_project_identity()
        self.tasks = TaskManager(config.TASK_DEFINITIONS, self.analyzer, self.memory, ai_model=self.ai_model)
        self.log_monitor = LogMonitor(self.memory, config.LOG_DIR)
        self.complexity_tracker = ComplexityTracker(config.COMPLEXITY_HISTORY)
        self.app = FastAPI(title="CodeMemoryAI API")
        self._setup_api_routes()
        self.background_tasks = set()
        self.last_average_complexity = 0.0
        self._start_background_tasks()
        logger.info(
            "CodeMemoryAI inicializado com DeepSeek-R1 via OpenRouter", code_root=config.CODE_ROOT
        )

    def _start_background_tasks(self):
        for coro in [self._learning_loop(), self.log_monitor.monitor_logs(), self.analyzer.deep_scan_app(), self.analyzer.watch_app_directory()]:
            task = asyncio.create_task(coro)
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    def _setup_api_routes(self):
        try:
            import jwt  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            jwt = None

        def _auth(token: str) -> bool:
            if not config.API_SECRET:
                return True
            if jwt:
                try:
                    jwt.decode(token, config.API_SECRET, algorithms=["HS256"])
                    return True
                except Exception:
                    return False
            return token == config.API_SECRET

        @self.app.post("/analyze")
        async def analyze_code(query: str):
            return await self.generate_response(query)

        @self.app.get("/memory")
        async def search_memory(query: str, top_k: int = 5, level: str | None = None):
            return self.memory.search(query, top_k, level=level)

        @self.app.post("/feedback")
        async def submit_feedback(memory_id: int, is_positive: bool):
            self.memory.record_feedback(memory_id, is_positive)
            return {"status": "success"}

        @self.app.get("/graph")
        async def show_graph():
            return self.analyzer.get_code_graph()

        @self.app.get("/status")
        async def get_status():
            return {
                "code_chunks": len(self.analyzer.code_chunks),
                "memory_items": len(self.memory.indexed_ids),
                "learned_rules": len(self.analyzer.learned_rules),
                "last_activity": self.analyzer.last_analysis_time.isoformat(),
            }

        @self.app.get("/metrics")
        async def get_metrics():
            return metrics.summary()

        @self.app.get("/admin")
        async def admin_panel():
            return {
                "files_indexed": len(self.analyzer.code_chunks),
                "last_scan": self.analyzer.last_analysis_time.isoformat(),
            }

        @self.app.post("/admin/rescan")
        async def admin_rescan():
            await self.analyzer.deep_scan_app()
            return {"status": "rescanned"}

        @self.app.get("/files")
        async def list_files(path: str = ""):
            """Expose CODE_ROOT contents through the API."""
            return await self.analyzer.list_dir(path)

        @self.app.get("/file")
        async def get_file(file: str, start: int = 1, end: Optional[int] = None):
            lines = await self.analyzer.read_lines(file, start, end)
            return {"file": file, "lines": lines}

        @self.app.post("/file/edit")
        async def edit_file(file: str, line: int, content: str, token: str = ""):
            if not _auth(token):
                return {"error": "unauthorized"}
            ok = await self.analyzer.edit_line(file, line, content)
            return {"status": "ok" if ok else "error"}

        @self.app.post("/file/create")
        async def create_file(file: str, content: str = "", token: str = ""):
            if not _auth(token):
                return {"error": "unauthorized"}
            ok = await self.analyzer.create_file(file, content)
            return {"status": "ok" if ok else "error"}

        @self.app.post("/file/delete")
        async def delete_file(file: str, token: str = ""):
            if not _auth(token):
                return {"error": "unauthorized"}
            ok = await self.analyzer.delete_file(file)
            return {"status": "ok" if ok else "error"}

        @self.app.post("/dir/create")
        async def create_dir(path: str, token: str = ""):
            if not _auth(token):
                return {"error": "unauthorized"}
            ok = await self.analyzer.create_directory(path)
            return {"status": "ok" if ok else "error"}

        @self.app.post("/dir/delete")
        async def delete_dir(path: str, token: str = ""):
            if not _auth(token):
                return {"error": "unauthorized"}
            ok = await self.analyzer.delete_directory(path)
            return {"status": "ok" if ok else "error"}

        @self.app.get("/history")
        async def get_history(file: str, token: str = ""):
            if not _auth(token):
                return {"error": "unauthorized"}
            return self.history.history(file)

        os.makedirs("static", exist_ok=True)
        self.app.mount("/static", StaticFiles(directory="static"), name="static")

    async def _learning_loop(self):
        error_count = 0
        while True:
            try:
                await self.analyzer.scan_app()
                await self.log_monitor.monitor_logs()
                await self._run_scheduled_tasks()
                await self._generate_automatic_insights()
                metrics.update_resources()
                error_count = 0
                await asyncio.sleep(config.LEARNING_LOOP_INTERVAL)
            except Exception as e:
                error_count += 1
                logger.error("Erro no loop de aprendizado", error=str(e), count=error_count)
                wait_time = min(30 * error_count, 300)
                await asyncio.sleep(wait_time)

    async def _run_scheduled_tasks(self):
        if datetime.now().hour == 3:
            await self.tasks.run_task("code_review")
            self.memory.cleanup()

    async def _generate_automatic_insights(self):
        complex_functions = []
        for name, chunk in self.analyzer.code_chunks.items():
            if len(chunk["dependencies"]) > 10:
                complex_functions.append(name)
        if complex_functions:
            self.memory.save(
                {
                    "type": "insight",
                    "content": f"Funções complexas detectadas: {', '.join(complex_functions)}",
                    "metadata": {
                        "functions": complex_functions,
                        "metric": "dependencies_count",
                        "threshold": 10,
                    },
                    "tags": ["insight", "complexity"],
                }
            )

        if self.analyzer.code_chunks:
            avg = sum(c.get("complexity", 0) for c in self.analyzer.code_chunks.values()) / len(self.analyzer.code_chunks)
            if self.last_average_complexity and abs(avg - self.last_average_complexity) / self.last_average_complexity > 0.2:
                self.memory.save(
                    {
                        "type": "metric",
                        "content": "Variação significativa na complexidade média",
                        "metadata": {"previous": self.last_average_complexity, "current": avg},
                        "tags": ["metric", "complexity"],
                    }
                )
            self.last_average_complexity = avg
            self.complexity_tracker.record(avg)

    async def generate_response(self, query: str) -> str:
        try:
            contextual_memories = self.memory.search(query, top_k=3, level="short")
            relevant_chunks = self._find_relevant_code(query)
            graph_summary = self.analyzer.graph_text_summary()
            actions = self.tasks.get_recent_actions()
            logs = await self.log_monitor.collect_recent_logs(50)
            context = self.project_identity
            for chunk in relevant_chunks[:3]:
                context += f"\n// {chunk['file']} ({chunk['type']} {chunk['name']})\n{chunk['code']}"
            from .prompt_engine import build_cot_prompt
            prompt = build_cot_prompt(query, context, graph_summary, contextual_memories, logs, actions)
            return await self.ai_model.generate(prompt)
        except Exception as e:
            logger.error("Erro ao gerar resposta", error=str(e))
            return f"Erro ao gerar resposta: {str(e)}"

    def _find_relevant_code(self, query: str) -> List[Dict]:
        exact = [c for n, c in self.analyzer.code_chunks.items() if n.lower() in query.lower()]
        if exact:
            return exact
        if hasattr(self.analyzer, "embedding_model"):
            query_emb = self.analyzer.embedding_model.encode(query).astype(float)
            sims = []
            for name, chunk in self.analyzer.code_chunks.items():
                chunk_emb = self.analyzer.embedding_model.encode(f"{chunk['name']} {chunk['docstring']}").astype(float)
                sim = float(query_emb @ chunk_emb / (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb)))
                sims.append((sim, chunk))
            sims.sort(reverse=True, key=lambda x: x[0])
            return [c for _, c in sims[:5]]
        return []

    def _extract_tags(self, text: str) -> List[str]:
        tags = set()
        if "erro" in text.lower():
            tags.add("erro")
        if "warning" in text.lower():
            tags.add("warning")
        if "sugestão" in text.lower():
            tags.add("sugestão")
        emoji_tags = {"🚩": "atenção", "⚠️": "aviso", "⚙️": "técnico", "🧠": "aprendizado", "✅": "sucesso"}
        for emoji, tag in emoji_tags.items():
            if emoji in text:
                tags.add(tag)
        return list(tags)

    async def analyze_impact(self, changed: List[str]) -> List[Dict]:
        impacted = defaultdict(list)
        for func in changed:
            if func in self.analyzer.code_graph:
                for dep in nx.descendants(self.analyzer.code_graph, func):
                    impacted[dep].append(func)
        report = []
        for target, deps in impacted.items():
            analysis = await self.tasks.run_task("impact_analysis", target)
            report.append({"target": target, "triggers": deps, "findings": analysis})
        logger.info("Análise de impacto concluída", changed_functions=changed)
        return report

    async def verify_compliance(self, spec: Dict) -> List[str]:
        findings = []
        for func, expected in spec.items():
            if func not in self.analyzer.code_chunks:
                findings.append(f"Função {func} não encontrada no código")
                continue
            chunk = self.analyzer.code_chunks[func]
            if "expected_inputs" in expected:
                actual_inputs = self._extract_inputs(chunk["code"])
                if set(actual_inputs) != set(expected["expected_inputs"]):
                    findings.append(
                        f"⚠️ {func} espera inputs {expected['expected_inputs']} mas recebe {actual_inputs}"
                    )
            if "expected_output" in expected:
                return_type = self._infer_return_type(chunk["code"])
                if return_type and return_type != expected["expected_output"]:
                    findings.append(
                        f"⚠️ {func} parece retornar {return_type} mas deveria retornar {expected['expected_output']}"
                    )
        return findings

    def _extract_inputs(self, code: str) -> List[str]:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return [arg.arg for arg in node.args.args]
        except Exception as e:
            logger.error("Erro ao extrair inputs", error=str(e))
        return []

    def _load_project_identity(self) -> str:
        path = Path("project_identity.yaml")
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except Exception as e:
                logger.error("Erro ao ler project_identity", error=str(e))
        return ""

    def _infer_return_type(self, code: str) -> Optional[str]:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Return):
                    if node.value is None:
                        return "None"
                    if isinstance(node.value, ast.Num):
                        return "number"
                    if isinstance(node.value, ast.Str):
                        return "string"
                    if isinstance(node.value, ast.List):
                        return "list"
                    if isinstance(node.value, ast.Dict):
                        return "dict"
        except Exception as e:
            logger.error("Erro ao inferir tipo de retorno", error=str(e))
        return None

    async def run(self):
        cfg = uvicorn.Config(self.app, host="0.0.0.0", port=config.API_PORT, log_level="info")
        server = uvicorn.Server(cfg)
        await server.serve()
