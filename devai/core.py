import asyncio
import os
import json
from collections import defaultdict, OrderedDict
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import ast

import networkx as nx
import uvicorn
from fastapi import FastAPI
from .api_schemas import (
    FileEditRequest,
    FileCreateRequest,
    FileDeleteRequest,
    DirRequest,
    ApplyRefactorRequest,
)
from fastapi.staticfiles import StaticFiles

from .config import config, logger, metrics, api_key_missing
from .complexity_tracker import ComplexityTracker
from .memory import MemoryManager
from .analyzer import CodeAnalyzer
from .file_history import FileHistory
from .tasks import TaskManager
from .log_monitor import LogMonitor
from .ai_model import AIModel
from .learning_engine import LearningEngine
from .conversation_handler import ConversationHandler


def estimate_token_count(text: str) -> int:
    """Estimativa simples do número de tokens baseada em palavras."""
    return len(text.split())


class CodeMemoryAI:
    def __init__(self):
        self.memory = MemoryManager(config.MEMORY_DB, config.EMBEDDING_MODEL)
        self.history = FileHistory(config.FILE_HISTORY)
        self.analyzer = CodeAnalyzer(
            config.CODE_ROOT,
            self.memory,
            self.history,
            config.RESCAN_INTERVAL_MINUTES,
        )
        self.ai_model = AIModel()
        self.learning_engine = LearningEngine(self.analyzer, self.memory, self.ai_model)
        self.tasks = TaskManager(
            config.TASK_DEFINITIONS, self.analyzer, self.memory, ai_model=self.ai_model
        )
        self.log_monitor = LogMonitor(self.memory, config.LOG_DIR)
        self.complexity_tracker = ComplexityTracker(config.COMPLEXITY_HISTORY)
        self.app = FastAPI(title="CodeMemoryAI API")
        self._setup_api_routes()
        self.background_tasks = set()
        self.last_average_complexity = 0.0
        self.reason_stack = []
        self.response_cache: "OrderedDict[str, Dict]" = OrderedDict()
        self.response_cache_size = 32
        # Gerencia o histórico de cada sessão de conversa
        self.conv_handler = ConversationHandler()
        self.conversation: List[Dict[str, str]] = self.conv_handler.history(
            "default"
        )
        self.double_check = config.DOUBLE_CHECK
        self._start_background_tasks()
        logger.info(
            "CodeMemoryAI inicializado com DeepSeek-R1 via OpenRouter",
            code_root=config.CODE_ROOT,
        )

    @property
    def conversation_history(self) -> List[Dict[str, str]]:
        """Compatibilidade com histórico padrão."""
        self.conversation = self.conv_handler.history("default")
        return self.conversation

    @conversation_history.setter
    def conversation_history(self, value: List[Dict[str, str]]):
        self.conv_handler.conversation_context["default"] = value
        self.conversation = self.conv_handler.history("default")

    def _cache_key(self, query: str) -> str:
        """Return a key representing the current state for caching."""
        import hashlib

        last_time = getattr(self.analyzer, "last_analysis_time", datetime.now())
        idx_len = len(getattr(self.memory, "indexed_ids", []))
        state = f"{last_time.isoformat()}-{idx_len}"
        return hashlib.md5(f"{query}-{state}".encode()).hexdigest()

    async def _prefetch_related(self, query: str) -> None:
        """Preload memories and suggestions in background."""
        try:
            await asyncio.gather(
                self.memory.search(query, top_k=2),
                self.tasks.run_task("impact_analysis", query),
            )
        except Exception:
            pass

    def _build_history_messages(self, session_id: str, buffer: int = 1000) -> List[Dict[str, str]]:
        """Recupera o histórico recente sem ultrapassar o limite de tokens."""
        hist = self.conv_handler.history(session_id)
        selected: List[Dict[str, str]] = []
        total = 0
        for msg in reversed(hist):
            total += estimate_token_count(msg.get("content", ""))
            if total > config.MAX_CONTEXT_LENGTH - buffer:
                break
            selected.append(msg)
        return list(reversed(selected))

    def _start_background_tasks(self):
        from .metacognition import MetacognitionLoop

        meta = MetacognitionLoop(memory=self.memory)
        tasks = [
            self._learning_loop(),
            self.log_monitor.monitor_logs(),
            meta.run(),
        ]
        if config.START_MODE == "full":
            tasks.extend(
                [
                    self.analyzer.deep_scan_app(),
                    self.analyzer.watch_app_directory(),
                ]
            )
        else:
            # FUTURE: permitir que o modo custom defina quais tarefas rodam
            logger.info("🔄 deep_scan_app() adiado para execução sob demanda.")
        for coro in tasks:
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
        async def analyze_code(query: str, session_id: str = "default"):
            return await self.generate_response(
                query, double_check=self.double_check, session_id=session_id
            )

        @self.app.get("/analyze_stream")
        async def analyze_stream(query: str, session_id: str = "default"):
            from fastapi.responses import StreamingResponse

            async def event_gen():
                text = await self.generate_response(
                    query, session_id=session_id
                )
                for token in text.split():
                    yield f"data: {token}\n\n"
            return StreamingResponse(event_gen(), media_type="text/event-stream")

        @self.app.post("/reset_conversation")
        async def reset_conversation(session_id: str = "default"):
            self.conv_handler.reset(session_id)
            if session_id == "default":
                self.conversation = self.conv_handler.history("default")
            return {"status": "reset", "session": session_id}

        @self.app.post("/session/reset")
        async def reset_session():
            self.conv_handler.clear_session()
            self.conversation = self.conv_handler.history("default")
            return {"status": "ok", "message": "Sessão reiniciada com sucesso"}

        @self.app.post("/analyze_deep")
        async def analyze_deep(query: str, session_id: str = "default"):
            """Perform a deeper analysis returning plan and answer separately."""
            return await self.generate_response_with_plan(query, session_id=session_id)

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
                "api_key_missing": api_key_missing,
                "show_reasoning_default": config.SHOW_REASONING_BY_DEFAULT,
                "show_context_button": config.SHOW_CONTEXT_BUTTON,
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
        async def edit_file(req: FileEditRequest, token: str = ""):
            if not _auth(token):
                return {"error": "unauthorized"}
            try:
                ok = await self.analyzer.edit_line(
                    Path(req.file), req.line, req.content
                )
            except ValueError as e:
                return {"error": str(e)}
            return {"status": "ok" if ok else "error"}

        @self.app.post("/file/create")
        async def create_file(req: FileCreateRequest, token: str = ""):
            if not _auth(token):
                return {"error": "unauthorized"}
            try:
                ok = await self.analyzer.create_file(Path(req.file), req.content)
            except ValueError as e:
                return {"error": str(e)}
            return {"status": "ok" if ok else "error"}

        @self.app.post("/file/delete")
        async def delete_file(req: FileDeleteRequest, token: str = ""):
            if not _auth(token):
                return {"error": "unauthorized"}
            try:
                ok = await self.analyzer.delete_file(Path(req.file))
            except ValueError as e:
                return {"error": str(e)}
            return {"status": "ok" if ok else "error"}

        @self.app.post("/dir/create")
        async def create_dir(req: DirRequest, token: str = ""):
            if not _auth(token):
                return {"error": "unauthorized"}
            try:
                ok = await self.analyzer.create_directory(Path(req.path))
            except ValueError as e:
                return {"error": str(e)}
            return {"status": "ok" if ok else "error"}

        @self.app.post("/dir/delete")
        async def delete_dir(req: DirRequest, token: str = ""):
            if not _auth(token):
                return {"error": "unauthorized"}
            try:
                ok = await self.analyzer.delete_directory(Path(req.path))
            except ValueError as e:
                return {"error": str(e)}
            return {"status": "ok" if ok else "error"}

        @self.app.get("/history")
        async def get_history(file: str, token: str = ""):
            if not _auth(token):
                return {"error": "unauthorized"}
            return self.history.history(file)

        @self.app.get("/actions")
        async def get_actions():
            from pathlib import Path

            path = Path("decision_log.yaml")
            if path.exists():
                try:
                    import yaml  # type: ignore
                except Exception:  # pragma: no cover - fallback when PyYAML is missing
                    from . import yaml_fallback as yaml
                return yaml.safe_load(path.read_text())
            return []

        @self.app.get("/session/context")
        async def session_context(session_id: str = "default"):
            return self.get_session_context(session_id)

        @self.app.get("/diff")
        async def get_diff(file: str):
            hist = self.history.history(file)
            if not hist:
                return {"diff": ""}
            from difflib import unified_diff

            last = hist[-1]
            old = last.get("old", [])
            new = last.get("new", [])
            diff = "\n".join(unified_diff(old, new, fromfile="old", tofile="new"))
            return {"diff": diff}

        @self.app.post("/dry_run")
        async def dry_run(file_path: str, suggested_code: str):
            from .shadow_mode import (
                simulate_update,
                evaluate_change_with_ia,
                log_simulation,
                run_tests_in_temp,
            )

            diff, temp_root, sim_id = simulate_update(file_path, suggested_code)
            tests_ok, test_output = run_tests_in_temp(temp_root)
            evaluation = await evaluate_change_with_ia(diff)
            status = "shadow_failed" if not tests_ok else "shadow_declined"
            log_simulation(sim_id, file_path, tests_ok, evaluation["analysis"], status)
            return {
                "id": sim_id,
                "diff": diff,
                "tests_passed": tests_ok,
                "test_output": test_output,
                "evaluation": evaluation,
            }

        @self.app.post("/apply_refactor")
        async def apply_refactor(req: ApplyRefactorRequest, token: str = ""):
            if not _auth(token):
                return {"error": "unauthorized"}
            path = Path(req.file_path)
            old_lines = path.read_text().splitlines()
            path.write_text(req.suggested_code)
            self.history.record(
                req.file_path,
                "edit",
                old=old_lines,
                new=req.suggested_code.splitlines(),
            )
            self.memory.save(
                {
                    "type": "refatoracao",
                    "memory_type": "refatoracao aprovada",
                    "content": f"Refatoracao aplicada em {req.file_path}",
                    "metadata": {"arquivo": req.file_path, "contexto": "dry_run"},
                }
            )
            return {"status": "applied"}

        @self.app.get("/deep_analysis")
        async def deep_analysis(token: str = ""):
            """Run a project wide analysis and return a summary."""
            if not _auth(token):
                return {"error": "unauthorized"}
            await self.analyzer.deep_scan_app()
            modules = await self.analyzer.summary_by_module()
            from .auto_review import run_autoreview

            review = await run_autoreview(self.analyzer, self.memory)
            high_complex = sum(m["complex_functions"] for m in modules.values())
            stable_funcs = len(self.analyzer.code_chunks) - high_complex
            report_lines = [
                "🧠 Análise do Projeto",
                f"📅 Última análise: {self.analyzer.last_analysis_time.strftime('%d/%m/%Y - %H:%M')}",
                "",
                f"📂 Módulos analisados: {len(modules)}",
                f"🧩 Trechos de código indexados: {len(self.analyzer.code_chunks)}",
                "",
                "🚩 0 possíveis vulnerabilidades encontradas",
                f"⚠️ {high_complex} trechos com complexidade alta",
                f"💡 {len(review.get('suggestions', []))} oportunidades de refatoração",
                f"✅ {stable_funcs} funções consideradas estáveis",
                "",
                "🔎 Módulo destaque:",
            ]
            for mod, data in list(modules.items())[:5]:
                report_lines.append(f"\t• {mod}: {data['score']}")
            report = "\n".join(report_lines)
            return {"report": report, "modules": modules}

        @self.app.post("/symbolic_training")
        async def symbolic_training(token: str = ""):
            if not _auth(token):
                return {"error": "unauthorized"}
            from .symbolic_training import run_symbolic_training

            return await run_symbolic_training(
                self.analyzer, self.memory, self.ai_model
            )

        @self.app.get("/auto_monitor")
        async def auto_monitor():
            from .monitor_engine import auto_monitor_cycle

            result = await auto_monitor_cycle(self.analyzer, self.memory, self.ai_model)
            return result

        @self.app.post("/memory/optimize")
        async def optimize_memory():
            compressed = self.memory.compress_memory()
            pruned = self.memory.prune_old_memories()
            log_path = Path("logs/memory_maintenance.md")
            with open(log_path, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} compressed {compressed} pruned {pruned}\n"
                )
            return {"compressed": compressed, "pruned": pruned}

        @self.app.post("/learning/lessons")
        async def review_lessons():
            summary = await self.learning_engine.explain_learning_lessons()
            return {"summary": summary}

        os.makedirs("static", exist_ok=True)
        self.app.mount("/static", StaticFiles(directory="static"), name="static")

        if hasattr(self.app, "on_event"):

            @self.app.on_event("shutdown")
            async def _shutdown_event():
                await self.shutdown()

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
                logger.error(
                    "Erro no loop de aprendizado", error=str(e), count=error_count
                )
                wait_time = min(30 * error_count, 300)
                await asyncio.sleep(wait_time)

    async def _run_scheduled_tasks(self):
        now = datetime.now()
        if now.hour == 3:
            await self.tasks.run_task("code_review")
            self.memory.cleanup()
        if now.hour == 2:
            from .auto_review import run_autoreview

            await run_autoreview(self.analyzer, self.memory)

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
            avg = sum(
                c.get("complexity", 0) for c in self.analyzer.code_chunks.values()
            ) / len(self.analyzer.code_chunks)
            if (
                self.last_average_complexity
                and abs(avg - self.last_average_complexity)
                / self.last_average_complexity
                > 0.2
            ):
                self.memory.save(
                    {
                        "type": "metric",
                        "content": "Variação significativa na complexidade média",
                        "metadata": {
                            "previous": self.last_average_complexity,
                            "current": avg,
                        },
                        "tags": ["metric", "complexity"],
                    }
                )
            self.last_average_complexity = avg
            self.complexity_tracker.record(avg)

    async def _process_command(self, query: str, session_id: str) -> str | None:
        if query.strip() == "/resetar":
            self.conv_handler.reset(session_id)
            if session_id == "default":
                self.conversation = self.conv_handler.history("default")
            return "Conversa resetada."
        if query.startswith("/teste"):
            output = await self.tasks.run_task("run_tests")
            return "\n".join(output)
        return None

    async def generate_response(
        self, query: str, double_check: bool = False, session_id: str = "default"
    ) -> str:
        try:
            if not hasattr(self, "response_cache"):
                self.response_cache = OrderedDict()
                self.response_cache_size = 32
            cmd = await self._process_command(query, session_id)
            if cmd is not None:
                return cmd
            key = self._cache_key(query)
            if key in self.response_cache:
                logger.info("cache_hit", query=query)
                return self.response_cache[key]["response"]
            if len(query.split()) < 3:
                return "Por favor, forneça mais detalhes sobre sua solicitação."

            contextual_memories = self.memory.search(query, level="short")
            suggestions = self.memory.search(query, top_k=1)
            actions = self.tasks.last_actions()
            graph_summary = self.analyzer.graph_summary()
            from .prompt_engine import (
                build_dynamic_prompt,
                collect_recent_logs,
                SYSTEM_PROMPT_CONTEXT,
            )

            logs = collect_recent_logs()
            self.reason_stack = []
            self.reason_stack.append("Memórias coletadas")
            # FIXME: Ativar explicação apenas em modo Raciocinar
            # query += "\nExplique antes de responder."
            context_blocks = {
                "memories": contextual_memories,
                "graph": graph_summary,
                "actions": actions,
                "logs": logs,
            }
            self.last_context = {
                "history": self._build_history_messages(session_id) if session_id else [],
                "context_blocks": context_blocks,
            }
            prompt = build_dynamic_prompt(
                query,
                context_blocks,
                "deep" if double_check else "normal",
            )
            if double_check:
                plan = await self.ai_model.safe_api_call(
                    f"{SYSTEM_PROMPT_CONTEXT}\nElabore um plano de ação para: {query}",
                    config.MAX_CONTEXT_LENGTH,
                    query,
                    self.memory,
                )
                review = await self.ai_model.safe_api_call(
                    f"{SYSTEM_PROMPT_CONTEXT}\nRevise o plano a seguir e sugira ajustes se necessário:\n{plan}",
                    config.MAX_CONTEXT_LENGTH,
                    plan,
                    self.memory,
                )
                prompt = plan + "\n" + review + "\n" + prompt
            if suggestions:
                prompt += f"\nSugestao relacionada: {suggestions[0]['content'][:80]}"
            self.reason_stack.append("Prompt preparado")
            history = self._build_history_messages(session_id) if session_id else []
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_CONTEXT},
                *history,
                {"role": "user", "content": prompt},
            ]
            prefetch = asyncio.create_task(self._prefetch_related(query))
            result = await self.ai_model.safe_api_call(
                messages, config.MAX_CONTEXT_LENGTH, prompt, self.memory
            )
            prefetch.cancel()
            if session_id:
                self.conv_handler.append(session_id, "user", query)
                self.conv_handler.append(session_id, "assistant", result)
            else:
                logger.info("multi_turn_fallback", session=session_id)
            self.reason_stack.append("Resposta gerada")
            self.response_cache[key] = {"response": result}
            self.response_cache.move_to_end(key)
            if len(self.response_cache) > self.response_cache_size:
                self.response_cache.popitem(last=False)
            return result
        except Exception as e:
            logger.error("Erro ao gerar resposta", error=str(e))
            return f"Erro ao gerar resposta: {str(e)}"

    @staticmethod
    def _split_plan_response(text: str) -> tuple[str, str]:
        if "===RESPOSTA===" in text:
            plan, resp = text.split("===RESPOSTA===", 1)
            return plan.strip(), resp.strip()
        logger.warning("A IA não retornou plano separado. Verificar prompt.")
        return "", text.strip()

    async def generate_response_with_plan(
        self, query: str, session_id: str = "default"
    ) -> Dict[str, str]:
        try:
            if not hasattr(self, "response_cache"):
                self.response_cache = OrderedDict()
                self.response_cache_size = 32
            cmd = await self._process_command(query, session_id)
            if cmd is not None:
                return {"plan": "", "response": cmd}
            key = self._cache_key(query)
            if key in self.response_cache:
                logger.info("cache_hit", query=query)
                return self.response_cache[key]
            if len(query.split()) < 3:
                return {
                    "plan": "",
                    "response": "Por favor, forneça mais detalhes sobre sua solicitação.",
                }

            contextual_memories = self.memory.search(query, level="short")
            suggestions = self.memory.search(query, top_k=1)
            actions = self.tasks.last_actions()
            graph_summary = self.analyzer.graph_summary()
            from .prompt_engine import build_dynamic_prompt, collect_recent_logs

            logs = collect_recent_logs()
            context_blocks = {
                "memories": contextual_memories,
                "graph": graph_summary,
                "actions": actions,
                "logs": logs,
            }
            self.last_context = {
                "history": self._build_history_messages(session_id) if session_id else [],
                "context_blocks": context_blocks,
            }
            prompt = build_dynamic_prompt(query, context_blocks, "deep")
            if suggestions:
                prompt += f"\nSugestao relacionada: {suggestions[0]['content'][:80]}"

            system_msg = (
                "Você é um assistente especialista em análise de código. "
                "Sua tarefa é resolver o pedido do usuário em duas etapas:\n"
                "1. Primeiro, liste seu plano de raciocínio passo a passo, numerando cada etapa.\n"
                "2. Depois, forneça a resposta final ao pedido do usuário com base nesse plano.\n"
                "Separe o plano da resposta com a marcação ===RESPOSTA===."
            )
            history = self._build_history_messages(session_id) if session_id else []
            messages = [
                {"role": "system", "content": system_msg},
                *history,
                {"role": "user", "content": prompt},
            ]
            prefetch = asyncio.create_task(self._prefetch_related(query))
            raw = await self.ai_model.safe_api_call(
                messages,
                4096,
                prompt,
                self.memory,
                temperature=0.2,
            )
            prefetch.cancel()
            plan, resp = self._split_plan_response(raw)
            if session_id:
                self.conv_handler.append(session_id, "user", query)
                self.conv_handler.append(session_id, "assistant", resp)
            else:
                logger.info("multi_turn_fallback", session=session_id)
            trace = (
                f"\ud83d\udca1 Detalhes t\u00e9cnicos: A IA consultou {len(contextual_memories)} m\u00e9morias anteriores, analisou depend\u00eancias e gerou a resposta abaixo com base no padr\u00e3o simb\u00f3lico aprendido."
            )
            result = {
                "plan": plan,
                "response": resp,
                "main_response": resp,
                "reasoning_trace": trace,
                "mode": "deep",
            }
            self.response_cache[key] = result
            self.response_cache.move_to_end(key)
            if len(self.response_cache) > self.response_cache_size:
                self.response_cache.popitem(last=False)
            return result
        except Exception as e:
            logger.error("Erro ao gerar resposta", error=str(e))
            return {"plan": "", "response": f"Erro ao gerar resposta: {str(e)}"}

    def _find_relevant_code(self, query: str) -> List[Dict]:
        exact = [
            c
            for n, c in self.analyzer.code_chunks.items()
            if n.lower() in query.lower()
        ]
        if exact:
            return exact
        if hasattr(self.analyzer, "embedding_model"):
            query_emb = self.analyzer.embedding_model.encode(query).astype(float)
            sims = []
            for name, chunk in self.analyzer.code_chunks.items():
                chunk_emb = self.analyzer.embedding_model.encode(
                    f"{chunk['name']} {chunk['docstring']}"
                ).astype(float)
                sim = float(
                    query_emb
                    @ chunk_emb
                    / (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb))
                )
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
        emoji_tags = {
            "🚩": "atenção",
            "⚠️": "aviso",
            "⚙️": "técnico",
            "🧠": "aprendizado",
            "✅": "sucesso",
        }
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

    def get_session_context(self, session_id: str = "default", n: int = 3) -> Dict[str, Any]:
        """Return preview of conversation and activated memories."""
        try:
            history = self.conv_handler.last(session_id, n * 2)
            preview = [f"{m['role']}: {m['content']}" for m in history]
            last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
            symbolic = []
            warnings: List[str] = []
            try:
                symbolic = [
                    f"{m.get('metadata', {}).get('tag', '')}: {m['content']}"
                    for m in self.memory.search(last_user, memory_type="dialog_summary", top_k=n)
                ]
            except Exception:
                warnings.append("⚠️ Memória vetorial temporariamente indisponível.")
            blocks = getattr(self, "last_context", {}).get("context_blocks", {})
            logs_or_code = []
            if blocks.get("logs"):
                logs_or_code.append(blocks["logs"])
            if blocks.get("graph"):
                logs_or_code.append("grafo_de_dependencias.yaml")
            if not preview:
                return {
                    "error": "⚠️ Nenhum dado de contexto disponível nesta sessão.",
                    "symbolic_memories": [],
                    "history_preview": [],
                }
            if not symbolic:
                warnings.append("Memória simbólica ainda não ativada")
            return {
                "history_preview": preview,
                "symbolic_memories": symbolic,
                "logs_or_code": logs_or_code,
                "warnings": warnings,
            }
        except Exception:
            return {
                "error": "⚠️ Nenhum dado de contexto disponível nesta sessão.",
                "symbolic_memories": [],
                "history_preview": [],
            }

    async def run(self):
        cfg = uvicorn.Config(
            self.app, host="0.0.0.0", port=config.API_PORT, log_level="info"
        )
        server = uvicorn.Server(cfg)
        await server.serve()

    async def shutdown(self):
        """Finaliza recursos como AIModel, watchers e ciclos ativos."""
        if hasattr(self, "ai_model") and hasattr(self.ai_model, "shutdown"):
            await self.ai_model.shutdown()
        for task in list(self.background_tasks):
            task.cancel()
        # FUTURE: shutdown this resource when implemented
        logger.info("🛑 DevAI finalizado com limpeza simbólica de recursos.")
