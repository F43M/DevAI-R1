import asyncio
import os
import json
import hashlib
from collections import defaultdict, OrderedDict
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
import re
from pathlib import Path
import ast
import contextlib

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
from .error_handler import load_persisted_errors, persist_errors
from .complexity_tracker import ComplexityTracker
from .memory import MemoryManager
from .analyzer import CodeAnalyzer
from .file_history import FileHistory
from .tasks import TaskManager
from .log_monitor import LogMonitor
from .ai_model import AIModel
from .learning_engine import LearningEngine
from .conversation_handler import ConversationHandler
from .intent_router import detect_intent
try:
    from . import rlhf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    rlhf = None


def estimate_token_count(text: str) -> int:
    """Estimativa simples do número de tokens baseada em palavras."""
    return len(text.split())


async def run_scheduled_rlhf(memory: MemoryManager) -> dict:
    """Run RLHF fine-tuning when the example dataset changes."""

    ds_dir = Path(config.RLHF_OUTPUT_DIR) / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(ds_dir.glob("*.sha256"))
    prev_hash = existing[-1].read_text().strip() if existing else ""

    tuner = rlhf.RLFineTuner(memory)
    before_files = set(ds_dir.glob("*.sha256"))
    data, new_hash = tuner.collect_examples(with_hash=True)
    after_files = set(ds_dir.glob("*.sha256"))
    new_file = (
        sorted(after_files - before_files)[-1] if after_files - before_files else None
    )

    if new_hash == prev_hash:
        if new_file:
            try:
                Path(new_file).unlink()
            except Exception:
                pass
        logger.info("RLHF dataset unchanged")
        return {"status": "skipped", "reason": "dataset_unchanged"}

    if len(data) < config.RLHF_THRESHOLD:
        logger.info("RLHF threshold not reached", num_examples=len(data))
        return {"status": "skipped", "reason": "threshold", "num_examples": len(data)}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config.RLHF_OUTPUT_DIR) / ts
    metrics = await rlhf.train_from_memory(config.model_name, str(out_dir))
    memory.save(
        {
            "type": "rlhf_training",
            "metadata": {
                "hash": new_hash,
                "num_examples": len(data),
                "output_dir": str(out_dir),
            },
        }
    )
    metrics.update(
        {"hash": new_hash, "num_examples": len(data), "output_dir": str(out_dir)}
    )
    return metrics


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
        self.background_tasks: Dict[str, asyncio.Task] = {}
        self.task_status: Dict[str, Dict[str, Any]] = {}
        # Rastreamento de loops e watchers em execução
        self.watchers: Dict[str, asyncio.Task] = {}
        self.last_average_complexity = 0.0
        self.reason_stack = []
        self.response_cache: "OrderedDict[str, Dict]" = OrderedDict()
        self.response_cache_size = 32
        # Gerencia o histórico de cada sessão de conversa
        self.conv_handler = ConversationHandler(memory=self.memory)
        self.conversation: List[Dict[str, str]] = self.conv_handler.history("default")
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

    def _build_history_messages(
        self, session_id: str, buffer: int = 1000
    ) -> List[Dict[str, str]]:
        """Recupera o histórico recente sem ultrapassar o limite de tokens."""
        # always sync with disk and prune to max_history
        self.conv_handler.prune(session_id)
        hist = self.conv_handler.last(session_id, self.conv_handler.max_history)
        selected: List[Dict[str, str]] = []
        total = 0
        for msg in reversed(hist):
            total += estimate_token_count(msg.get("content", ""))
            if total > config.MAX_CONTEXT_LENGTH - buffer:
                break
            selected.append(msg)
        return list(reversed(selected))

    def _symbolic_training_progress(self) -> float:
        """Read progress of symbolic training from status file."""
        try:
            path = Path(config.LOG_DIR) / "symbolic_training_status.json"
            if path.exists():
                data = json.loads(path.read_text())
                return float(data.get("progress", 0.0))
        except Exception:
            pass
        return 0.0

    def _learning_progress(self) -> float:
        """Read progress of learning from status file."""
        try:
            path = Path(config.LOG_DIR) / "learning_progress.json"
            if path.exists():
                data = json.loads(path.read_text())
                return float(data.get("progress", 0.0))
        except Exception:
            pass
        return 0.0

    def start_deep_scan(self) -> bool:
        """Queue deep_scan_app as background task if not running."""
        if not hasattr(self, "background_tasks"):
            self.background_tasks = {}
        if not hasattr(self, "task_status"):
            self.task_status = {}
        if "deep_scan_app" in self.background_tasks:
            return False
        self.task_status["deep_scan_app"] = {"progress": 0.0, "running": True}

        async def _run():
            def _update(p: float, _msg: str) -> None:
                self.task_status["deep_scan_app"]["progress"] = round(p / 100, 2)

            await self.analyzer.deep_scan_app(progress_cb=_update)

        task = asyncio.create_task(_run(), name="deep_scan_app")
        self.background_tasks["deep_scan_app"] = task
        if not hasattr(self, "watchers"):
            self.watchers = {}
        self.watchers["deep_scan_app"] = task

        def _done(_t: asyncio.Task) -> None:
            self.background_tasks.pop("deep_scan_app", None)
            if hasattr(self, "watchers"):
                self.watchers.pop("deep_scan_app", None)
            self.task_status["deep_scan_app"] = {"progress": 1.0, "running": False}

        task.add_done_callback(_done)
        return True

    def queue_symbolic_training(self) -> bool:
        """Run symbolic training in background and notify when done."""
        if not hasattr(self, "background_tasks"):
            self.background_tasks = {}
        if not hasattr(self, "task_status"):
            self.task_status = {}
        if "symbolic_training" in self.background_tasks:
            return False

        async def _run():
            from .symbolic_training import run_symbolic_training

            monitor = asyncio.create_task(_monitor(), name="symbolic_training_progress")
            try:
                result = await run_symbolic_training(
                    self.analyzer, self.memory, self.ai_model
                )
            finally:
                monitor.cancel()
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await monitor

            try:
                from .notifier import Notifier

                Notifier().send(
                    "Treinamento simbólico concluído", result.get("report", "")[:200]
                )
            except Exception:
                pass
            return result

        async def _monitor() -> None:
            while True:
                await asyncio.sleep(0.1)
                self.task_status["symbolic_training"]["progress"] = round(
                    self._symbolic_training_progress(),
                    2,
                )

        self.task_status["symbolic_training"] = {"progress": 0.0, "running": True}
        task = asyncio.create_task(_run(), name="symbolic_training")
        self.background_tasks["symbolic_training"] = task
        if not hasattr(self, "watchers"):
            self.watchers = {}
        self.watchers["symbolic_training"] = task

        def _done(_t: asyncio.Task) -> None:
            self.background_tasks.pop("symbolic_training", None)
            if hasattr(self, "watchers"):
                self.watchers.pop("symbolic_training", None)
            self.task_status["symbolic_training"] = {"progress": 1.0, "running": False}

        task.add_done_callback(_done)
        return True

    def _start_background_tasks(self):
        from .metacognition import MetacognitionLoop

        if not hasattr(self, "watchers"):
            self.watchers = {}

        mode = getattr(config, "OPERATING_MODE", "standard")
        if mode == "sandbox":
            logger.info("Modo sandbox ativo: background tasks desativadas.")
            return

        meta = MetacognitionLoop(memory=self.memory)
        task_coros = [
            ("learning_loop", self._learning_loop()),
            ("metacognition", meta.run()),
        ]
        watcher_names = {
            "learning_loop",
            "metacognition",
            "log_monitor",
            "auto_monitor_cycle",
            "watch_app_directory",
        }
        if config.START_MODE != "custom" or "monitor" in config.START_TASKS:
            task_coros.insert(1, ("log_monitor", self.log_monitor.monitor_logs()))
        else:
            logger.info("Monitor de logs desativado.")
        mode = getattr(config, "OPERATING_MODE", "standard")
        if mode == "continuous":
            from .monitor_engine import auto_monitor_cycle

            task_coros.append(
                (
                    "auto_monitor_cycle",
                    auto_monitor_cycle(self.analyzer, self.memory, self.ai_model),
                )
            )
        run_scan = False
        run_watch = False
        if config.START_MODE == "full":
            run_scan = True
            run_watch = True
        elif config.START_MODE == "custom":
            run_scan = "scan" in config.START_TASKS
            run_watch = "watch" in config.START_TASKS
        if run_scan:
            self.start_deep_scan()
        else:
            logger.info("🔄 deep_scan_app() adiado para execução sob demanda.")
        if run_watch:
            task_coros.append(
                ("watch_app_directory", self.analyzer.watch_app_directory())
            )
        for name, coro in task_coros:
            task = asyncio.create_task(coro, name=name)
            self.background_tasks[name] = task
            if name in watcher_names:
                self.watchers[name] = task
            task.add_done_callback(
                lambda t, n=name: (
                    self.background_tasks.pop(n, None),
                    self.watchers.pop(n, None),
                )
            )

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
            # reload conversation from disk before processing
            self.conv_handler.history(session_id)
            result = await self.generate_response(
                query, double_check=self.double_check, session_id=session_id
            )
            if session_id == "default":
                self.conversation = self.conv_handler.history("default")
            return result

        @self.app.get("/analyze_stream")
        async def analyze_stream(query: str, session_id: str = "default"):
            from fastapi.responses import StreamingResponse

            async def event_gen():
                async for token in self.generate_response_stream(
                    query, session_id=session_id
                ):
                    yield f"data: {token}\n\n"

            return StreamingResponse(event_gen(), media_type="text/event-stream")

        @self.app.post("/reset_conversation")
        async def reset_conversation(session_id: str):
            self.conv_handler.reset(session_id)
            if session_id == "default":
                self.conversation = self.conv_handler.history("default")
            return {"status": "reset", "session": session_id}

        @self.app.post("/session/reset")
        async def reset_session(session_id: str = "default"):
            self.conv_handler.clear_session(session_id)
            if session_id == "default":
                self.conversation = self.conv_handler.history("default")
            return {"status": "ok", "session": session_id}

        @self.app.get("/session/history")
        async def session_history(session_id: str = "default"):
            return self.conv_handler.history(session_id)

        @self.app.get("/history")
        async def history(session_id: str = "default"):
            return self.conv_handler.history(session_id)

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
            tasks_info = {}
            for name, task in self.background_tasks.items():
                progress = None
                if name == "deep_scan_app":
                    progress = round(self.analyzer.scan_progress, 2)
                elif name == "symbolic_training":
                    progress = round(self._symbolic_training_progress(), 2)
                status = {
                    "running": not task.done(),
                    "progress": progress,
                }
                tasks_info[name] = status
            return {
                "code_chunks": len(self.analyzer.code_chunks),
                "memory_items": len(self.memory.indexed_ids),
                "learned_rules": len(self.analyzer.learned_rules),
                "last_activity": self.analyzer.last_analysis_time.isoformat(),
                "api_key_missing": api_key_missing,
                "show_reasoning_default": config.SHOW_REASONING_BY_DEFAULT,
                "show_context_button": config.SHOW_CONTEXT_BUTTON,
                "scan_progress": round(self.analyzer.scan_progress, 2),
                "learning_progress": round(self._learning_progress(), 2),
                "background_tasks": tasks_info,
            }

        @self.app.get("/metrics")
        async def get_metrics():
            return metrics.summary()

        @self.app.get("/logs/recent")
        async def recent_logs(limit: int = 20):
            return self.memory.recent_entries("log_analysis", limit)

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

        @self.app.get("/file_history")
        async def get_file_history(file: str, token: str = ""):
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

        @self.app.get("/context/search")
        async def context_search(tag: str, session_id: str = "default"):
            return self.conv_handler.search_history(session_id, tag)

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
            )

            diff, tests_ok, test_output, sim_id, patch_file = simulate_update(
                file_path, suggested_code
            )
            evaluation = await evaluate_change_with_ia(diff)
            status = "shadow_failed" if not tests_ok else "shadow_declined"
            patch_hash = hashlib.sha1(diff.encode("utf-8")).hexdigest()
            log_simulation(
                sim_id,
                file_path,
                tests_ok,
                evaluation["analysis"],
                status,
                patch_hash=patch_hash,
                test_output=test_output,
            )
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

            from .update_manager import UpdateManager

            path = Path(req.file_path)
            old_lines = path.read_text().splitlines()

            def apply_func(p: Path) -> None:
                p.write_text(req.suggested_code)

            updater = UpdateManager()
            success = updater.safe_apply(path, apply_func, keep_backup=True)

            if success:
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
                status = "applied"
            else:
                status = "rolled_back"

            return {"status": status}

        @self.app.get("/deep_analysis")
        async def deep_analysis(token: str = ""):
            """Run a project wide analysis and return a summary."""
            if not _auth(token):
                return {"error": "unauthorized"}
            if self.start_deep_scan():
                task = self.background_tasks.get("deep_scan_app")
                if hasattr(task, "__await__"):
                    await task
                else:
                    await self.analyzer.deep_scan_app()
            elif "deep_scan_app" in self.background_tasks:
                task = self.background_tasks["deep_scan_app"]
                if hasattr(task, "__await__"):
                    await task
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
            if "symbolic_training" in self.background_tasks:
                return {"status": "running"}
            self.queue_symbolic_training()
            return {"status": "queued"}

        @self.app.post("/train/rlhf")
        async def train_rlhf(token: str = ""):
            if not _auth(token):
                return {"error": "unauthorized"}
            return await run_scheduled_rlhf(self.memory)

        @self.app.get("/auto_monitor")
        async def auto_monitor():
            from .monitor_engine import auto_monitor_cycle

            result = await auto_monitor_cycle(self.analyzer, self.memory, self.ai_model)
            return result

        @self.app.get("/monitor/history")
        async def monitor_history():
            cur = self.memory.conn.cursor()
            cur.execute(
                "SELECT timestamp, reason, training_executed, new_rules FROM monitoring_history ORDER BY timestamp DESC"
            )
            rows = [
                {
                    "timestamp": r[0],
                    "reason": r[1],
                    "training_executed": bool(r[2]),
                    "new_rules": r[3],
                }
                for r in cur.fetchall()
            ]
            return rows

        @self.app.get("/complexity/history")
        async def complexity_history(limit: int = 100, window: int = 5):
            history = self.complexity_tracker.get_history()[-limit:]
            trend = self.complexity_tracker.summarize_trend(window)
            return {"history": history, "trend": trend}

        @self.app.get("/metacognition/summary")
        async def metacognition_summary():
            path = Path("devai/meta/score_map.json")
            if not path.exists():
                return {"critical": {}}
            try:
                scores = json.loads(path.read_text())
            except Exception:
                scores = {}
            negatives = {f: s for f, s in scores.items() if s < 0}
            return {"critical": negatives}

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

        from .approval import wait_for_request, resolve_request, verify_token

        @self.app.get("/approval_request")
        async def approval_wait():
            return await wait_for_request()

        @self.app.post("/approval_request")
        async def approval_answer(approved: bool, token: str):
            if not verify_token(token):
                return {"status": "invalid"}
            resolve_request(approved)
            return {"status": "ok"}

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
        if now.hour == 4:
            await run_scheduled_rlhf(self.memory)

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

            intent = detect_intent(query)
            from .prompt_engine import (
                build_dynamic_prompt,
                gather_context_async,
                generate_plan_async,
                generate_final_async,
                SYSTEM_PROMPT_CONTEXT,
            )

            context_blocks, suggestions = await gather_context_async(self, query)
            self.reason_stack = []
            self.reason_stack.append("Memórias coletadas")
            mode = getattr(config, "MODE", "")
            if mode == "reasoning":
                query += "\nExplique antes de responder."
            elif mode:
                logger.info("Modo %s não reconhecido", mode)
            self.last_context = {
                "history": (
                    self._build_history_messages(session_id) if session_id else []
                ),
                "context_blocks": context_blocks,
            }
            prompt = build_dynamic_prompt(
                query,
                context_blocks,
                "deep" if double_check else "normal",
                intent=intent,
            )
            if double_check:
                plan = await generate_plan_async(
                    self, query, context_blocks, intent=intent
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
            if session_id and not history and self.conv_handler.history(session_id):
                logger.warning(
                    "\u26a0\ufe0f Possível quebra de contexto entre turnos – revisar prompt ou histórico.",
                    session=session_id,
                )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_CONTEXT},
                *history,
                {"role": "user", "content": prompt},
            ]
            result = await generate_final_async(
                self, query, context_blocks, plan if double_check else "", history, intent
            )
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

    async def generate_response_stream(
        self, query: str, session_id: str = "default"
    ) -> AsyncGenerator[str, None]:
        intent = detect_intent(query)
        from .prompt_engine import (
            build_dynamic_prompt_async,
            gather_context_async,
            SYSTEM_PROMPT_CONTEXT,
        )

        history = self._build_history_messages(session_id) if session_id else []
        context_blocks, suggestions = await gather_context_async(self, query)
        prompt = await build_dynamic_prompt_async(
            query, context_blocks, "normal", intent=intent
        )
        if suggestions:
            prompt += f"\nSugestao relacionada: {suggestions[0]['content'][:80]}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_CONTEXT},
            *history,
            {"role": "user", "content": prompt},
        ]
        collected = []
        async for token in self.ai_model.safe_api_stream(
            messages, config.MAX_CONTEXT_LENGTH
        ):
            collected.append(token)
            yield token
        if session_id:
            self.conv_handler.append(session_id, "user", query)
            self.conv_handler.append(session_id, "assistant", "".join(collected))

    @staticmethod
    def _split_plan_response(text: str) -> tuple[str, str]:
        """Split the AI response into plan and final answer.

        Accept minor variations like spaces around the marker.
        """
        m = re.search(r"===\s*RESPOSTA\s*===", text, re.IGNORECASE)
        if m:
            plan = text[: m.start()]
            resp = text[m.end() :]
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

            intent = detect_intent(query)
            from .prompt_engine import (
                build_dynamic_prompt,
                gather_context_async,
                generate_plan_async,
                generate_final_async,
            )

            context_blocks, suggestions = await gather_context_async(self, query)
            self.last_context = {
                "history": (
                    self._build_history_messages(session_id) if session_id else []
                ),
                "context_blocks": context_blocks,
            }
            prompt = build_dynamic_prompt(query, context_blocks, "deep", intent=intent)
            if suggestions:
                prompt += f"\nSugestao relacionada: {suggestions[0]['content'][:80]}"

            history = self._build_history_messages(session_id) if session_id else []
            plan = await generate_plan_async(self, query, context_blocks, intent=intent)
            resp = await generate_final_async(
                self, query, context_blocks, plan, history, intent
            )
            extra_plan, clean_resp = self._split_plan_response(resp)
            if extra_plan:
                plan = f"{plan}\n{extra_plan}".strip()
            resp = clean_resp
            if session_id:
                self.conv_handler.append(session_id, "user", query)
                self.conv_handler.append(session_id, "assistant", resp)
            else:
                logger.info("multi_turn_fallback", session=session_id)
            trace = f"\ud83d\udca1 Detalhes t\u00e9cnicos: A IA consultou {len(context_blocks['memories'])} m\u00e9morias anteriores, analisou depend\u00eancias e gerou a resposta abaixo com base no padr\u00e3o simb\u00f3lico aprendido."
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

    def get_session_context(
        self, session_id: str = "default", n: int = 3
    ) -> Dict[str, Any]:
        """Return preview of conversation and activated memories."""
        try:
            history = self.conv_handler.last(session_id, n * 2)
            preview = [f"{m['role']}: {m['content']}" for m in history]
            last_user = next(
                (m["content"] for m in reversed(history) if m["role"] == "user"), ""
            )
            symbolic = []
            warnings: List[str] = []
            try:
                symbolic = [
                    f"{m.get('metadata', {}).get('tag', '')}: {m['content']}"
                    for m in self.memory.search(
                        last_user, memory_type="dialog_summary", top_k=n
                    )
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
        watchers = list(getattr(self, "watchers", {}).items())
        for name, task in watchers:
            task.cancel()
        for name, task in watchers:
            try:
                await asyncio.wait_for(task, timeout=2)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning("Timeout ao encerrar watcher", watcher=name)
            except Exception as e:
                logger.error("Erro ao finalizar watcher", watcher=name, error=str(e))
            finally:
                if hasattr(self, "watchers"):
                    self.watchers.pop(name, None)
                if hasattr(self, "background_tasks"):
                    self.background_tasks.pop(name, None)

        tasks = list(getattr(self, "background_tasks", {}).items())
        for name, task in tasks:
            task.cancel()
        if tasks:
            results = []
            for name, task in tasks:
                try:
                    await asyncio.wait_for(task, timeout=5)
                    results.append(None)
                except asyncio.CancelledError:
                    results.append(None)
                except asyncio.TimeoutError:
                    logger.warning("Timeout ao encerrar task", task=name)
                    results.append(None)
                except Exception as e:
                    results.append(e)
            for (name, _), result in zip(tasks, results):
                if isinstance(result, Exception) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    logger.error("Erro ao finalizar task", task=name, error=str(result))
                if hasattr(self, "background_tasks"):
                    self.background_tasks.pop(name, None)
                if hasattr(self, "watchers"):
                    self.watchers.pop(name, None)
        if hasattr(self, "memory") and hasattr(self.memory, "close"):
            self.memory.close()
        await persist_errors()
        if hasattr(self, "ai_model") and hasattr(self.ai_model, "shutdown"):
            await self.ai_model.shutdown()
        logger.info("🛑 DevAI finalizado com limpeza simbólica de recursos.")
