import ast
import asyncio
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
import aiofiles.os
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import logger
from .memory import MemoryManager


class CodeAnalyzer:
    """Parse source files and build dependency graphs."""

    def __init__(self, code_root: str, memory: MemoryManager, history=None):
        self.code_root = Path(code_root)
        self.memory = memory
        self.history = history
        self.code_chunks: Dict[str, Dict] = {}
        self.code_graph = nx.DiGraph()
        self.vectorizer = TfidfVectorizer()
        self.file_hashes = {}
        self.external_dependencies = set()
        self.learned_rules: Dict[str, str] = {}
        self.last_analysis_time = datetime.now()

    async def deep_scan_app(self):
        logger.info("Iniciando varredura profunda do aplicativo")
        await self.scan_app()
        await self._build_semantic_relations()
        await self._analyze_patterns()
        logger.info(
            "Varredura profunda concluída",
            functions=len(self.code_chunks),
            relations=self.code_graph.number_of_edges(),
        )

    async def scan_app(self):
        tasks = []
        patterns = [
            "*.py",
            "*.js",
            "*.ts",
            "*.cpp",
            "*.hpp",
            "*.html",
            "*.java",
            "*.cs",
            "*.rb",
            "*.php",
        ]
        for pat in patterns:
            for file_path in self.code_root.rglob(pat):
                if file_path.is_file():
                    tasks.append(self.parse_file(file_path))
        await asyncio.gather(*tasks)
        logger.info("Aplicativo escaneado", files_processed=len(tasks))

    async def parse_file(self, file_path: Path):
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = await f.read()
            file_hash = hashlib.md5(content.encode()).hexdigest()
            if str(file_path) in self.file_hashes and self.file_hashes[str(file_path)] == file_hash:
                return
            self.file_hashes[str(file_path)] = file_hash
            if file_path.suffix == ".py":
                await self._parse_python(content, file_path, file_hash)
            else:
                await self._parse_generic(content, file_path, file_hash)
        except SyntaxError as e:
            logger.error("Erro de sintaxe", file=str(file_path), error=str(e))
        except Exception as e:
            logger.error("Erro ao analisar arquivo", file=str(file_path), error=str(e))

    async def _parse_python(self, content: str, file_path: Path, file_hash: str):
        try:
            tree = ast.parse(content)
            chunks = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    chunk = {
                        "name": node.name,
                        "type": type(node).__name__,
                        "code": ast.unparse(node),
                        "file": str(file_path),
                        "hash": file_hash,
                        "last_modified": datetime.now().isoformat(),
                        "dependencies": self._get_dependencies(node),
                        "calls": self._get_function_calls(node),
                        "external_deps": self._get_external_dependencies(node),
                        "docstring": ast.get_docstring(node) or "",
                        "line_start": node.lineno,
                        "line_end": node.end_lineno,
                        "complexity": self._compute_complexity(node),
                    }
                    self.code_graph.add_node(node.name, **chunk)
                    for dep in chunk["dependencies"]:
                        self.code_graph.add_edge(node.name, dep, type="dependency")
                    for call in chunk["calls"]:
                        self.code_graph.add_edge(
                            node.name,
                            call["function"],
                            type="call",
                            line=call["line"],
                        )
                    self.external_dependencies.update(chunk["external_deps"])
                    chunks.append(chunk)
                    if node.name in self.code_chunks and self.code_chunks[node.name]["hash"] != file_hash:
                        logger.warning(
                            "Código modificado",
                            chunk=node.name,
                            old_hash=self.code_chunks[node.name]["hash"],
                            new_hash=file_hash,
                        )
            self.code_chunks.update({c["name"]: c for c in chunks})
            for chunk in chunks:
                from .symbolic_memory_tagger import tag_memory_entry
                base_tags = ["code", chunk['type'].lower(), os.path.basename(chunk['file'])]
                memory_entry = {
                    "type": "code_chunk",
                    "content": f"{chunk['type']} {chunk['name']} em {chunk['file']}",
                    "metadata": chunk,
                    "tags": base_tags + tag_memory_entry(chunk),
                }
                self.memory.save(memory_entry)
        except Exception as e:
            logger.error("Erro ao analisar arquivo", file=str(file_path), error=str(e))

    async def _parse_generic(self, content: str, file_path: Path, file_hash: str):
        chunks = []
        pattern = None
        ftype = file_path.suffix.lower().lstrip(".")
        if ftype in {"js", "ts"}:
            pattern = re.compile(r"function\s+(\w+)\s*\(")
        elif ftype in {"cpp", "hpp"}:
            pattern = re.compile(r"[\w:]+\s+(\w+)\s*\([^)]*\)\s*{", re.MULTILINE)
        elif ftype == "java":
            pattern = re.compile(
                r"(?:public|protected|private|static|final|\s)+\s+[\w<>,\[\]]+\s+(\w+)\s*\(",
                re.MULTILINE,
            )
        elif ftype == "cs":
            pattern = re.compile(
                r"(?:public|protected|private|internal|static|virtual|override|async|\s)+\s+[\w<>,\[\]]+\s+(\w+)\s*\(",
                re.MULTILINE,
            )
        elif ftype == "rb":
            pattern = re.compile(r"def\s+(\w+)")
        elif ftype == "php":
            pattern = re.compile(r"function\s+(\w+)\s*\(")
        if pattern:
            for m in pattern.finditer(content):
                name = m.group(1)
                chunk = {
                    "name": name,
                    "type": ftype + "_func",
                    "code": "",
                    "file": str(file_path),
                    "hash": file_hash,
                    "last_modified": datetime.now().isoformat(),
                    "dependencies": [],
                    "calls": [],
                    "external_deps": [],
                    "docstring": "",
                    "line_start": content[: m.start()].count("\n") + 1,
                    "line_end": content[: m.end()].count("\n") + 1,
                    "complexity": 1,
                }
                chunks.append(chunk)
                self.code_graph.add_node(name, **chunk)
        else:
            name = file_path.stem
            chunk = {
                "name": name,
                "type": ftype + "_file",
                "code": content,
                "file": str(file_path),
                "hash": file_hash,
                "last_modified": datetime.now().isoformat(),
                "dependencies": [],
                "calls": [],
                "external_deps": [],
                "docstring": "",
                "line_start": 1,
                "line_end": content.count("\n") + 1,
                "complexity": 1,
            }
            chunks.append(chunk)
            self.code_graph.add_node(name, **chunk)

        self.code_chunks.update({c["name"]: c for c in chunks})
        from .symbolic_memory_tagger import tag_memory_entry
        for chunk in chunks:
            base_tags = ["code", ftype, os.path.basename(chunk['file'])]
            self.memory.save({
                "type": "code_chunk",
                "content": f"{chunk['type']} {chunk['name']} em {chunk['file']}",
                "metadata": chunk,
                "tags": base_tags + tag_memory_entry(chunk),
            })

    def _get_dependencies(self, node):
        deps = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                deps.add(child.id)
            elif isinstance(child, ast.Attribute):
                deps.add(child.attr)
        return list(deps)

    def _get_function_calls(self, node):
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name):
                    calls.append({"function": func.id, "line": child.lineno})
                elif isinstance(func, ast.Attribute):
                    calls.append({"function": func.attr, "line": child.lineno})
        return calls

    def _get_external_dependencies(self, node):
        deps = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                for n in child.names:
                    deps.add(n.name.split(".")[0])
            elif isinstance(child, ast.ImportFrom) and child.module:
                deps.add(child.module.split(".")[0])
        return list(deps)

    def _compute_complexity(self, node) -> int:
        """Rudimentary cyclomatic complexity estimation."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Match, ast.With, ast.Try)):
                complexity += 1
        return complexity

    async def _build_semantic_relations(self):
        logger.info("Construindo relações semânticas")
        cursor = self.memory.conn.cursor()
        cursor.execute(
            """
            SELECT id, metadata FROM memory
            WHERE type = 'code_chunk'
            AND json_extract(metadata, '$.name') IS NOT NULL
            """
        )
        code_memories = {json.loads(row[1])["name"]: row[0] for row in cursor.fetchall()}
        for source, target, data in self.code_graph.edges(data=True):
            if source in code_memories and target in code_memories:
                rel = "calls" if data.get("type") == "call" else "depends_on"
                strength = 0.5 if rel == "calls" else 0.9
                self.memory.add_semantic_relation(
                    code_memories[source],
                    code_memories[target],
                    rel,
                    strength=strength,
                )
        all_code = [(mid, json.loads(meta)["name"]) for name, mid in code_memories.items()]
        for i, (mid1, name1) in enumerate(all_code):
            for j, (mid2, name2) in enumerate(all_code[i + 1 :], i + 1):
                if self._name_similarity(name1, name2) > 0.7:
                    self.memory.add_semantic_relation(mid1, mid2, "similar_name", strength=0.7)
        logger.info("Relações semânticas construídas", relations=len(self.code_graph.edges()))

    def _name_similarity(self, name1: str, name2: str) -> float:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()

    async def _analyze_patterns(self):
        logger.info("Analisando padrões para regras aprendidas")
        for chunk in self.code_chunks.values():
            try:
                tree = ast.parse(chunk["code"])
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and len(node.args.args) > 5:
                        rule_name = "funcao_com_muitos_parametros"
                        condition = "len(chunk['code'].split('def ')[1].split('(')[1].split(')')[0].split(',')) > 5"
                        if rule_name not in self.learned_rules:
                            self.learned_rules[rule_name] = condition
                            self.memory.save(
                                {
                                    "type": "learned_rule",
                                    "content": f"Função com muitos parâmetros detectada: {chunk['name']}",
                                    "metadata": {"rule": rule_name, "condition": condition, "example": chunk["name"]},
                                    "tags": ["rule", "code_smell"],
                                }
                            )
            except Exception as e:
                logger.error("Erro ao analisar padrão", chunk=chunk["name"], error=str(e))
        logger.info("Análise de padrões concluída", rules=len(self.learned_rules))

    def get_code_graph(self) -> Dict:
        return {
            "nodes": [{"id": n, **data} for n, data in self.code_graph.nodes(data=True)],
            "links": [{"source": u, "target": v} for u, v in self.code_graph.edges()],
        }

    def graph_summary(self, limit: int = 20) -> str:
        """Return a textual summary of the dependency graph."""
        lines = []
        for node in list(self.code_graph.nodes)[:limit]:
            deps = list(self.code_graph.successors(node))
            if deps:
                lines.append(f"Função {node} chama {', '.join(deps)}")
        return "\n".join(lines)

    async def watch_app_directory(self, interval: int = 5):
        logger.info("Iniciando monitoramento da pasta de código", path=str(self.code_root))
        while True:
            try:
                await self.scan_app()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error("Erro no monitoramento da pasta", error=str(e))
                await asyncio.sleep(interval)

    async def list_dir(self, subpath: str = "") -> List[str]:
        """List files and folders inside CODE_ROOT."""
        target = (self.code_root / subpath).resolve()
        if not str(target).startswith(str(self.code_root.resolve())) or not target.exists():
            return []
        return [
            f"{p.relative_to(self.code_root)}{'/' if p.is_dir() else ''}"
            for p in sorted(target.iterdir())
        ]

    async def read_lines(self, file_path: str, start: int = 1, end: Optional[int] = None) -> List[str]:
        """Return selected lines from a file inside CODE_ROOT."""
        path = (self.code_root / file_path).resolve()
        if not path.is_file() or not str(path).startswith(str(self.code_root.resolve())):
            return []
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        lines = content.splitlines()
        start = max(1, start)
        end = len(lines) if end is None else min(end, len(lines))
        return [lines[i - 1].rstrip("\n") for i in range(start, end + 1)]

    async def edit_line(self, file_path: str, line_no: int, new_content: str) -> bool:
        """Edit a specific line of a file and reparse it."""
        path = (self.code_root / file_path).resolve()
        if not path.is_file() or not str(path).startswith(str(self.code_root.resolve())):
            return False
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not 1 <= line_no <= len(lines):
            return False
        old = lines[line_no - 1].rstrip("\n")
        lines[line_no - 1] = new_content + "\n"
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        if self.history:
            self.history.record(str(path.relative_to(self.code_root)), "edit", [old], [new_content])
        await self.parse_file(path)
        return True

    async def create_file(self, file_path: str, content: str = "") -> bool:
        path = (self.code_root / file_path).resolve()
        if not str(path).startswith(str(self.code_root.resolve())):
            return False
        if path.exists():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        if self.history:
            self.history.record(str(path.relative_to(self.code_root)), "create", new=content.splitlines())
        if path.suffix == ".py":
            await self.parse_file(path)
        return True

    async def delete_file(self, file_path: str) -> bool:
        path = (self.code_root / file_path).resolve()
        if not path.is_file() or not str(path).startswith(str(self.code_root.resolve())):
            return False
        with open(path, "r", encoding="utf-8") as f:
            old = f.read().splitlines()
        path.unlink()
        if self.history:
            self.history.record(str(path.relative_to(self.code_root)), "delete", old=old)
        self.code_chunks = {k: v for k, v in self.code_chunks.items() if v.get("file") != str(path)}
        return True

    async def create_directory(self, dir_path: str) -> bool:
        path = (self.code_root / dir_path).resolve()
        if not str(path).startswith(str(self.code_root.resolve())):
            return False
        if path.exists():
            return False
        path.mkdir(parents=True, exist_ok=False)
        if self.history:
            self.history.record(str(path.relative_to(self.code_root)), "mkdir")
        return True

    async def delete_directory(self, dir_path: str) -> bool:
        path = (self.code_root / dir_path).resolve()
        if not path.is_dir() or not str(path).startswith(str(self.code_root.resolve())):
            return False
        import shutil
        shutil.rmtree(path)
        if self.history:
            self.history.record(str(path.relative_to(self.code_root)), "rmdir")
        return True

    async def get_history(self, file_path: str):
        if self.history:
            return self.history.history(file_path)
        return []
