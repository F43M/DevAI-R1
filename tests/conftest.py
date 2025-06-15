import os
import sys
import types

# Run HuggingFace libraries in offline mode during tests to avoid network calls
os.environ.setdefault("HUGGINGFACE_HUB_OFFLINE", "1")

try:  # prefer real library when available
    import networkx  # type: ignore  # noqa: F401
except Exception:

    class DiGraph:
        def __init__(self):
            self._adj = {}

        def add_node(self, n):
            self._adj.setdefault(n, set())

        def add_edge(self, u, v):
            self.add_node(u)
            self.add_node(v)
            self._adj[u].add(v)

        @property
        def nodes(self):
            return list(self._adj.keys())

        def edges(self):
            return [(u, v) for u, vs in self._adj.items() for v in vs]

        def successors(self, n):
            return list(self._adj.get(n, []))

        def number_of_edges(self):
            return sum(len(v) for v in self._adj.values())

    def descendants(graph, node):
        seen = set()
        stack = list(graph._adj.get(node, []))
        while stack:
            cur = stack.pop()
            if cur not in seen:
                seen.add(cur)
                stack.extend(graph._adj.get(cur, []))
        return seen

    sys.modules["networkx"] = types.SimpleNamespace(
        DiGraph=DiGraph, descendants=descendants
    )

# Provide lightweight Rich stubs when the library is unavailable
try:  # pragma: no cover - prefer real library when present
    from rich.console import Console  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - fallback for test env
    console_mod = types.ModuleType("rich.console")

    class Console:
        def __init__(self, *a, **k):
            pass

        def print(self, *a, **k):
            print(*a)

    console_mod.Console = Console

    panel_mod = types.ModuleType("rich.panel")

    class Panel(str):
        pass

    panel_mod.Panel = Panel

    syntax_mod = types.ModuleType("rich.syntax")

    class Syntax(str):
        def __new__(cls, code: str, *_a, **_k):
            return str.__new__(cls, code)

    syntax_mod.Syntax = Syntax

    text_mod = types.ModuleType("rich.text")

    class Text(str):
        def __new__(cls, text: str = "", *_, style=None, **__):
            return str.__new__(cls, text)

    text_mod.Text = Text

    table_mod = types.ModuleType("rich.table")

    class Table:
        def __init__(self, *a, **k):
            pass

        @classmethod
        def grid(cls, *a, **k):
            return cls()

        def add_column(self, *a, **k):
            pass

        def add_row(self, *a, **k):
            pass

    table_mod.Table = Table

    rich_mod = types.ModuleType("rich")
    rich_mod.console = console_mod
    rich_mod.panel = panel_mod
    rich_mod.syntax = syntax_mod
    rich_mod.text = text_mod
    rich_mod.table = table_mod

    sys.modules.setdefault("rich", rich_mod)
    sys.modules.setdefault("rich.console", console_mod)
    sys.modules.setdefault("rich.panel", panel_mod)
    sys.modules.setdefault("rich.syntax", syntax_mod)
    sys.modules.setdefault("rich.text", text_mod)
    sys.modules.setdefault("rich.table", table_mod)

try:
    from fastapi import FastAPI  # type: ignore  # noqa: F401
    from fastapi.staticfiles import StaticFiles  # noqa: F401
    from fastapi.responses import StreamingResponse  # noqa: F401
except Exception:

    class FastAPI:
        def __init__(self, *a, **k):
            pass

        def get(self, path):
            def decorator(fn):
                return fn

            return decorator

        def post(self, path):
            def decorator(fn):
                return fn

            return decorator

        def mount(self, *a, **k):
            pass

    class StaticFiles:
        def __init__(self, *a, **k):
            pass

    class StreamingResponse:
        def __init__(self, content, media_type=None):
            self.content = content
            self.media_type = media_type

    fastapi_stub = types.ModuleType("fastapi")
    staticfiles_module = types.ModuleType("fastapi.staticfiles")
    staticfiles_module.StaticFiles = StaticFiles
    responses_module = types.ModuleType("fastapi.responses")
    responses_module.StreamingResponse = StreamingResponse
    fastapi_stub.FastAPI = FastAPI
    fastapi_stub.staticfiles = staticfiles_module
    fastapi_stub.responses = responses_module
    sys.modules["fastapi"] = fastapi_stub
    sys.modules["fastapi.staticfiles"] = staticfiles_module
    sys.modules["fastapi.responses"] = responses_module

try:
    import uvicorn  # type: ignore  # noqa: F401
except Exception:

    class Config:
        def __init__(self, *a, **k):
            pass

    class Server:
        def __init__(self, config):
            self.config = config

        async def serve(self):
            pass

    def run(app, *a, **k):
        return True

    uvicorn_stub = types.ModuleType("uvicorn")
    uvicorn_stub.Config = Config
    uvicorn_stub.Server = Server
    uvicorn_stub.run = run
    sys.modules["uvicorn"] = uvicorn_stub

try:
    from aiohttp import ClientSession, ClientTimeout  # type: ignore  # noqa: F401
except Exception:

    class ClientSession:
        async def post(self, *a, **k):
            class Resp:
                async def json(self):
                    return {}

                async def text(self):
                    return ""

            return Resp()

        async def close(self):
            pass

    class ClientTimeout:
        def __init__(self, *a, **k):
            pass

    aiohttp_stub = types.ModuleType("aiohttp")
    aiohttp_stub.ClientSession = ClientSession
    aiohttp_stub.ClientTimeout = ClientTimeout
    aiohttp_stub.ClientError = Exception
    aiohttp_stub.ClientConnectionError = Exception
    sys.modules["aiohttp"] = aiohttp_stub

try:
    import aiofiles  # type: ignore  # noqa: F401
    import aiofiles.os  # type: ignore  # noqa: F401
except Exception:
    import os

    class _AIOFile:
        def __init__(self, path, mode="r"):
            self._f = open(path, mode)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            self._f.close()

        async def read(self):
            return self._f.read()

        async def readlines(self):
            return self._f.readlines()

        async def write(self, data):
            self._f.write(data)

    aiofiles_stub = types.ModuleType("aiofiles")
    aiofiles_stub.open = lambda path, mode="r": _AIOFile(path, mode)
    aiofiles_os = types.ModuleType("aiofiles.os")

    async def makedirs(path, exist_ok=False):
        os.makedirs(path, exist_ok=exist_ok)

    aiofiles_os.makedirs = makedirs
    aiofiles_stub.os = aiofiles_os
    sys.modules["aiofiles"] = aiofiles_stub
    sys.modules["aiofiles.os"] = aiofiles_os

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore  # noqa: F401
except Exception:

    class TF:
        def fit_transform(self, docs):
            class Mat(list):
                @property
                def shape(self):
                    return (len(docs), 0)

            return Mat()

        def get_feature_names_out(self):
            return []

    sklearn_stub = types.ModuleType("sklearn")
    fe = types.ModuleType("sklearn.feature_extraction")
    text = types.ModuleType("sklearn.feature_extraction.text")
    text.TfidfVectorizer = TF
    fe.text = text
    sklearn_stub.feature_extraction = fe
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.feature_extraction"] = fe
    sys.modules["sklearn.feature_extraction.text"] = text
