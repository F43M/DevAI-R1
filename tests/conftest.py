import sys
import types

if 'networkx' not in sys.modules:
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
    sys.modules['networkx'] = types.SimpleNamespace(DiGraph=DiGraph, descendants=descendants)

if 'fastapi' not in sys.modules:
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

    fastapi_stub = types.ModuleType('fastapi')
    staticfiles_module = types.ModuleType('fastapi.staticfiles')
    staticfiles_module.StaticFiles = StaticFiles
    responses_module = types.ModuleType('fastapi.responses')
    responses_module.StreamingResponse = StreamingResponse
    fastapi_stub.FastAPI = FastAPI
    fastapi_stub.staticfiles = staticfiles_module
    fastapi_stub.responses = responses_module
    sys.modules['fastapi'] = fastapi_stub
    sys.modules['fastapi.staticfiles'] = staticfiles_module
    sys.modules['fastapi.responses'] = responses_module

if 'uvicorn' not in sys.modules:
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

    uvicorn_stub = types.ModuleType('uvicorn')
    uvicorn_stub.Config = Config
    uvicorn_stub.Server = Server
    uvicorn_stub.run = run
    sys.modules['uvicorn'] = uvicorn_stub

if 'aiohttp' not in sys.modules:
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

    aiohttp_stub = types.ModuleType('aiohttp')
    aiohttp_stub.ClientSession = ClientSession
    aiohttp_stub.ClientTimeout = ClientTimeout
    aiohttp_stub.ClientError = Exception
    aiohttp_stub.ClientConnectionError = Exception
    sys.modules['aiohttp'] = aiohttp_stub

if 'aiofiles' not in sys.modules:
    import os
    class _AIOFile:
        def __init__(self, path, mode='r'):
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

    aiofiles_stub = types.ModuleType('aiofiles')
    aiofiles_stub.open = lambda path, mode='r': _AIOFile(path, mode)
    aiofiles_os = types.ModuleType('aiofiles.os')
    async def makedirs(path, exist_ok=False):
        os.makedirs(path, exist_ok=exist_ok)
    aiofiles_os.makedirs = makedirs
    aiofiles_stub.os = aiofiles_os
    sys.modules['aiofiles'] = aiofiles_stub
    sys.modules['aiofiles.os'] = aiofiles_os

if 'sklearn' not in sys.modules:
    class TF:
        def fit_transform(self, docs):
            class Mat(list):
                @property
                def shape(self):
                    return (len(docs), 0)
            return Mat()
        def get_feature_names_out(self):
            return []
    sklearn_stub = types.ModuleType('sklearn')
    fe = types.ModuleType('sklearn.feature_extraction')
    text = types.ModuleType('sklearn.feature_extraction.text')
    text.TfidfVectorizer = TF
    fe.text = text
    sklearn_stub.feature_extraction = fe
    sys.modules['sklearn'] = sklearn_stub
    sys.modules['sklearn.feature_extraction'] = fe
    sys.modules['sklearn.feature_extraction.text'] = text
