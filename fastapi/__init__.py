print("\u26A0\uFE0F Dependência ausente: fastapi. Reverter para stub temporário.")

class FastAPI:
    def __init__(self, *a, **k):
        self.routes = {}

    def _add(self, method, path, fn):
        self.routes.setdefault(path, {})[method] = fn

    def post(self, path, *a, **k):
        def decorator(fn):
            self._add("POST", path, fn)
            return fn
        return decorator

    def get(self, path, *a, **k):
        def decorator(fn):
            self._add("GET", path, fn)
            return fn
        return decorator

    def mount(self, *a, **k):
        pass

from .staticfiles import StaticFiles
