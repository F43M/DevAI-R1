class FastAPI:
    def __init__(self, *a, **k):
        pass
    def post(self, *a, **k):
        def decorator(fn):
            return fn
        return decorator
    def get(self, *a, **k):
        def decorator(fn):
            return fn
        return decorator
    def mount(self, *a, **k):
        pass

from .staticfiles import StaticFiles
