from . import os
class open:
    def __init__(self, *a, **k):
        pass
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        pass
    async def read(self):
        return ""
    async def write(self, data):
        pass
