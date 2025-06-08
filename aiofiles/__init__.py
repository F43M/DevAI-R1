from . import os
print("\u26A0\uFE0F Dependência ausente: aiofiles. Reverter para stub temporário.")
import asyncio
import builtins


class open:
    def __init__(self, file, mode="r", *args, **kwargs):
        self.file = file
        self.mode = mode
        self.args = args
        self.kwargs = kwargs
        self._f = None

    async def __aenter__(self):
        loop = asyncio.get_event_loop()
        self._f = await loop.run_in_executor(
            None, builtins.open, self.file, self.mode, *self.args, **self.kwargs
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._f:
            await asyncio.get_event_loop().run_in_executor(None, self._f.close)

    async def read(self):
        return await asyncio.get_event_loop().run_in_executor(None, self._f.read)

    async def write(self, data):
        await asyncio.get_event_loop().run_in_executor(None, self._f.write, data)
