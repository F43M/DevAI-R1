import asyncio
import os


async def makedirs(path, exist_ok=False):
    await asyncio.get_event_loop().run_in_executor(
        None, os.makedirs, path, exist_ok
    )
