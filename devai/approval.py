from .config import config
from .decision_log import is_remembered
import asyncio

_approval_event = asyncio.Event()
_approval_future: asyncio.Future | None = None
_approval_message = ""

WRITE_ACTIONS = {"patch", "edit", "create", "delete"}


def requires_approval(action: str, path: str | None = None) -> bool:
    """Return True if the given action requires confirmation."""
    if path and is_remembered(action, path):
        return False

    mode = getattr(config, "APPROVAL_MODE", "suggest").lower()
    if mode == "full_auto":
        return False
    if mode == "auto_edit":
        return action == "shell"
    return action in WRITE_ACTIONS or action == "shell"


async def request_approval(message: str) -> bool:
    """Trigger approval flow in web mode and wait for result."""
    global _approval_future, _approval_message
    if _approval_future is not None:
        raise RuntimeError("Another approval in progress")
    _approval_future = asyncio.get_running_loop().create_future()
    _approval_message = message
    _approval_event.set()
    result = await _approval_future
    _approval_future = None
    return bool(result)


async def wait_for_request() -> dict:
    """Wait until a request is available for the frontend."""
    await _approval_event.wait()
    _approval_event.clear()
    return {"message": _approval_message}


def resolve_request(approved: bool) -> None:
    """Resolve the current pending approval request."""
    global _approval_future
    if _approval_future and not _approval_future.done():
        _approval_future.set_result(bool(approved))
