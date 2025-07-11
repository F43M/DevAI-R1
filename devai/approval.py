from .config import config
from .decision_log import is_remembered
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4
from .notifier import Notifier

_approval_event = asyncio.Event()
_approval_future: asyncio.Future | None = None
_approval_message = ""
_approval_details: str | None = None
_approval_token = ""

# Remaining actions allowed without manual approval
auto_approve_remaining = 0

# Timestamp until which approvals are automatic
auto_approve_until: datetime | None = None

WRITE_ACTIONS = {"patch", "edit", "create", "delete"}

# Shell execution categories
SHELL_ACTIONS = {"shell", "shell_safe"}

# Commands that are read-only and safe to run automatically
SAFE_ACTIONS = {"shell_safe"}


def match_glob(pattern: str, target: str) -> bool:
    """Return True if ``target`` matches the glob ``pattern``."""
    try:
        return Path(target).match(pattern)
    except Exception:
        return False


def requires_approval(action: str, path: str | None = None) -> bool:
    """Return True if the given action requires confirmation."""
    global auto_approve_remaining, auto_approve_until
    now = datetime.now()
    if auto_approve_until and now < auto_approve_until:
        return False
    if auto_approve_remaining > 0:
        auto_approve_remaining -= 1
        return False
    if path and is_remembered(action, path):
        return False

    for rule in getattr(config, "AUTO_APPROVAL_RULES", []):
        try:
            if rule.get("action") == action and path and match_glob(
                rule.get("path", ""), path
            ):
                return not rule.get("approve", False)
        except Exception:
            continue

    # Actions explicitly marked as safe never require confirmation
    if action in SAFE_ACTIONS:
        return False

    mode = getattr(config, "APPROVAL_MODE", "suggest").lower()
    if mode == "full_auto":
        return False
    if mode == "auto_edit":
        return action == "shell"
    return action in WRITE_ACTIONS or action in SHELL_ACTIONS


async def request_approval(message: str, details: str | None = None) -> bool:
    """Trigger approval flow in web mode and wait for result."""
    global _approval_future, _approval_message, _approval_details, _approval_token
    if _approval_future is not None:
        raise RuntimeError("Another approval in progress")
    _approval_future = asyncio.get_running_loop().create_future()
    _approval_message = message
    _approval_details = details
    _approval_token = uuid4().hex
    notifier = Notifier()
    if notifier.enabled:
        base_url = f"http://localhost:{config.API_PORT}"
        approve = f"{base_url}/approval_request?token={_approval_token}&approved=true"
        reject = f"{base_url}/approval_request?token={_approval_token}&approved=false"
        body = f"{message}\nAprovar: {approve}\nRejeitar: {reject}"
        notifier.send("Confirmação necessária", body, details=details)
    _approval_event.set()
    result = await _approval_future
    _approval_future = None
    _approval_details = None
    return bool(result)


async def wait_for_request() -> dict:
    """Wait until a request is available for the frontend."""
    await _approval_event.wait()
    _approval_event.clear()
    return {
        "message": _approval_message,
        "token": _approval_token,
        "details": _approval_details,
    }


def resolve_request(approved: bool) -> None:
    """Resolve the current pending approval request."""
    global _approval_future, _approval_token, _approval_details
    if _approval_future and not _approval_future.done():
        _approval_future.set_result(bool(approved))
    _approval_token = ""
    _approval_details = None


def verify_token(token: str) -> bool:
    """Return True if token matches the current approval token."""
    return bool(token) and token == _approval_token
