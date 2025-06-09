from .config import config
from .decision_log import is_remembered

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
