from .config import config

WRITE_ACTIONS = {"patch", "edit", "create", "delete"}


def requires_approval(action: str) -> bool:
    """Return True if the given action requires confirmation."""
    mode = getattr(config, "APPROVAL_MODE", "suggest").lower()
    if mode == "full_auto":
        return False
    if mode == "auto_edit":
        return action == "shell"
    return action in WRITE_ACTIONS or action == "shell"
