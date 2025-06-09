from .config import config

_actions_suggest = {"patch", "shell", "edit", "delete", "create"}

def requires_approval(action: str) -> bool:
    """Return True if the given action requires confirmation."""
    mode = getattr(config, "APPROVAL_MODE", "suggest")
    if mode == "manual":
        return True
    if mode == "auto":
        return False
    return action in _actions_suggest
