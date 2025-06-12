import importlib.util

from devai import dependency_check


class DummyLogger:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def error(self, msg, *args, **kwargs):
        self.errors.append((msg, args, kwargs))


def test_check_dependencies_no_error_when_all_present(monkeypatch):
    logger = DummyLogger()
    monkeypatch.setattr(dependency_check, "logger", logger)
    dependency_check.check_dependencies()
    assert logger.errors == []


def test_check_dependencies_reports_missing(monkeypatch):
    logger = DummyLogger()
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name):
        if name == "fastapi":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(dependency_check, "logger", logger)
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    dependency_check.check_dependencies()
    assert logger.errors
    assert "fastapi" in logger.errors[0][1][0]
