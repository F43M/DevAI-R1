from devai import dependency_check


class DummyLogger:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def error(self, msg, *args, **kwargs):
        self.errors.append((msg, args, kwargs))


def test_check_dependencies_warns_for_stubs(monkeypatch):
    logger = DummyLogger()
    monkeypatch.setattr(dependency_check, "logger", logger)
    dependency_check.check_dependencies()
    assert any("simplificada" in w for w in logger.warnings)
    assert not logger.errors
