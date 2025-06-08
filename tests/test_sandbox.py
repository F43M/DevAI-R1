import pytest
from devai import sandbox


def test_run_not_implemented():
    sb = sandbox.Sandbox()
    with pytest.raises(NotImplementedError):
        sb.run(["echo", "hi"])
