import textwrap
from devai.post_processor import is_valid_python, fix_code


def test_is_valid_python():
    assert is_valid_python("print(1)")
    assert not is_valid_python("print(")


def test_fix_code_adds_missing_colon():
    bad_code = textwrap.dedent(
        """
    def foo()
        return 1
    """
    )
    fixed = fix_code(bad_code)
    assert is_valid_python(fixed)
    assert "def foo():" in fixed
