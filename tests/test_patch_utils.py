import difflib
from pathlib import Path

import devai.patch_utils as patch_utils
import unidiff


def test_apply_patch_to_file(tmp_path, monkeypatch):
    file = tmp_path / "test.txt"
    file.write_text("old\n")
    diff = "".join(
        difflib.unified_diff(
            ["old\n"],
            ["new\n"],
            fromfile="a/test.txt",
            tofile="b/test.txt",
        )
    )

    orig = unidiff.PatchSet

    def wrapper(text):
        if hasattr(text, "read"):
            text = text.read()
        return orig(text)

    monkeypatch.setattr(unidiff, "PatchSet", wrapper)

    patch_utils.apply_patch_to_file(file, diff)
    assert file.read_text() == "new\n"
