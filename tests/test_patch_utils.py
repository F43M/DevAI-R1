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


import pytest


def test_apply_patch_context_mismatch(tmp_path, monkeypatch):
    file = tmp_path / "test.txt"
    file.write_text("other\n")
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

    with pytest.raises(RuntimeError):
        patch_utils.apply_patch_to_file(file, diff)


def test_apply_patch_multi_file(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("oldA\n")
    b.write_text("oldB\n")
    diff = (
        "diff --git a/a.txt b/a.txt\n"
        "--- a/a.txt\n"
        "+++ b/a.txt\n"
        "@@ -1 +1 @@\n"
        "-oldA\n"
        "+newA\n"
        "diff --git a/b.txt b/b.txt\n"
        "--- a/b.txt\n"
        "+++ b/b.txt\n"
        "@@ -1 +1 @@\n"
        "-oldB\n"
        "+newB\n"
    )
    patch_utils.apply_patch(diff)
    assert a.read_text() == "newA\n"
    assert b.read_text() == "newB\n"


def test_apply_patch_rollback(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("oldA\n")
    b.write_text("other\n")
    diff = (
        "diff --git a/a.txt b/a.txt\n"
        "--- a/a.txt\n"
        "+++ b/a.txt\n"
        "@@ -1 +1 @@\n"
        "-oldA\n"
        "+newA\n"
        "diff --git a/b.txt b/b.txt\n"
        "--- a/b.txt\n"
        "+++ b/b.txt\n"
        "@@ -1 +1 @@\n"
        "-oldB\n"
        "+newB\n"
    )
    with pytest.raises(RuntimeError):
        patch_utils.apply_patch(diff)
    assert a.read_text() == "oldA\n"
    assert b.read_text() == "other\n"
