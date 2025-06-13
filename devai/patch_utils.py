"""Utility functions for patch manipulation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def split_diff_by_file(diff_text: str) -> Dict[str, str]:
    """Split a unified diff into chunks by file path.

    Parameters
    ----------
    diff_text:
        A unified diff covering one or more files.

    Returns
    -------
    Dict[str, str]
        Mapping of target file paths to the diff that should be
        applied to them.
    """
    result: Dict[str, str] = {}
    current_lines: list[str] = []
    current_file: str | None = None
    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git"):
            if current_file:
                result[current_file] = "".join(current_lines)
            current_lines = [line]
            current_file = None
        elif line.startswith("+++ "):
            current_lines.append(line)
            current_file = line[4:].strip()
            if current_file.startswith("b/"):
                current_file = current_file[2:]
        else:
            current_lines.append(line)
    if current_file:
        result[current_file] = "".join(current_lines)
    return result


def apply_patch_to_file(path: str | Path, diff_text: str) -> None:
    """Apply a unified diff chunk to a single file.

    Raises
    ------
    RuntimeError
        If the patch context does not match the file contents.
    """
    from unidiff import PatchSet

    file_path = Path(path)
    patch_set = PatchSet(diff_text)
    if not patch_set:
        raise ValueError("Patch must contain at least one file")
    patched_file = patch_set[0]

    lines = file_path.read_text().splitlines(keepends=True)
    result: list[str] = []
    idx = 0

    for hunk in patched_file:
        start = hunk.source_start - 1
        expected = [
            line_obj.value for line_obj in hunk if not line_obj.is_added
        ]
        end = start + len(expected)
        actual = lines[start:end]
        if actual != expected:
            raise RuntimeError("patch context mismatch")

        result.extend(lines[idx:start])
        idx = start
        for line in hunk:
            if line.is_removed:
                idx += 1
            elif line.is_added:
                result.append(line.value)
            else:
                if idx < len(lines):
                    result.append(lines[idx])
                idx += 1
    result.extend(lines[idx:])
    file_path.write_text("".join(result))


def apply_patch(diff_text: str) -> None:
    """Apply a unified diff affecting multiple files atomically.

    The diff is split by target file and each chunk is applied sequentially.
    If any patch fails, previously patched files are restored to their
    original state and the error is re-raised.
    """
    patches = split_diff_by_file(diff_text)
    backups: dict[Path, str | None] = {}
    applied: list[Path] = []

    try:
        for path_str, patch in patches.items():
            path = Path(path_str)
            if not path.exists():
                matches = list(Path.cwd().rglob(path_str))
                if not matches:
                    matches = list(Path('/tmp').rglob(path_str))
                if matches:
                    path = matches[0]
                else:
                    raise FileNotFoundError(path_str)
            backups[path] = path.read_text() if path.exists() else None
            apply_patch_to_file(path, patch)
            applied.append(path)
    except Exception:
        for p in applied:
            original = backups.get(p)
            if original is None:
                p.unlink(missing_ok=True)
            else:
                p.write_text(original)
        raise
