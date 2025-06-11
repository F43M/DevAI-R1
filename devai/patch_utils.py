"""Utility functions for patch manipulation."""
from __future__ import annotations

from io import StringIO
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
        Mapping of target file paths to the diff that should be applied to them.
    """
    from unidiff import PatchSet

    patches = PatchSet(StringIO(diff_text))
    result: Dict[str, str] = {}
    for patched_file in patches:
        # ``patched_file.path`` gives the target path relative to repository root
        result[patched_file.path] = str(patched_file)
    return result


def apply_patch_to_file(path: str | Path, diff_text: str) -> None:
    """Apply a unified diff chunk to a single file.

    Parameters
    ----------
    path:
        Path to the file that should be patched.
    diff_text:
        Unified diff affecting only ``path``.
    """
    from unidiff import PatchSet

    file_path = Path(path)
    patch_set = PatchSet(StringIO(diff_text))
    if len(patch_set) != 1:
        raise ValueError("Patch must contain exactly one file")
    patched_file = patch_set[0]

    lines = file_path.read_text().splitlines(keepends=True)
    result: list[str] = []
    idx = 0

    for hunk in patched_file:
        start = hunk.source_start - 1
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

