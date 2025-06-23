from __future__ import annotations

from typing import Iterable

from .ai_model import AIModel
from .config import config
from .context_manager import CodeContext
from .post_processor import is_valid_python, fix_code


def _sliding_windows(lines: list[str], size: int, overlap: float) -> Iterable[str]:
    step = max(1, int(size * (1 - overlap)))
    for start in range(0, len(lines), step):
        end = start + size
        yield "\n".join(lines[start:end])
        if end >= len(lines):
            break


async def generate_long_code(
    prompt: str,
    model: AIModel,
    context: CodeContext | None = None,
    *,
    window_size: int = 50,
    overlap_ratio: float = 0.1,
) -> str:
    """Generate long code using a sliding window approach."""
    lines = prompt.splitlines()
    context = context or CodeContext()
    generated_parts: list[str] = []
    for chunk in _sliding_windows(lines, window_size, overlap_ratio):
        ctx_text = context.text()
        full_prompt = f"{ctx_text}\n{chunk}" if ctx_text else chunk
        part = await model.safe_api_call(full_prompt, config.MAX_CONTEXT_LENGTH)
        generated_parts.append(part)
        context.append(part)
    result = "\n".join(generated_parts)
    if not is_valid_python(result):
        result = fix_code(result)
    return result
