"""Minimal YAML helpers used when PyYAML is unavailable."""

from typing import Any


def safe_load(stream: Any) -> dict:
    content = stream.read() if hasattr(stream, "read") else str(stream)
    data = {}
    for line in content.strip().splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            data[k.strip()] = _parse_value(v.strip())
    return data


def _parse_value(val: str) -> Any:
    if val.isdigit():
        return int(val)
    try:
        return float(val)
    except ValueError:
        return val


def safe_dump(data: Any, allow_unicode: bool = True) -> str:
    """Very small subset of ``yaml.safe_dump`` used for tests."""
    lines = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                first = True
                for key, value in item.items():
                    prefix = "- " if first else "  "
                    lines.append(f"{prefix}{key}: {value}")
                    first = False
                if first:
                    lines.append("-")
            else:
                lines.append(f"- {item}")
    elif isinstance(data, dict):
        for key, value in data.items():
            lines.append(f"{key}: {value}")
    else:
        lines.append(str(data))
    return "\n".join(lines) + "\n"
