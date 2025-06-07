from typing import Sequence, Dict


def build_metacognition_prompt(history: Sequence[Dict]) -> str:
    """Compose prompt asking model for next steps based on decision history."""
    lines = []
    for item in history[-5:]:
        ts = item.get("timestamp", "")
        lines.append(f"{ts} - {item.get('tipo')} em {item.get('modulo')}")
    hist_text = "\n".join(lines)
    return (
        f"Com base nas decisões abaixo, qual próximo passo você recomendaria?\n{hist_text}\nResposta:".strip()
    )
