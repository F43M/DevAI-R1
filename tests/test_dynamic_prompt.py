import logging

from devai.prompt_engine import build_dynamic_prompt


def test_dynamic_prompt_includes_error_blocks():
    context = {
        "logs": "trace",
        "actions": [{"task": "run"}],
        "graph": "g",
        "memories": [],
    }
    prompt = build_dynamic_prompt("Por que deu erro?", context, "normal")
    assert "Logs recentes" in prompt
    assert "Ultimas ações" in prompt
    assert "Explique antes de responder." in prompt


def test_dynamic_prompt_fallback_logged(caplog):
    context = {"logs": "t", "actions": [{"task": "a"}], "graph": "g"}
    with caplog.at_level(logging.INFO):
        build_dynamic_prompt("Oi", context, "normal")
    assert any("Fallback" in r.message for r in caplog.records)


def test_dynamic_prompt_logs_reasons(caplog):
    context = {
        "logs": "trace",
        "actions": [{"task": "run"}],
        "graph": "g",
        "memories": [],
    }
    with caplog.at_level(logging.INFO):
        build_dynamic_prompt("Por que deu erro?", context, "normal", intent="debug")
    assert any("reasons=" in r.message for r in caplog.records)

