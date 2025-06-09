from devai.dialog_summarizer import DialogSummarizer
from devai import dialog_summarizer
from devai.config import config


def test_dialog_summarizer_basic():
    history = [
        {"role": "user", "content": "prefiro que você use o padrão snake_case"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "não use eval, é inseguro"},
        {"role": "assistant", "content": "certo"},
        {"role": "user", "content": "faça só com funções puras"},
        {"role": "assistant", "content": "beleza"},
    ]
    summarizer = DialogSummarizer()
    summary = summarizer.summarize_conversation(history)
    assert any(m["tag"] == "#licao_aprendida" for m in summary)
    assert any(s["content"].endswith("por razões de segurança.") for s in summary)
    tags = {m["tag"] for m in summary}
    assert "#estilo_requisitado" in tags
    assert "#preferencia_usuario" in tags


def test_dialog_summarizer_ai_fallback(monkeypatch):
    history = [{"role": "user", "content": "Oi"}]

    class DummyModel:
        async def safe_api_call(self, prompt, max_tokens, context="", memory=None, temperature=0.7):
            DummyModel.called += 1
            return "- #preferencia_usuario: respostas curtas"

        async def close(self):
            pass

    DummyModel.called = 0

    monkeypatch.setattr(dialog_summarizer, "AIModel", DummyModel)
    monkeypatch.setattr(config, "ENABLE_AI_SUMMARY", True)
    monkeypatch.setattr(DialogSummarizer, "_register_fallback", lambda self: None)

    saved = []

    class DummyMemory:
        def save(self, item):
            saved.append(item)

    summarizer = DialogSummarizer()
    summarizer.summarize_conversation(history, memory=DummyMemory())

    assert DummyModel.called == 1
    assert saved
