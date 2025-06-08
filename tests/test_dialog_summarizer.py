from devai.dialog_summarizer import DialogSummarizer


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
