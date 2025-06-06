from devai.core import CodeMemoryAI

ai = object.__new__(CodeMemoryAI)

def test_extract_tags():
    tags = CodeMemoryAI._extract_tags(ai, "Erro crítico ⚠️ TODO")
    assert "erro" in tags
    assert "aviso" in tags

def test_extract_inputs():
    code = "def f(a, b):\n    return a + b"
    assert CodeMemoryAI._extract_inputs(ai, code) == ["a", "b"]

def test_infer_return_type():
    code = "def f():\n    return 1"
    assert CodeMemoryAI._infer_return_type(ai, code) == "number"
