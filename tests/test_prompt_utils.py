from devai.prompt_utils import build_cot_prompt


def test_build_cot_prompt():
    prompt = build_cot_prompt(
        "teste",
        memories=[{"similarity_score": 1.0, "content": "m", "tags": []}],
        chunks=[{"file": "f.py", "type": "func", "name": "f", "dependencies": [], "code": "def f(): pass"}],
    )
    assert "passo a passo" in prompt
