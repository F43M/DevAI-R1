from devai.symbolic_verification import evaluate_ai_response


def test_evaluate_ai_response_basic():
    score, detail = evaluate_ai_response("def foo():\n    return 1")
    assert score.startswith("C")
    assert isinstance(detail, dict)
