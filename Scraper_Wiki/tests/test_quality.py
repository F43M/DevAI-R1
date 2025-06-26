import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy dependencies before importing utils package
sys.modules.setdefault("spacy", SimpleNamespace(load=lambda *a, **k: None))
sys.modules.setdefault(
    "sentence_transformers",
    SimpleNamespace(SentenceTransformer=lambda *a, **k: None),
)
sys.modules.setdefault("networkx", SimpleNamespace())

quality = importlib.import_module("utils.quality")


def test_classify_github_repo_high_with_tests():
    repo = {"stars": 10, "open_issues": 1, "has_tests": True}
    q, reason = quality.classify_github_repo(repo)
    assert q == "high"
    assert "tests" in reason


def test_classify_github_repo_low():
    repo = {"stars": 1, "open_issues": 10}
    q, _ = quality.classify_github_repo(repo)
    assert q == "low"


def test_classify_stackoverflow_answer_medium():
    ans = {"score": 3, "is_accepted": False}
    q, reason = quality.classify_stackoverflow_answer(ans)
    assert q == "medium"
    assert "community" in reason


def test_balance_quality_truncates_groups():
    recs = [
        {"id": 1, "quality": "high"},
        {"id": 2, "quality": "high"},
        {"id": 3, "quality": "medium"},
        {"id": 4, "quality": "low"},
    ]
    balanced = quality.balance_quality(recs)
    counts = {
        q: len([r for r in balanced if r["quality"] == q])
        for q in ["high", "medium", "low"]
    }
    assert counts == {"high": 1, "medium": 1, "low": 1}


def test_analyze_comment_sentiment_positive():
    comments = ["Great job", "I love it"]
    assert quality.analyze_comment_sentiment(comments) == "positive"


def test_analyze_comment_sentiment_negative():
    comments = ["terrible bug", "bad"]
    assert quality.analyze_comment_sentiment(comments) == "negative"


def test_classify_github_repo_with_maintenance():
    repo = {
        "stars": 2,
        "open_issues": 1,
        "commit_frequency": 5,
        "issue_closure_rate": 0.9,
    }
    q, reason = quality.classify_github_repo(repo)
    assert q == "high"
    assert "maintenance" in reason
