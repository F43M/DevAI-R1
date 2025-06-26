"""Quality classification utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple
import re
from statistics import mean
from .code_sniffer import scan
from .ast_tools import get_functions_complexity
from .code import normalize_indentation, remove_comments, detect_programming_language
from .sonarqube import analyze_code


def evaluate_code_quality(code: str, language: str | None = None) -> Dict[str, float]:
    """Return static analysis metrics and a quality score for ``code``.

    Parameters
    ----------
    code: str
        Source code snippet.
    language: str | None
        Optional language hint. If ``None`` the language is detected.

    Returns
    -------
    Dict[str, float]
        Mapping with ``complexity``, ``lint_errors`` and ``score`` fields.
    """

    lang = language or detect_programming_language(code)
    cleaned = normalize_indentation(remove_comments(code, lang))
    problems, _ = scan(cleaned)
    complexities = get_functions_complexity(cleaned, lang)
    max_complexity = max(complexities.values()) if complexities else 1
    sonar_metrics = analyze_code(cleaned)
    lint_errors = len(problems) + int(sonar_metrics.get("code_smells", 0))
    score = max(0.0, 10.0 - max_complexity - lint_errors)
    return {
        "language": lang,
        "complexity": float(max_complexity),
        "lint_errors": float(lint_errors),
        "score": float(score),
        "sonarqube": sonar_metrics,
    }


POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "awesome",
    "nice",
    "love",
    "like",
    "well",
    "cool",
}

NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "hate",
    "bug",
    "poor",
    "worse",
    "worst",
}


def classify_github_repo(repo: Dict) -> Tuple[str, str]:
    """Classify the quality of a GitHub repository.

    Parameters
    ----------
    repo: Dict
        Repository metadata containing at least ``stars`` and ``open_issues``.

    Returns
    -------
    Tuple[str, str]
        Quality level (``high``, ``medium`` or ``low``) and reason.
    """
    stars = repo.get("stars") or repo.get("stargazers_count", 0)
    issues = repo.get("open_issues") or repo.get("open_issues_count", 0)
    score = repo.get("quality_score")
    if score is None:
        score = stars / (issues + 1)

    commit_freq = repo.get("commit_frequency", 0.0)
    closure_rate = repo.get("issue_closure_rate", 0.0)
    maintenance = commit_freq * closure_rate

    popularity = compute_popularity_metrics(repo) / 100.0
    score += maintenance + popularity

    has_tests = repo.get("has_tests") or repo.get("tests")

    if score >= 5:
        quality = "high"
        reason = "high star to issue ratio"
    elif score >= 2:
        quality = "medium"
        reason = "moderate star to issue ratio"
    else:
        quality = "low"
        reason = "low star to issue ratio"

    if maintenance > 1:
        reason += " and active maintenance"
    elif maintenance > 0:
        reason += " and limited maintenance"

    if popularity > 1:
        reason += " popular project"

    if has_tests:
        reason += " with tests"
    return quality, reason


def classify_stackoverflow_answer(answer: Dict) -> Tuple[str, str]:
    """Classify the quality of a StackOverflow answer.

    Parameters
    ----------
    answer: Dict
        Data containing ``score`` and optionally ``is_accepted``.

    Returns
    -------
    Tuple[str, str]
        Quality level and reason.
    """
    score = answer.get("score", 0)
    accepted = answer.get("is_accepted") or answer.get("accepted", False)

    if score >= 5 and accepted:
        return "high", "accepted answer with high score"
    if score >= 2:
        return "medium", "community upvoted"
    return "low", "low score"


def comment_sentiment_score(comment: str) -> float:
    """Return a simple sentiment polarity score for ``comment``."""

    tokens = re.findall(r"\w+", comment.lower())
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def analyze_comment_sentiment(comments: List[str]) -> str:
    """Classify average sentiment of a list of comments."""

    scores = [comment_sentiment_score(c) for c in comments if c]
    avg = mean(scores) if scores else 0.0
    if avg > 0.1:
        return "positive"
    if avg < -0.1:
        return "negative"
    return "neutral"


def compute_popularity_metrics(repo: Dict) -> float:
    """Return a popularity score using stars, forks and watchers."""

    stars = repo.get("stars") or repo.get("stargazers_count", 0)
    forks = repo.get("forks", 0)
    watchers = repo.get("watchers", 0)
    return stars + 0.5 * forks + 0.2 * watchers


def balance_quality(records: List[Dict]) -> List[Dict]:
    """Balance dataset records by quality level.

    The function keeps the same number of records for each quality
    class by truncating larger groups.
    """
    groups: Dict[str, List[Dict]] = {"high": [], "medium": [], "low": []}
    for rec in records:
        groups.setdefault(rec.get("quality", "medium"), []).append(rec)

    counts = [len(v) for v in groups.values() if v]
    if not counts:
        return records
    min_count = min(counts)
    balanced: List[Dict] = []
    for q in ["high", "medium", "low"]:
        balanced.extend(groups.get(q, [])[:min_count])
    return balanced


def generate_challenge_prompt(problems: List[str]) -> str:
    """Generate a Portuguese prompt challenging the user to fix the code.

    Parameters
    ----------
    problems: List[str]
        List of problems detected in the code.

    Returns
    -------
    str
        A short text in Portuguese describing the issues.
    """

    if not problems:
        return ""

    if len(problems) == 1:
        return f"Este código tem um bug: {problems[0]}. Corrija-o."

    joined = "; ".join(problems)
    return f"Este código tem alguns bugs: {joined}. Corrija-os."
