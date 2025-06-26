import random
import numpy as np

from utils.web import normalize_url, decide_navigation_action


def test_normalize_url_basic():
    url = "HTTP://www.Example.com:80/a/../b/?b=2&a=1#frag"
    assert normalize_url(url) == "http://example.com/b?a=1&b=2"


def test_normalize_url_tracking():
    url = "https://example.com/page?utm_source=foo&x=1"
    assert normalize_url(url) == "https://example.com/page?x=1"


class DummyModel:
    def predict(self, X):
        return np.tile([[0.1, 0.7, 0.2]], (len(X), 1))


def test_decide_navigation_action_basic():
    link = {"url": "https://example.com/", "text": "next", "element": ["nav"]}
    kb = {
        "site_patterns": {
            "example.com": {"navigation_elements": [{"classes": ["nav"]}]}
        }
    }
    config = {
        "priority_urls": [],
        "content_keywords": [],
        "navigation_keywords": ["next"],
        "exploration_rate": 0.0,
        "exploration_decay": 1.0,
        "min_exploration_rate": 0.0,
    }
    action = decide_navigation_action(link, "example.com", kb, DummyModel(), config)
    assert action == 1


def test_decide_navigation_action_exploration():
    random.seed(0)
    link = {"url": "https://example.com/", "text": ""}
    kb = {"site_patterns": {}}
    config = {
        "priority_urls": [],
        "content_keywords": [],
        "navigation_keywords": [],
        "exploration_rate": 1.0,
        "exploration_decay": 0.5,
        "min_exploration_rate": 0.1,
    }
    action = decide_navigation_action(link, "example.com", kb, DummyModel(), config)
    assert 0 <= action <= 2
    assert config["exploration_rate"] == 0.5
