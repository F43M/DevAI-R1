import sys
import json
from pathlib import Path

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dq


def test_deduplicate_by_simhash_removes_near_duplicates():
    records = [
        {"content": "Python is great for testing algorithms"},
        {"content": "Python is great for testing algorithm"},
        {"content": "Completely different text"},
    ]
    unique, removed = dq.deduplicate_by_simhash(records, distance=3)
    assert removed == 1
    assert len(unique) == 2


def test_complete_missing_fields_adds_new_keys():
    records = [
        {"title": "Python", "language": "en"},
    ]
    extra = [
        {"title": "Python", "language": "en", "wikidata_id": "Q1", "image_url": "img"}
    ]
    res = dq.complete_missing_fields(records, extra)
    assert res[0]["wikidata_id"] == "Q1"
    assert res[0]["image_url"] == "img"


def test_detect_leaks_by_hash_identifies_overlap():
    ref = [{"id": "1", "content": "abc", "language": "en"}]
    recs = [
        {"id": "2", "content": "abc", "language": "en"},
        {"id": "3", "content": "xyz", "language": "en"},
    ]
    leaks = dq.detect_leaks_by_hash(recs, ref)
    assert len(leaks) == 1
    assert leaks[0]["id"] == "2"


def test_detect_leaks_by_embedding_identifies_similar():
    ref = [{"content_embedding": [0.0, 1.0]}]
    recs = [
        {"content_embedding": [0.0, 0.99]},
        {"content_embedding": [1.0, 0.0]},
    ]
    leaks = dq.detect_leaks_by_embedding(recs, ref, threshold=0.98)
    assert len(leaks) == 1


def test_strip_credentials_replaces_tokens():
    code = "token = 'ghp_0123456789abcdef0123456789abcdef0123'"
    sanitized = dq.strip_credentials(code)
    assert "<REDACTED>" in sanitized


def test_remove_pii_redacts_data():
    text = "Contact me at user@example.com or call 555-123-4567"
    cleaned = dq.remove_pii(text)
    assert "<PII>" in cleaned


def test_detect_code_plagiarism_flags_duplicates():
    recs = [
        {"content": "print('hi')"},
        {"content": "print('hi')"},
        {"content": "print('bye')"},
    ]
    plag = dq.detect_code_plagiarism(recs, threshold=0.9)
    assert len(plag) == 1


def test_deduplicate_by_simhash_no_false_positive():
    records = [
        {"content": "alpha beta gamma"},
        {"content": "completely different"},
        {"content": "another unrelated text"},
    ]
    unique, removed = dq.deduplicate_by_simhash(records)
    assert removed == 0
    assert len(unique) == 3


def test_check_sensitive_embeddings_flags_near_match(tmp_path):
    ref_path = tmp_path / "sens.json"
    h = dq._embedding_to_simhash([0.1, 0.2])
    ref_path.write_text(json.dumps([hex(h.value)]), encoding="utf-8")
    hashes = dq.load_sensitive_hashes(str(ref_path))
    recs = [{"content_embedding": [0.1, 0.21]}, {"content_embedding": [1.0, 0.0]}]
    leaks = dq.check_sensitive_embeddings(recs, hashes, distance=12)
    assert len(leaks) == 1
