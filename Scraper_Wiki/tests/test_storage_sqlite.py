import json
import sqlite3
from integrations.storage_sqlite import (
    save_to_db,
    get_last_processed,
    set_last_processed,
)


def test_save_to_db_creates_table_and_inserts(tmp_path):
    db_file = tmp_path / "test.sqlite"
    data = {"a": 1}
    save_to_db(data, table="info", db_path=str(db_file))
    conn = sqlite3.connect(db_file)
    row = conn.execute("SELECT data FROM info").fetchone()
    conn.close()
    assert row is not None
    assert json.loads(row[0]) == data


def test_save_to_db_compression(tmp_path):
    db_file = tmp_path / "c.sqlite"
    data = {"x": 5}
    save_to_db(data, table="info", db_path=str(db_file), compression="gzip")
    conn = sqlite3.connect(db_file)
    row = conn.execute("SELECT data FROM info").fetchone()[0]
    conn.close()
    import gzip

    assert json.loads(gzip.decompress(row).decode()) == data


def test_metadata_functions(tmp_path):
    db_file = tmp_path / "meta.sqlite"
    assert get_last_processed("repo", db_path=str(db_file)) is None
    set_last_processed("repo", "2024-01-01T00:00:00Z", db_path=str(db_file))
    assert get_last_processed("repo", db_path=str(db_file)) == "2024-01-01T00:00:00Z"
    set_last_processed("repo", "2024-02-02T00:00:00Z", db_path=str(db_file))
    assert get_last_processed("repo", db_path=str(db_file)) == "2024-02-02T00:00:00Z"
