import json
from pathlib import Path

from backend.core import operational_store


def test_checkpoint_round_trip_uses_sqlite(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(operational_store, "DB_PATH", tmp_path / "astra.sqlite3")
    operational_store.initialize()
    saved = operational_store.save_checkpoint(
        "job-1", status="needs_attention", step="form",
        completed_fields=["Name"], unanswered_fields=["Referral"],
        answers={"Why us?": "The work matches my experience."},
    )
    loaded = operational_store.get_checkpoint("job-1")
    assert loaded is not None
    assert loaded["status"] == saved["status"]
    assert loaded["completed_fields"] == ["Name"]
    assert loaded["unanswered_fields"] == ["Referral"]
    assert loaded["answers"]["Why us?"].startswith("The work")


def test_activity_and_json_mirror_are_transactional(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(operational_store, "DB_PATH", tmp_path / "astra.sqlite3")
    source = tmp_path / "applications.json"
    source.write_text(json.dumps({"applications": {"one": {"company": "ASTRA"}}}), encoding="utf-8")
    assert operational_store.mirror_json("applications", source)
    operational_store.add_activity("integrity", "repaired", "Removed one missing reference")
    assert operational_store.activities()[0]["kind"] == "integrity"
    assert operational_store.status()["snapshots"] == 1


def test_indexed_record_sync_replaces_collection(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(operational_store, "DB_PATH", tmp_path / "astra.sqlite3")
    assert operational_store.sync_records("application", [
        {"id": "a", "company": "Acme", "role": "ML Engineer", "lifecycle_stage": "generated"},
        {"id": "b", "company": "Beta", "role": "AI Engineer", "lifecycle_stage": "submitted"},
    ]) == 2
    assert operational_store.record_counts()["application"] == 2
    operational_store.sync_records("application", [{"id": "b", "company": "Beta"}])
    assert operational_store.record_counts()["application"] == 1
