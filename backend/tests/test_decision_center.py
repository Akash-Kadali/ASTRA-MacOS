import io
import json
import zipfile
from pathlib import Path

from backend.api import applications, decision_center, jobright


def _setup(tmp_path: Path, monkeypatch):
    job_store = tmp_path / "jobs.json"
    app_store = tmp_path / "applications.json"
    decision_store = tmp_path / "decisions.json"
    pdf = tmp_path / "resume.pdf"
    pdf.write_bytes(b"%PDF-1.4\ntest")
    jobs = [
        {"id": "match", "company": "Acme Corp", "title": "[Remote] Machine Learning Engineer",
         "url": "https://jobright.ai/jobs/info/match", "posted": "1 hour ago", "captured_at": "2026-01-01T00:00:00Z",
         "description": "Responsibilities\nBuild ML systems\n\nRequired qualifications\nMaster's degree and 2 years experience",
         "description_sections": {"required": "Master's degree and 2 years experience"}},
        {"id": "phd", "company": "Research Co", "title": "Research Scientist",
         "url": "https://jobright.ai/jobs/info/phd", "posted": "2 hours ago", "captured_at": "2026-01-01T00:00:00Z",
         "description": "Required qualifications\nPursuing PhD in Computer Science",
         "description_sections": {"required": "Pursuing PhD in Computer Science"}},
    ]
    job_store.write_text(json.dumps({"items": jobs}), encoding="utf-8")
    app_store.write_text(json.dumps({"applications": {"app1": {
        "id": "app1", "company": "Acme", "role": "Machine Learning Engineer", "status": "applied",
        "job_url": "", "jd_text": "", "files": [{"path": str(pdf)}],
    }}}), encoding="utf-8")
    monkeypatch.setattr(jobright, "STORE", job_store)
    monkeypatch.setattr(applications, "STORE_PATH", app_store)
    monkeypatch.setattr(decision_center, "STORE", decision_store)
    monkeypatch.setattr(decision_center, "BACKUPS", tmp_path / "backups")
    decision_center._APP_CACHE_STAMP = -1
    decision_center._APP_CACHE = []
    return pdf


def test_decision_centre_explains_matches_and_eligibility(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    result = decision_center.list_decisions(view="all", q="", category="all", limit=20)
    by_id = {x["id"]: x for x in result["items"]}
    assert by_id["match"]["decision"] == "applied"
    assert by_id["match"]["match_details"]["pdf_path"].endswith("resume.pdf")
    assert by_id["phd"]["decision"] == "ineligible"
    assert "PhD" in by_id["phd"]["decision_reason"]


def test_not_applied_override_restores_job(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    response = decision_center.job_action("match", {"action": "not_applied"})
    assert response["item"]["decision"] == "actionable"
    assert decision_center.list_decisions(view="actionable", q="", category="all", limit=20)["total"] == 1


def test_application_package_contains_evidence(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    response = decision_center.package("match")
    archive = zipfile.ZipFile(io.BytesIO(response.body))
    assert {"resume.pdf", "job-description.txt", "job-link.txt", "decision-report.json"} <= set(archive.namelist())
    assert archive.read("resume.pdf").startswith(b"%PDF")


def test_backup_copies_tracker_and_decisions(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    decision_center._save(decision_center._empty())
    result = decision_center.backup()
    assert result["ok"]
    assert "applications.json" in result["files"]
    assert "decisions.json" in result["files"]


def test_true_duplicate_captures_are_grouped(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    store = jobright._read_store()
    duplicate = {**store["items"][0], "id": "match-copy", "captured_at": "2026-01-02T00:00:00Z"}
    store["items"].append(duplicate)
    jobright.STORE.write_text(json.dumps(store), encoding="utf-8")
    result = decision_center.list_decisions(view="all", q="", category="all", limit=20)
    match = next(x for x in result["items"] if x["id"] in {"match", "match-copy"})
    assert result["raw_total"] == 3
    assert result["total"] == 2
    assert match["duplicate_count"] == 2
    assert set(match["duplicate_ids"]) == {"match", "match-copy"}


def test_submission_confirmation_saves_requisition(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    decision_center.job_action("phd", {
        "action": "applied", "requisition_id": "REQ-42", "submitted_at": "2026-10-31T10:00:00Z",
    })
    saved = applications._load()["applications"]
    record = next(x for x in saved.values() if x.get("requisition_id") == "REQ-42")
    assert record["applied_at"] == "2026-10-31T10:00:00Z"
    assert record["job_url"].endswith("/phd")
