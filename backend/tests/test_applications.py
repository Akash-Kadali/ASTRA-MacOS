from pathlib import Path

from fastapi.testclient import TestClient

from backend.api import applications
from main import app


def test_archive_scan_deduplicates_and_tracks_dates(tmp_path: Path, monkeypatch):
    archive = tmp_path / "Job Applications" / "A" / "Acme"
    archive.mkdir(parents=True)
    (archive / "Acme - ML Engineer - Resume.pdf").write_bytes(b"%PDF")
    (archive / "Acme - ML Engineer - Resume - Version 2.pdf").write_bytes(b"%PDF2")
    monkeypatch.setattr(applications, "STORE_PATH", tmp_path / "tracker.json")
    monkeypatch.setattr(applications, "_roots", lambda: [tmp_path / "Job Applications"])

    result = applications.scan_archive()
    assert result["files_scanned"] == 2
    assert result["total_applications"] == 1

    client = TestClient(app)
    listing = client.get("/api/applications", params={"limit": 10}).json()
    assert listing["total"] == 1
    assert listing["items"][0]["company"] == "Acme"
    assert listing["items"][0]["lifecycle_stage"] == "generated"
    assert listing["items"][0]["applied_at"] is None
    assert len(listing["items"][0]["files"]) == 2

    app_id = listing["items"][0]["id"]
    updated = client.patch(
        f"/api/applications/{app_id}",
        json={"status": "recruiter_call", "call_at": "2026-10-25T12:00:00Z"},
    )
    assert updated.status_code == 200
    assert updated.json()["status"] == "recruiter_call"
    assert updated.json()["call_at"] == "2026-10-25T12:00:00Z"


def test_employment_archive_uses_company_before_applications(tmp_path: Path, monkeypatch):
    employment = tmp_path / "02 Employment"
    archive = employment / "Parspec" / "Applications" / "Older Resumes"
    archive.mkdir(parents=True)
    resume = archive / "Parspec - Machine Learning Engineer 1 - Resume.pdf"
    resume.write_bytes(b"%PDF")
    (archive / "Parspec - Machine Learning Engineer 1 - Resume.tex").write_text("source")
    monkeypatch.setattr(applications, "STORE_PATH", tmp_path / "tracker.json")
    monkeypatch.setattr(applications, "_roots", lambda: [employment])

    result = applications.scan_archive()
    assert result["total_applications"] == 1
    assert result["files_scanned"] == 1

    record = next(iter(applications._load()["applications"].values()))
    assert record["company"] == "Parspec"
    assert record["role"] == "Machine Learning Engineer 1"
    assert len(record["files"]) == 1
    assert record["files"][0]["path"].endswith(".pdf")


def test_generated_jobright_resume_keeps_exact_job_url(tmp_path: Path, monkeypatch):
    resume = tmp_path / "Sri Akash Kadali - Acme - ML Engineer.pdf"
    resume.write_bytes(b"%PDF generated")
    monkeypatch.setattr(applications, "STORE_PATH", tmp_path / "tracker.json")

    applications.record_generated(
        "Acme", "ML Engineer", resume,
        "https://jobs.example.com/acme/ml-engineer", "Jobright",
    )

    record = next(iter(applications._load()["applications"].values()))
    assert record["job_url"] == "https://jobs.example.com/acme/ml-engineer"
    assert record["source"] == "Jobright"
    assert record["lifecycle_stage"] == "generated"


def test_only_confirmation_promotes_to_submitted(tmp_path: Path, monkeypatch):
    resume = tmp_path / "resume.pdf"
    resume.write_bytes(b"%PDF")
    monkeypatch.setattr(applications, "STORE_PATH", tmp_path / "tracker.json")
    applications.record_generated("Acme", "ML Engineer", resume)
    record = next(iter(applications._load()["applications"].values()))
    assert record["lifecycle_stage"] == "generated"
    import asyncio
    submitted = asyncio.run(applications.update_application(record["id"], {"status": "submitted"}))
    assert submitted["lifecycle_stage"] == "submitted"
    assert submitted["applied_at"]


def test_unconfirmed_outcomes_do_not_corrupt_response_rate():
    items = [
        {"id": "generated", "status": "applied", "applied_date_source": "file-created date", "call_at": "2026-01-01"},
        {"id": "submitted", "status": "submitted", "lifecycle_stage": "submitted", "call_at": "2026-01-02"},
    ]
    stats = applications._stats(items)
    assert stats["submitted"] == 1
    assert stats["calls"] == 1
    assert stats["response_rate"] == 100.0
