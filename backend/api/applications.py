"""Persistent application tracker built from ASTRA's local resume archive."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import Response

from backend.core import config, operational_store

router = APIRouter(prefix="/api/applications", tags=["applications"])

STORE_PATH = Path(config.DATA_DIR) / "application_tracker.json"
_LOCK = threading.RLock()
_ALLOWED_SUFFIXES = {".pdf"}
STATUSES = (
    "generated", "prepared", "submitted", "saved", "applied", "recruiter_call", "interview",
    "offer", "rejected", "withdrawn",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _roots() -> List[Path]:
    configured = os.getenv("ASTRA_APPLICATION_ROOTS", "").strip()
    if configured:
        candidates = [Path(x).expanduser() for x in configured.split(os.pathsep) if x.strip()]
    else:
        home = Path.home()
        candidates = [
            home / "Desktop" / "ASTRA Resumes",
            home / "Desktop" / "01 Career" / "Job Applications",
            home / "Downloads" / "01 Career" / "Job Applications",
            home / "Documents" / "01 Career" / "Job Applications",
            home / "Desktop" / "02 Employment" / "Parspec" / "Applications",
            home / "Documents" / "02 Employment" / "Parspec" / "Applications",
            Path(config.OPTIMIZED_DIR),
            Path(config.HUMANIZED_DIR),
        ]
    seen, result = set(), []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def _empty_store() -> Dict[str, Any]:
    return {"version": 1, "updated_at": _now(), "last_scan_at": None, "applications": {}}


def _load() -> Dict[str, Any]:
    try:
        data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("applications"), dict):
            for app in data["applications"].values():
                _normalize_lifecycle(app)
            return data
    except Exception:
        pass
    return _empty_store()


def _save(data: Dict[str, Any]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = _now()
    pending = STORE_PATH.with_suffix(".tmp")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(pending, STORE_PATH)
    operational_store.sync_records("application", list(data.get("applications", {}).values()))


def _normalize_lifecycle(app: Dict[str, Any]) -> str:
    stage = str(app.get("lifecycle_stage") or "").casefold()
    if stage in {"generated", "prepared", "submitted"}:
        return stage
    status = str(app.get("status") or "").casefold()
    source = str(app.get("applied_date_source") or "").casefold()
    if status in {"recruiter_call", "interview", "offer", "rejected", "withdrawn", "submitted"}:
        stage = "submitted"
    elif status == "applied" and source not in {"file-created date", "not confirmed", ""}:
        stage = "submitted"
    elif status == "prepared":
        stage = "prepared"
    else:
        stage = "generated"
    app["lifecycle_stage"] = stage
    return stage


def _key(company: str, role: str) -> str:
    normalized = re.sub(r"\s+", " ", f"{company}|{role}").strip().casefold()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:20]


def desktop_resume_path(company: str, role: str, payload: bytes) -> Path:
    """Save every generated resume in the user-facing Desktop library."""
    def clean(value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r'[/\\:*?"<>|]', "-", value)).strip(" .-")
    folder = Path.home() / "Desktop" / "ASTRA Resumes"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"Sri Akash Kadali - {clean(company) or 'Company'} - {clean(role) or 'Role'}"[:220].rstrip(" .-")
    candidate = folder / f"{stem}.pdf"
    if candidate.is_file():
        try:
            if hashlib.sha256(candidate.read_bytes()).digest() == hashlib.sha256(payload).digest():
                return candidate
        except OSError:
            pass
        version = 2
        while (folder / f"{stem} - v{version}.pdf").exists():
            version += 1
        candidate = folder / f"{stem} - v{version}.pdf"
    pending = candidate.with_suffix(".tmp")
    pending.write_bytes(payload)
    os.replace(pending, candidate)
    return candidate


_TAIL = re.compile(
    r"\s+-\s+(?:Resume|Humanized|Optimized|Cover Letter)(?:\s+-\s+.*)?$",
    re.IGNORECASE,
)


def _parse_file(path: Path, root: Path) -> Optional[Dict[str, Any]]:
    name = path.stem.strip()
    is_astra_resume_library = root.name.casefold() == "astra resumes"
    if ("resume" not in name.casefold() and not is_astra_resume_library) or "cover letter" in name.casefold():
        return None
    relative = path.relative_to(root)
    parts = relative.parts
    company = ""
    application_idx = next(
        (i for i, part in enumerate(parts) if part.casefold() == "applications"),
        None,
    )
    if root.name.casefold() == "applications" and root.parent.name:
        company = root.parent.name.strip()
    elif application_idx is not None and application_idx > 0:
        company = parts[application_idx - 1].strip()
    elif len(parts) >= 3 and len(parts[0]) <= 2:
        company = parts[1].strip()
    elif len(parts) >= 2:
        company = parts[-2].strip()

    cleaned = _TAIL.sub("", name)
    cleaned = re.sub(r"\s+-\s+(?:Version\s*\d+|Copy|HIREX Archive|Duplicate Archive).*$", "", cleaned, flags=re.I)
    # New library convention: Sri Akash Kadali - Company - Role.pdf
    if is_astra_resume_library:
        named_parts = [part.strip() for part in cleaned.split(" - ") if part.strip()]
        if len(named_parts) >= 3 and named_parts[0].casefold() == "sri akash kadali":
            company = named_parts[1]
            cleaned = " - ".join(named_parts[1:])
    if not company:
        company = cleaned.split(" - ", 1)[0].strip()
    prefix = f"{company} - "
    role = cleaned[len(prefix):].strip() if cleaned.casefold().startswith(prefix.casefold()) else cleaned
    role = re.sub(r"^Sri(?:_Kadali)?_", "", role, flags=re.I).replace("_", " ").strip(" -_")
    if not role or role.casefold() == company.casefold():
        role = "Role not identified"

    stat = path.stat()
    created = getattr(stat, "st_birthtime", stat.st_mtime)
    created_at = datetime.fromtimestamp(created, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "company": company,
        "role": role,
        "path": str(path),
        "created_at": created_at,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "size": stat.st_size,
    }


def scan_archive() -> Dict[str, Any]:
    with _LOCK:
        store = _load()
        apps = store["applications"]
        discovered = 0
        touched = set()
        for root in _roots():
            try:
                paths = root.rglob("*")
                for path in paths:
                    if not path.is_file() or path.suffix.casefold() not in _ALLOWED_SUFFIXES:
                        continue
                    parsed = _parse_file(path, root)
                    if not parsed:
                        continue
                    discovered += 1
                    app_id = _key(parsed["company"], parsed["role"])
                    touched.add(app_id)
                    existing = apps.get(app_id, {})
                    files = {f.get("path"): f for f in existing.get("files", []) if f.get("path")}
                    files[parsed["path"]] = {k: parsed[k] for k in ("path", "created_at", "modified_at", "size")}
                    first_seen = min((f["created_at"] for f in files.values()), default=parsed["created_at"])
                    apps[app_id] = {
                        "id": app_id,
                        "company": existing.get("company") or parsed["company"],
                        "role": existing.get("role") or parsed["role"],
                        "status": existing.get("status") or "generated",
                        "lifecycle_stage": _normalize_lifecycle(existing) if existing else "generated",
                        "generated_at": existing.get("generated_at") or first_seen,
                        "applied_at": existing.get("applied_at"),
                        "applied_date_source": existing.get("applied_date_source") or "not confirmed",
                        "call_at": existing.get("call_at"),
                        "interview_at": existing.get("interview_at"),
                        "offer_at": existing.get("offer_at"),
                        "rejected_at": existing.get("rejected_at"),
                        "follow_up_at": existing.get("follow_up_at"),
                        "source": existing.get("source") or "ASTRA archive",
                        "job_url": existing.get("job_url") or "",
                        "notes": existing.get("notes") or "",
                        "files": sorted(files.values(), key=lambda f: f["modified_at"], reverse=True),
                        "first_seen_at": existing.get("first_seen_at") or first_seen,
                        "updated_at": _now(),
                    }
            except (OSError, ValueError):
                continue
        # Resumes may have been moved into Desktop/ASTRA Resumes. Do not keep
        # dead legacy paths because the UI can otherwise select a missing file.
        for app in apps.values():
            app["files"] = [f for f in app.get("files", []) if Path(str(f.get("path", ""))).is_file()]
        store["last_scan_at"] = _now()
        _save(store)
        return {
            "files_scanned": discovered,
            "applications_found": len(touched),
            "total_applications": len(apps),
            "roots": [str(p) for p in _roots()],
            "updated_at": store["last_scan_at"],
        }


def record_generated(company: str, role: str, resume_path: Path, job_url: str = "",
                     source: str = "ASTRA", jd_text: str = "") -> None:
    """Register new ASTRA output immediately; submission remains user-confirmed."""
    if not company.strip() or not role.strip():
        return
    with _LOCK:
        store = _load()
        app_id = _key(company, role)
        existing = store["applications"].get(app_id, {})
        stat = resume_path.stat() if resume_path.exists() else None
        created_at = _now()
        files = {f.get("path"): f for f in existing.get("files", []) if f.get("path")}
        if stat:
            files[str(resume_path)] = {
                "path": str(resume_path), "created_at": created_at,
                "modified_at": created_at, "size": stat.st_size,
            }
        store["applications"][app_id] = {
            "id": app_id, "company": company.strip(), "role": role.strip(),
            "status": existing.get("status") or "generated",
            "lifecycle_stage": existing.get("lifecycle_stage") or "generated",
            "generated_at": existing.get("generated_at") or created_at,
            "applied_at": existing.get("applied_at"),
            "applied_date_source": existing.get("applied_date_source") or "not confirmed",
            "call_at": existing.get("call_at"), "interview_at": existing.get("interview_at"),
            "offer_at": existing.get("offer_at"), "rejected_at": existing.get("rejected_at"),
            "follow_up_at": existing.get("follow_up_at"), "source": existing.get("source") or source,
            "job_url": existing.get("job_url") or job_url, "notes": existing.get("notes") or "",
            "jd_text": existing.get("jd_text") or jd_text,
            "files": sorted(files.values(), key=lambda f: f["modified_at"], reverse=True),
            "first_seen_at": existing.get("first_seen_at") or created_at, "updated_at": _now(),
        }
        _save(store)


def mark_lifecycle(company: str, role: str, stage: str, job_url: str = "") -> bool:
    if stage not in {"generated", "prepared", "submitted"}:
        raise ValueError("Invalid application lifecycle stage")
    with _LOCK:
        store = _load()
        app = store.get("applications", {}).get(_key(company, role))
        if not app:
            return False
        app["lifecycle_stage"] = stage
        if stage == "prepared" and app.get("status") in {"saved", "generated", "applied"}:
            app["status"] = "prepared"
        if stage == "submitted":
            app["status"] = "submitted"
            app["applied_at"] = app.get("applied_at") or _now()
            app["applied_date_source"] = "agent confirmation"
        if job_url:
            app["job_url"] = job_url
        app["updated_at"] = _now()
        _save(store)
        return True


def resolve_job_description(app: Dict[str, Any]) -> str:
    """Recover a saved JD from the application or its originating Jobright job."""
    direct = str(app.get("jd_text") or "").strip()
    if direct:
        return direct
    try:
        from backend.api import jobright
        jobs = jobright._read_store().get("items", [])
    except Exception:
        return ""
    company = str(app.get("company") or "").strip().casefold()
    role = str(app.get("role") or "").strip().casefold()
    url = str(app.get("job_url") or "").strip()
    for job in jobs:
        urls = {str(job.get("url") or "").strip(), str(job.get("application_url") or "").strip()}
        exact_url = bool(url and url in urls)
        exact_name = (str(job.get("company") or "").strip().casefold() == company and
                      str(job.get("title") or "").strip().casefold() == role)
        if exact_url or exact_name:
            return str(job.get("description") or job.get("card_text") or "").strip()
    return ""


def attach_job_url_for_resume(resume_path: Path, job_url: str, source: str = "Jobright") -> int:
    """Attach an exact source URL to every application record containing this PDF."""
    if not job_url.strip():
        return 0
    target = str(resume_path)
    updated = 0
    with _LOCK:
        store = _load()
        for app in store["applications"].values():
            if any(str(item.get("path", "")) == target for item in app.get("files", [])):
                app["job_url"] = job_url.strip()
                app["source"] = app.get("source") or source
                app["updated_at"] = _now()
                updated += 1
        if updated:
            _save(store)
    return updated


def _stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {status: 0 for status in STATUSES}
    companies = set()
    calls = interviews = offers = 0
    for app in items:
        status = app.get("status", "applied")
        counts[status] = counts.get(status, 0) + 1
        companies.add(str(app.get("company", "")).casefold())
        if _normalize_lifecycle(app) == "submitted":
            calls += bool(app.get("call_at"))
            interviews += bool(app.get("interview_at"))
            offers += bool(app.get("offer_at"))
    submitted_items = [app for app in items if _normalize_lifecycle(app) == "submitted"]
    total = len(items)
    submitted = len(submitted_items)
    return {
        "total": total,
        "generated": sum(_normalize_lifecycle(app) == "generated" for app in items),
        "prepared": sum(_normalize_lifecycle(app) == "prepared" for app in items),
        "submitted": submitted,
        "companies": len(companies - {""}),
        "calls": calls,
        "interviews": interviews,
        "offers": offers,
        "response_rate": round((calls / submitted * 100), 1) if submitted else 0,
        "interview_rate": round((interviews / submitted * 100), 1) if submitted else 0,
        "status_counts": counts,
    }


@router.post("/scan")
async def scan_applications():
    return scan_archive()


@router.get("")
@router.get("/")
async def list_applications(
    q: str = Query(""), status: str = Query("all"),
    lifecycle: str = Query("all"),
    page: int = Query(1, ge=1), limit: int = Query(50, ge=1, le=250),
):
    with _LOCK:
        store = _load()
    if not store["applications"]:
        scan_archive()
        with _LOCK:
            store = _load()
    all_items = list(store["applications"].values())
    needle = q.strip().casefold()
    items = [a for a in all_items if (status == "all" or a.get("status") == status)]
    if lifecycle != "all":
        items = [a for a in items if _normalize_lifecycle(a) == lifecycle]
    if needle:
        items = [a for a in items if needle in f"{a.get('company','')} {a.get('role','')} {a.get('notes','')}".casefold()]
    items.sort(key=lambda a: a.get("applied_at") or a.get("first_seen_at") or "", reverse=True)
    start = (page - 1) * limit
    return {
        "items": items[start:start + limit], "total": len(items), "page": page,
        "limit": limit, "stats": _stats(all_items), "last_scan_at": store.get("last_scan_at"),
        "statuses": list(STATUSES), "lifecycles": ["generated", "prepared", "submitted"],
    }


@router.patch("/bulk")
async def bulk_update_applications(payload: Dict[str, Any] = Body(...)):
    ids = [str(x) for x in payload.get("ids", []) if str(x).strip()]
    patch = dict(payload.get("patch") or {})
    if not ids or not patch:
        raise HTTPException(400, "ids and patch are required")
    allowed = {"status", "follow_up_at", "source", "notes"}
    if patch.get("status") and patch["status"] not in STATUSES:
        raise HTTPException(400, "Invalid application status")
    with _LOCK:
        store = _load()
        updated = []
        for application_id in ids:
            app = store["applications"].get(application_id)
            if not app:
                continue
            for key, value in patch.items():
                if key in allowed:
                    app[key] = value
            if patch.get("status") in {"applied", "submitted", "recruiter_call", "interview", "offer", "rejected", "withdrawn"}:
                app["lifecycle_stage"] = "submitted"
                app["applied_at"] = app.get("applied_at") or _now()
            elif patch.get("status") == "prepared":
                app["lifecycle_stage"] = "prepared"
            app["updated_at"] = _now()
            updated.append(application_id)
        _save(store)
    return {"updated": len(updated), "ids": updated}


@router.patch("/{application_id}")
async def update_application(application_id: str, patch: Dict[str, Any] = Body(...)):
    allowed = {
        "company", "role", "status", "applied_at", "call_at", "interview_at",
        "offer_at", "rejected_at", "follow_up_at", "source", "job_url", "notes",
    }
    with _LOCK:
        store = _load()
        app = store["applications"].get(application_id)
        if not app:
            raise HTTPException(404, "Application not found")
        if patch.get("status") and patch["status"] not in STATUSES:
            raise HTTPException(400, "Invalid application status")
        for key, value in patch.items():
            if key in allowed:
                app[key] = value
        if patch.get("status") in {"applied", "submitted", "recruiter_call", "interview", "offer", "rejected", "withdrawn"}:
            app["lifecycle_stage"] = "submitted"
            app["applied_at"] = app.get("applied_at") or _now()
            app["applied_date_source"] = app.get("applied_date_source") if app.get("applied_date_source") not in {"", "not confirmed", "file-created date", None} else "status confirmation"
        elif patch.get("status") == "prepared":
            app["lifecycle_stage"] = "prepared"
        if "applied_at" in patch:
            app["applied_date_source"] = "manual"
        app["updated_at"] = _now()
        _save(store)
        return app


@router.post("/{application_id}/call")
async def record_call(application_id: str, payload: Dict[str, Any] = Body(default={})):
    payload = dict(payload or {})
    payload["call_at"] = payload.get("call_at") or _now()
    payload["status"] = payload.get("status") or "recruiter_call"
    return await update_application(application_id, payload)


@router.post("/{application_id}/timeline")
async def add_application_timeline_event(application_id: str, payload: Dict[str, Any] = Body(...)):
    """Attach a dated note, contact, assessment, interview, or document event."""
    event_type = str(payload.get("type") or "note").strip()
    allowed = {"note", "applied", "recruiter_contact", "assessment", "interview", "follow_up", "offer", "rejected", "document"}
    if event_type not in allowed:
        raise HTTPException(400, "Invalid timeline event type")
    event = {
        "id": hashlib.sha1(f"{application_id}:{_now()}:{event_type}".encode()).hexdigest()[:16],
        "type": event_type, "at": str(payload.get("at") or _now()),
        "note": str(payload.get("note") or ""), "person": str(payload.get("person") or ""),
        "attachment": str(payload.get("attachment") or ""),
    }
    with _LOCK:
        store = _load()
        app = store["applications"].get(application_id)
        if not app:
            raise HTTPException(404, "Application not found")
        app.setdefault("timeline", []).insert(0, event)
        if event_type == "applied":
            app.update(status="submitted", lifecycle_stage="submitted", applied_at=event["at"],
                       applied_date_source="timeline confirmation")
        app["updated_at"] = _now()
        _save(store)
    return event


@router.get("/{application_id}/resume")
async def open_resume(application_id: str, file_index: int = Query(0, ge=0)):
    with _LOCK:
        app = _load()["applications"].get(application_id)
    if not app:
        raise HTTPException(404, "Application not found")
    pdfs = [Path(item.get("path", "")) for item in app.get("files", [])
            if Path(item.get("path", "")).suffix.casefold() == ".pdf"]
    if file_index < len(pdfs) and pdfs[file_index].is_file():
        try:
            return Response(
                _read_pdf_bytes(pdfs[file_index]), media_type="application/pdf",
                headers={"Cache-Control": "no-store", "Content-Disposition": "inline"},
            )
        except OSError as exc:
            raise HTTPException(500, f"Resume PDF could not be read: {exc}") from exc
    raise HTTPException(404, "Resume file is no longer available")


@router.get("/{application_id}/files")
async def list_application_files(application_id: str):
    with _LOCK:
        app = _load()["applications"].get(application_id)
    if not app:
        raise HTTPException(404, "Application not found")
    files = []
    for item in app.get("files", []):
        path = Path(item.get("path", ""))
        if path.suffix.casefold() != ".pdf":
            continue
        files.append({**item, "name": path.name, "available": path.is_file()})
    return {"application_id": application_id, "files": files}


def _extract_pdf_text(path: Path) -> str:
    """Extract readable resume text without making archive browsing depend on LaTeX."""
    try:
        stat = path.stat()
        cache_key = hashlib.sha1(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
        cache_path = Path(config.DATA_DIR) / "pdf_text_cache" / f"{cache_key}.txt"
        if cache_path.is_file():
            cached = cache_path.read_text(encoding="utf-8").strip()
            if cached:
                return cached
    except Exception:
        cache_path = None

    extracted = ""
    try:
        from pypdf import PdfReader
        extracted = "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages).strip()
    except Exception:
        pass
    if not extracted:
      try:
        pdftotext = shutil.which("pdftotext")
        if not pdftotext:
            pdftotext = next((str(p) for p in (
                Path("/opt/homebrew/bin/pdftotext"),
                Path("/usr/local/bin/pdftotext"),
            ) if p.is_file()), "")
        if not pdftotext:
            return ""
        child_env = os.environ.copy()
        # PyInstaller library paths can break external Homebrew binaries.
        for variable in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "LD_LIBRARY_PATH"):
            original = child_env.get(f"{variable}_ORIG")
            if original is not None:
                child_env[variable] = original
            else:
                child_env.pop(variable, None)
        result = subprocess.run(
            [pdftotext, "-layout", str(path), "-"],
            capture_output=True, text=True, timeout=20, check=False, env=child_env,
        )
        extracted = result.stdout.strip() if result.returncode == 0 else ""
      except Exception:
        extracted = ""
    if extracted and cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(extracted, encoding="utf-8")
        except Exception:
            pass
    return extracted


def _managed_pdf_path(path: Path) -> Path:
    """Stable internal mirror used when macOS denies packaged-app folder access."""
    try:
        stat = path.stat()
        key = hashlib.sha1(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
    except OSError:
        key = hashlib.sha1(str(path).encode()).hexdigest()
    return Path(config.DATA_DIR) / "pdf_library" / f"{key}.pdf"


def _read_pdf_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as original:
        try:
            return _managed_pdf_path(path).read_bytes()
        except OSError:
            raise original


@router.get("/{application_id}/context")
async def get_application_context(application_id: str, file_index: int = Query(0, ge=0)):
    """Return the selected historical PDF as usable Talk/Cover Letter context."""
    with _LOCK:
        app = _load()["applications"].get(application_id)
    if not app:
        raise HTTPException(404, "Application not found")
    pdfs = [Path(item.get("path", "")) for item in app.get("files", [])
            if Path(item.get("path", "")).suffix.casefold() == ".pdf"]
    if file_index >= len(pdfs) or not pdfs[file_index].is_file():
        raise HTTPException(404, "Resume file is no longer available")
    resume_text = _extract_pdf_text(pdfs[file_index])
    if not resume_text:
        raise HTTPException(422, "Text could not be extracted from this PDF")
    return {
        "id": app["id"], "company": app.get("company", ""), "role": app.get("role", ""),
        "resume_text": resume_text, "jd_text": app.get("jd_text", ""),
        "job_url": app.get("job_url", ""), "notes": app.get("notes", ""),
        "file_index": file_index, "file_name": pdfs[file_index].name,
    }


@router.post("/{application_id}/reveal")
async def reveal_resume(application_id: str, file_index: int = Query(0, ge=0)):
    with _LOCK:
        app = _load()["applications"].get(application_id)
    if not app:
        raise HTTPException(404, "Application not found")
    pdfs = [Path(item.get("path", "")) for item in app.get("files", [])
            if Path(item.get("path", "")).suffix.casefold() == ".pdf"]
    if file_index >= len(pdfs) or not pdfs[file_index].is_file():
        raise HTTPException(404, "Resume file is no longer available")
    try:
        subprocess.Popen(["open", "-R", str(pdfs[file_index])])
    except OSError as exc:
        raise HTTPException(500, f"Could not reveal file: {exc}") from exc
    return {"ok": True, "path": str(pdfs[file_index])}
