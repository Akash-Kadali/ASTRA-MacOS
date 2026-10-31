"""Explainable job decisions, corrections, packages, review, and recovery."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import Response

from backend.api import applications, jobright, jobright_agent
from backend.core import config

router = APIRouter(prefix="/api/decision-center", tags=["decision-center"])
STORE = Path(config.DATA_DIR) / "decision_center.json"
BACKUPS = Path(config.DATA_DIR) / "backups"
_APP_CACHE_STAMP = -1
_APP_CACHE: list[dict[str, Any]] = []


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _empty() -> dict[str, Any]:
    return {"version": 1, "overrides": {}, "blocked_companies": [], "timeline": {}, "saved_views": []}


def _load() -> dict[str, Any]:
    try:
        value = json.loads(STORE.read_text(encoding="utf-8"))
        return {**_empty(), **value} if isinstance(value, dict) else _empty()
    except Exception:
        return _empty()


def _save(value: dict[str, Any]) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    pending = STORE.with_suffix(".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(pending, STORE)


def _similarity(left: str, right: str) -> float:
    a, b = jobright._text_tokens(left), jobright._text_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _role_similarity(left: str, right: str) -> float:
    a, b = set(jobright._normalized_role(left).split()), set(jobright._normalized_role(right).split())
    return len(a & b) / max(1, len(a | b))


def _company_similarity(left: str, right: str) -> float:
    a, b = jobright._normalized_name(left), jobright._normalized_name(right)
    if a == b:
        return 1.0
    return 0.85 if jobright._company_match(a, b) else 0.0


def _application_candidates() -> list[dict[str, Any]]:
    global _APP_CACHE_STAMP, _APP_CACHE
    try:
        stamp = applications.STORE_PATH.stat().st_mtime_ns
    except OSError:
        stamp = 0
    if stamp == _APP_CACHE_STAMP:
        return _APP_CACHE
    result = []
    for app in applications._load().get("applications", {}).values():
        pdfs = [f for f in app.get("files", []) if str(f.get("path", "")).casefold().endswith(".pdf")]
        result.append({
            "app": app, "pdfs": pdfs,
            "url": jobright._normalized_url(app.get("job_url", "")),
            "company": jobright._normalized_name(app.get("company", "")),
            "role": jobright._normalized_role(app.get("role", "")),
            "role_tokens": set(jobright._normalized_role(app.get("role", "")).split()),
            "jd_tokens": jobright._text_tokens(app.get("jd_text", "")),
        })
    _APP_CACHE_STAMP, _APP_CACHE = stamp, result
    return result


def _best_application_match(job: dict[str, Any]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score = 0.0
    urls = {jobright._normalized_url(job.get("url", "")), jobright._normalized_url(job.get("application_url", ""))} - {""}
    job_company = jobright._normalized_name(job.get("company", ""))
    job_role = jobright._normalized_role(job.get("title", ""))
    job_role_tokens = set(job_role.split())
    job_jd_tokens = jobright._text_tokens(job.get("description", ""))
    for candidate in _application_candidates():
        app, pdfs = candidate["app"], candidate["pdfs"]
        app_url = candidate["url"]
        url_match = bool(app_url and app_url in urls)
        company_score = 1.0 if job_company == candidate["company"] else (0.85 if jobright._company_match(job_company, candidate["company"]) else 0.0)
        role_score = len(job_role_tokens & candidate["role_tokens"]) / max(1, len(job_role_tokens | candidate["role_tokens"]))
        jd_score = (len(job_jd_tokens & candidate["jd_tokens"]) / min(len(job_jd_tokens), len(candidate["jd_tokens"]))) if job_jd_tokens and candidate["jd_tokens"] else 0.0
        score = 1.0 if url_match else (0.40 * company_score + 0.35 * role_score + 0.25 * jd_score)
        is_match = url_match or (company_score > 0 and jobright._role_match(
            job_role, candidate["role"])) or (
            company_score > 0 and jd_score >= 0.85)
        if is_match and score > best_score:
            best_score = score
            best = {
                "application_id": app.get("id"), "company": app.get("company"), "role": app.get("role"),
                "status": app.get("status"), "job_url": app.get("job_url"),
                "pdf_path": pdfs[0].get("path") if pdfs else "",
                "company_similarity": round(company_score * 100),
                "role_similarity": round(role_score * 100),
                "jd_similarity": round(jd_score * 100),
                "confidence": round(score * 100),
                "reason": "Exact requisition URL" if url_match else (
                    "Same company and highly similar job description" if jd_score >= 0.85 else
                    "Matching company and role in résumé library"),
            }
    return best


def _qualification_evidence(job: dict[str, Any]) -> list[str]:
    sections = job.get("description_sections") or {}
    required = str(sections.get("required") or "")
    lines = [x.strip() for x in required.splitlines() if x.strip()]
    patterns = re.compile(
        r"ph\.?d|doctoral|pursuing|currently enrolled|working toward|graduat(?:e|ing|ion)|"
        r"citizen|permanent resident|green card|clearance|sponsor|years? of|experience",
        re.I,
    )
    return [line for line in lines if patterns.search(line)][:6]


def _eligibility(job: dict[str, Any]) -> dict[str, Any]:
    evidence = _qualification_evidence(job)
    reasons: list[str] = []
    title = str(job.get("title") or "")
    # Jobright sometimes returns a title before the full JD is fetched.  Titles
    # such as "Doctorate Degree" are already conclusive and must not sit in the
    # actionable queue while we wait for description enrichment.
    title_requires_phd = bool(re.search(r"\b(?:ph\.?d|doctorate|doctoral)\b", title, re.I)) and not bool(
        re.search(r"\b(?:ph\.?d|doctorate|doctoral)\s+(?:welcome|preferred|a plus)\b", title, re.I)
    )
    if jobright._requires_phd(job) or title_requires_phd:
        reasons.append("This role specifically requires PhD or doctoral eligibility.")
    title_requires_future_graduation = bool(re.search(
        r"\b(?:intern(?:ship)?|co-?op|summer analyst|graduation date|(?:new|university|college) grad)\b.*\b2027\b",
        title,
        re.I,
    ))
    if jobright._requires_current_student(job) or title_requires_future_graduation:
        reasons.append("This role requires current enrolment or a future graduation window.")
    required = "\n".join((job.get("description_sections") or {}).get("required", "").splitlines())
    if re.search(r"\b(?:u\.?s\.? citizen|united states citizen|green card|permanent resident)\b", required, re.I):
        reasons.append("The required qualifications contain a citizenship or permanent-residency restriction.")
    if re.search(r"\b(?:active|current)\s+(?:security\s+)?clearance\b", required, re.I):
        reasons.append("The role requires an active security clearance.")
    if re.search(r"(?:no|not).{0,35}(?:visa|immigration|sponsor)|(?:sponsor|visa).{0,35}(?:not available|unavailable)", required, re.I):
        reasons.append("The role states that employment sponsorship is unavailable.")
    exp = [int(x) for x in re.findall(r"\b(\d{1,2})\s*\+?\s*(?:-|to)?\s*\d*\s*years?", required, re.I)]
    if exp and min(exp) > 2:
        reasons.append(f"The role requires at least {min(exp)} years of experience; the saved profile uses 2 years.")
    if reasons:
        return {"status": "not_eligible", "score": 0, "reasons": reasons, "evidence": evidence}
    if len(str(job.get("description") or "")) < 300:
        return {"status": "possibly_eligible", "score": 50, "reasons": ["Full qualifications have not been loaded yet."], "evidence": evidence}
    return {"status": "eligible", "score": 100, "reasons": ["No saved-profile eligibility conflicts were found."], "evidence": evidence}


def _decorate(job: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    item = dict(job)
    item["category"], item["relevant"] = jobright._classify_role(str(item.get("title", "")))
    match = _best_application_match(item)
    override = dict(state.get("overrides", {}).get(str(item.get("id")), {}))
    blocked = jobright._normalized_name(str(item.get("company", ""))) in set(state.get("blocked_companies", []))
    eligibility = _eligibility(item)
    decision = "actionable"
    reason = "Eligible and no matching application was found."
    if blocked:
        decision, reason = "hidden", "Company is on your Never Show list."
    elif override.get("action") in {"applied", "hide", "review_rejected"}:
        decision = "applied" if override["action"] == "applied" else "hidden"
        reason = override.get("reason") or f"You marked this job as {override['action'].replace('_', ' ')}."
    elif override.get("action") == "review_approved":
        decision, reason, match = "actionable", "You reviewed and approved this job for preparation.", None
    elif override.get("action") == "not_applied":
        decision, reason, match = "actionable", "You confirmed that this job was not applied.", None
    elif eligibility["status"] == "not_eligible":
        decision, reason = "ineligible", eligibility["reasons"][0]
    elif match and match["confidence"] >= 65:
        decision, reason = "applied", match["reason"]
    elif match:
        decision, reason = "review", "A possible résumé/application match needs your review."
    elif eligibility["status"] == "possibly_eligible":
        decision, reason = "actionable", "Possibly eligible; load the full description before preparing the application."
    item.update({"decision": decision, "decision_reason": reason, "match_details": match,
                 "eligibility": eligibility, "override": override,
                 "timeline": state.get("timeline", {}).get(str(item.get("id")), [])})
    return item


def _group_duplicate_jobs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for item in items:
        url = jobright._normalized_url(item.get("application_url") or item.get("url") or "")
        key = f"url:{url}" if url else "text:" + "|".join((
            jobright._normalized_name(str(item.get("company", ""))),
            jobright._normalized_role(str(item.get("title", ""))),
            jobright._normalized_name(str(item.get("location", ""))),
        ))
        existing = by_key.get(key)
        if existing is None:
            item["duplicate_count"] = 1
            item["duplicate_ids"] = [str(item.get("id", ""))]
            by_key[key] = item
            grouped.append(item)
        else:
            existing["duplicate_count"] += 1
            existing["duplicate_ids"].append(str(item.get("id", "")))
    return grouped


@router.get("")
def list_decisions(
    view: str = Query("actionable"), q: str = Query(""), category: str = Query("all"),
    limit: int = Query(5000, ge=1, le=10000),
) -> dict[str, Any]:
    state = _load()
    raw_items = [x for x in jobright._read_store().get("items", []) if jobright._classify_role(str(x.get("title", "")))[1]]
    decorated = [_decorate(x, state) for x in raw_items]
    decorated.sort(key=lambda x: (jobright._posted_age_minutes(x.get("posted", "")), -jobright._timestamp(x.get("captured_at", ""))))
    items = _group_duplicate_jobs(decorated)
    counts = {name: sum(x["decision"] == name for x in items) for name in ("actionable", "review", "applied", "ineligible", "hidden")}
    if view != "all":
        items = [x for x in items if x["decision"] == view]
    if category != "all":
        items = [x for x in items if x["category"] == category]
    if q.strip():
        needle = q.casefold()
        items = [x for x in items if needle in f"{x.get('company','')} {x.get('title','')} {x.get('location','')}".casefold()]
    return {"items": items[:limit], "total": len(items), "raw_total": len(decorated), "counts": counts,
            "saved_views": state.get("saved_views", [])}


def _record_application(job: dict[str, Any], status: str = "applied", confirmation: dict[str, Any] | None = None) -> str:
    confirmation = confirmation or {}
    with applications._LOCK:
        store = applications._load()
        app_id = applications._key(str(job.get("company", "")), str(job.get("title", "")))
        existing = store["applications"].get(app_id, {})
        store["applications"][app_id] = {
            **existing, "id": app_id, "company": job.get("company", ""), "role": job.get("title", ""),
            "status": status, "applied_at": str(confirmation.get("submitted_at") or existing.get("applied_at") or _now()),
            "applied_date_source": "Decision Centre confirmation", "source": "Jobright",
            "job_url": job.get("application_url") or job.get("url") or "",
            "requisition_id": str(confirmation.get("requisition_id") or existing.get("requisition_id") or ""),
            "jd_text": job.get("description") or existing.get("jd_text") or "",
            "files": existing.get("files", []), "first_seen_at": existing.get("first_seen_at") or _now(),
            "updated_at": _now(), "notes": existing.get("notes", ""),
        }
        applications._save(store)
    return app_id


@router.post("/jobs/{job_id}/action")
def job_action(job_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    allowed = {"applied", "not_applied", "hide", "never_company", "unhide", "unblock_company", "review_approved", "review_rejected"}
    action = str(payload.get("action") or "")
    if action not in allowed:
        raise HTTPException(400, "Unsupported decision action")
    job = next((x for x in jobright._read_store().get("items", []) if str(x.get("id")) == job_id), None)
    if not job:
        raise HTTPException(404, "Job not found")
    state = _load()
    if action == "never_company":
        blocked = set(state.get("blocked_companies", []))
        blocked.add(jobright._normalized_name(str(job.get("company", ""))))
        state["blocked_companies"] = sorted(blocked)
    elif action == "unblock_company":
        blocked = set(state.get("blocked_companies", []))
        blocked.discard(jobright._normalized_name(str(job.get("company", ""))))
        state["blocked_companies"] = sorted(blocked)
        state.get("overrides", {}).pop(job_id, None)
    elif action == "unhide":
        state.get("overrides", {}).pop(job_id, None)
    else:
        state.setdefault("overrides", {})[job_id] = {"action": action, "at": _now(), "reason": str(payload.get("reason") or "")}
    event = {"at": _now(), "action": action, "note": str(payload.get("reason") or "")}
    state.setdefault("timeline", {}).setdefault(job_id, []).insert(0, event)
    application_id = _record_application(job, confirmation=payload) if action == "applied" else ""
    _save(state)
    return {"ok": True, "job_id": job_id, "action": action, "application_id": application_id,
            "item": _decorate(job, state)}


@router.post("/jobs/{job_id}/prepare")
def prepare_job(job_id: str) -> dict[str, Any]:
    result = jobright_agent.dry_run(job_id)
    state = _load()
    state.setdefault("timeline", {}).setdefault(job_id, []).insert(0, {"at": _now(), "action": "prepare_started", "note": "Agent dry run started; submission is disabled."})
    _save(state)
    return result


@router.get("/review")
def review_queue() -> dict[str, Any]:
    statuses = {"ready_to_submit", "needs_attention", "error", "review_approved", "review_rejected"}
    events = [x for x in jobright_agent._load(jobright_agent.LOG, []) if x.get("status") in statuses]
    return {"items": events[:250], "total": len(events)}


@router.get("/jobs/{job_id}/integrity")
def integrity(job_id: str) -> dict[str, Any]:
    job = next((x for x in jobright._read_store().get("items", []) if str(x.get("id")) == job_id), None)
    if not job:
        raise HTTPException(404, "Job not found")
    match = _best_application_match(job)
    path = Path(str((match or {}).get("pdf_path") or ""))
    result = {"has_match": bool(match), "pdf_path": str(path) if path else "", "available": False,
              "valid_pdf": False, "size": 0, "sha256": "", "issues": []}
    if path and path.is_file():
        try:
            data = applications._read_pdf_bytes(path)
            result.update(available=True, valid_pdf=data.startswith(b"%PDF"), size=len(data), sha256=hashlib.sha256(data).hexdigest())
            if not result["valid_pdf"]:
                result["issues"].append("File does not have a valid PDF header.")
        except OSError as exc:
            result["issues"].append(str(exc))
    elif match:
        result["issues"].append("Matched résumé file is missing.")
    else:
        result["issues"].append("No matching résumé is attached.")
    return result


@router.get("/integrity")
def integrity_audit() -> dict[str, Any]:
    missing, invalid, misnamed, hashes = [], [], [], {}
    checked = 0
    for app in applications._load().get("applications", {}).values():
        company = jobright._normalized_name(str(app.get("company", "")))
        role_words = set(jobright._normalized_role(str(app.get("role", ""))).split())
        for entry in app.get("files", []):
            path = Path(str(entry.get("path") or ""))
            if path.suffix.casefold() != ".pdf":
                continue
            checked += 1
            if not path.is_file():
                missing.append({"application_id": app.get("id"), "path": str(path)})
                continue
            try:
                data = applications._read_pdf_bytes(path)
            except OSError as exc:
                invalid.append({"application_id": app.get("id"), "path": str(path), "error": str(exc)})
                continue
            if not data.startswith(b"%PDF"):
                invalid.append({"application_id": app.get("id"), "path": str(path), "error": "Invalid PDF header"})
            digest = hashlib.sha256(data).hexdigest()
            hashes.setdefault(digest, []).append(str(path))
            stem = jobright._normalized_name(path.stem)
            if company and company not in stem and not (role_words & set(stem.split())):
                misnamed.append({"application_id": app.get("id"), "path": str(path), "company": app.get("company"), "role": app.get("role")})
    duplicates = [{"sha256": digest, "paths": paths} for digest, paths in hashes.items() if len(paths) > 1]
    return {"checked": checked, "missing": missing, "invalid": invalid, "misnamed": misnamed,
            "duplicates": duplicates, "healthy": checked - len(missing) - len(invalid)}


@router.post("/jobs/{job_id}/timeline")
def add_timeline_event(job_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    action = str(payload.get("action") or "note").strip()
    event = {"at": str(payload.get("at") or _now()), "action": action, "note": str(payload.get("note") or "")}
    state = _load()
    state.setdefault("timeline", {}).setdefault(job_id, []).insert(0, event)
    _save(state)
    return event


@router.post("/jobs/{job_id}/follow-up")
async def generate_follow_up(job_id: str) -> dict[str, Any]:
    job = next((x for x in jobright._read_store().get("items", []) if str(x.get("id")) == job_id), None)
    if not job:
        raise HTTPException(404, "Job not found")
    from backend.api.talk import ANSWER_MODEL, _call_openai
    from backend.api.superhuman import rewrite_single
    draft = await _call_openai(
        "Write a short, factual job-application follow-up. Do not invent contacts or events.",
        f"Write a polite 70-100 word follow-up for the {job.get('title')} application at {job.get('company')}. "
        "Say I remain interested and ask if there is any update. Do not claim an interview or recruiter conversation.\n\n"
        f"JOB CONTEXT:\n{str(job.get('description') or job.get('card_text') or '')[:2500]}",
        model=ANSWER_MODEL, temperature=0.3, max_completion_tokens=300,
    )
    final = await rewrite_single(draft, "application", "balanced", {"no_fabrication": True}, False, 3000)
    state = _load()
    result = {"text": final, "generated_at": _now(), "send_automatically": False}
    state.setdefault("followups", {})[job_id] = result
    state.setdefault("timeline", {}).setdefault(job_id, []).insert(0, {"at": _now(), "action": "follow_up_drafted", "note": "Draft only; not sent."})
    _save(state)
    return result


@router.post("/jobs/{job_id}/interview-kit")
async def generate_interview_kit(job_id: str) -> dict[str, Any]:
    job = next((x for x in jobright._read_store().get("items", []) if str(x.get("id")) == job_id), None)
    if not job:
        raise HTTPException(404, "Job not found")
    match = _best_application_match(job) or {}
    resume_text = ""
    pdf = Path(str(match.get("pdf_path") or ""))
    if pdf.is_file():
        resume_text = applications._extract_pdf_text(pdf)
    from backend.api.talk import ANSWER_MODEL, _call_openai
    raw = await _call_openai(
        "Create a factual interview preparation kit. Return only valid JSON.",
        "Return JSON with arrays behavioural, technical, system_design, and resume_questions. "
        "Each array must contain 5 concise questions grounded in the job and résumé. Do not invent experience.\n\n"
        f"JOB:\n{str(job.get('description') or job.get('card_text') or '')[:5000]}\n\nRESUME:\n{resume_text[:5000]}",
        model=ANSWER_MODEL, temperature=0.25, max_completion_tokens=1800,
    )
    try:
        kit = json.loads(re.search(r"\{[\s\S]*\}", raw).group(0))
    except Exception as exc:
        raise HTTPException(502, f"Interview kit returned invalid JSON: {exc}") from exc
    result = {"job_id": job_id, "company": job.get("company"), "role": job.get("title"), "generated_at": _now(), **kit}
    state = _load()
    state.setdefault("interview_kits", {})[job_id] = result
    _save(state)
    return result


@router.get("/jobs/{job_id}/package")
def package(job_id: str):
    job = next((x for x in jobright._read_store().get("items", []) if str(x.get("id")) == job_id), None)
    if not job:
        raise HTTPException(404, "Job not found")
    state = _load()
    decorated = _decorate(job, state)
    match = decorated.get("match_details") or {}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("job-description.txt", str(job.get("description") or job.get("card_text") or ""))
        archive.writestr("job-link.txt", str(job.get("application_url") or job.get("url") or ""))
        archive.writestr("decision-report.json", json.dumps(decorated, ensure_ascii=False, indent=2))
        archive.writestr("application-answers.json", json.dumps({"job_id": job_id, "answers": [], "note": "Answers are added when the agent prepares this application."}, indent=2))
        pdf = Path(str(match.get("pdf_path") or ""))
        if pdf.is_file():
            archive.writestr("resume.pdf", applications._read_pdf_bytes(pdf))
        context_key = f"{re.sub(r'[^A-Za-z0-9]+', '_', str(job.get('company',''))).strip('_')}__{re.sub(r'[^A-Za-z0-9]+', '_', str(job.get('title',''))).strip('_')}"
        ctx = config.get_contexts_dir() / f"{context_key}.json"
        if ctx.is_file():
            raw = json.loads(ctx.read_text(encoding="utf-8"))
            cover = raw.get("cover_letter") or {}
            cover_path = Path(str(cover.get("pdf_path") or ""))
            if cover_path.is_file():
                archive.writestr("cover-letter.pdf", cover_path.read_bytes())
            if cover.get("body_text"):
                archive.writestr("cover-letter.txt", str(cover["body_text"]))
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{job.get('company')}-{job.get('title')}-application-package").strip("-") + ".zip"
    return Response(buffer.getvalue(), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.post("/saved-views")
def save_view(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "View name is required")
    state = _load()
    view = {"id": hashlib.sha1(name.casefold().encode()).hexdigest()[:12], "name": name,
            "filters": dict(payload.get("filters") or {}), "updated_at": _now()}
    state["saved_views"] = [x for x in state.get("saved_views", []) if x.get("id") != view["id"]] + [view]
    _save(state)
    return view


@router.get("/analytics")
def analytics() -> dict[str, Any]:
    apps = list(applications._load().get("applications", {}).values())
    stats = applications._stats(apps)
    by_source: dict[str, int] = {}
    by_role: dict[str, int] = {}
    for app in apps:
        source = str(app.get("source") or "Unknown")
        by_source[source] = by_source.get(source, 0) + 1
        role = jobright._classify_role(str(app.get("role") or ""))[0]
        by_role[role] = by_role.get(role, 0) + 1
    return {"stats": stats, "by_source": by_source, "by_role": by_role}


@router.post("/backup")
def backup() -> dict[str, Any]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = BACKUPS / stamp
    folder.mkdir(parents=True, exist_ok=False)
    sources = [applications.STORE_PATH, jobright.STORE, jobright_agent.SETTINGS, jobright_agent.LOG, STORE]
    copied = []
    for source in sources:
        if source.is_file():
            shutil.copy2(source, folder / source.name)
            copied.append(source.name)
    return {"ok": True, "path": str(folder), "files": copied, "created_at": _now()}


@router.get("/backups")
def list_backups() -> dict[str, Any]:
    items = []
    if BACKUPS.is_dir():
        for folder in sorted((x for x in BACKUPS.iterdir() if x.is_dir()), reverse=True):
            items.append({"id": folder.name, "path": str(folder), "files": sorted(x.name for x in folder.iterdir() if x.is_file())})
    return {"items": items, "total": len(items)}
