"""Daily Jobright -> ASTRA resume -> ATS application worker."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from fastapi import APIRouter, Body, HTTPException

from backend.api import applications, jobright
from backend.core.config import DATA_DIR
from backend.core import operational_store

router = APIRouter(prefix="/api/jobright/agent", tags=["jobright-agent"])
SETTINGS = Path(DATA_DIR) / "jobright_agent.json"
RESUMES = Path.home() / "Desktop" / "ASTRA Resumes"
LEGACY_RESUMES = Path(DATA_DIR) / "agent_resumes"
DRY_RUNS = Path(DATA_DIR) / "agent_dry_runs"
LOG = Path(DATA_DIR) / "jobright_agent_log.json"
ANSWER_CACHE = Path(DATA_DIR) / "jobright_application_answer_cache.json"
_guard = threading.Lock()
_file_guard = threading.RLock()
_stop = threading.Event()
_runtime: dict[str, Any] = {"running": False, "current_job": "", "message": "Idle", "last_run": None}

DEFAULTS: dict[str, Any] = {
    "enabled": False, "run_at": "09:00", "daily_limit": 0, "auto_submit": False,
    "company_question_ai": True,
    "profile": {
        "first_name": "Sri Akash", "last_name": "Kadali", "email": "srikadaliakash@gmail.com",
        "phone": "", "location": "", "linkedin": "", "portfolio": "",
        "authorized_to_work": "", "requires_sponsorship": "", "years_experience": "",
        "immigration_status": "", "opt_expiration": "", "citizenship": "",
        "opt_start": "", "stem_opt_eligible": "", "stem_opt_start": "", "stem_opt_end": "",
        "other_citizenship_or_permanent_residency": "", "over_18": "",
        "willing_to_relocate": "", "onsite_hybrid_ok": "", "available_start": "",
        "salary_strategy": "", "non_compete": "", "contact_current_employer": "",
        "prior_company_history": "", "job_source": "", "employee_referral": "",
        "gender": "", "sexual_orientation": "", "cisgender": "", "pronouns": "",
        "hispanic_latino": "", "race_ethnicity": "", "veteran": "",
        "disability": "", "security_clearance": "", "accept_truth_declaration": "",
        "current_employer": "", "current_title": "", "current_employment_start": "",
        "languages": [], "interview_recording_consent": "", "marketing_consent": "",
        "future_opportunity_retention": "", "criminal_conviction": "",
        "sponsorship_explanation": "", "masters_completed": "", "currently_student": "",
        "profile_complete": False,
        "education": [],
    },
}

HIREX_APPLICATION_STYLE = """Answer this job application question using only facts in the resume,
saved candidate profile, and job description. Use simple, natural Indian English at approximately
IELTS Band 6.5. Sound like a real applicant, not AI. Be direct and concise (normally 2-5 sentences),
avoid buzzwords, and do not use em dashes. Never invent skills, experience, metrics, preferences, or
personal facts. If the answer is not supported, return exactly NEEDS_USER_INPUT."""

_FACTUAL_OR_SENSITIVE_QUESTION = re.compile(
    r"authori[sz]|sponsor|visa|immigra|citizen|resident|salary|compensation|pay|"
    r"gender|race|ethni|hispanic|veteran|disab|pronoun|criminal|convict|"
    r"address|phone|e.?mail|first name|last name|full name|legal name|age|birth|"
    r"start date|clearance|consent|record",
    re.I,
)


def _load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save(path: Path, value: Any) -> None:
    """Crash-safe JSON persistence for scheduler state and application history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    with _file_guard:
        pending.write_text(payload, encoding="utf-8")
        os.replace(pending, path)


def settings() -> dict[str, Any]:
    saved = _load(SETTINGS, {})
    return {**DEFAULTS, **saved, "profile": {**DEFAULTS["profile"], **saved.get("profile", {})}}


def _event(job: dict[str, Any], status: str, message: str, **extra: Any) -> None:
    with _file_guard:
        events = _load(LOG, [])
        events.insert(0, {"at": datetime.now().astimezone().isoformat(), "job_id": job.get("id"),
                          "company": job.get("company"), "title": job.get("title"),
                          "status": status, "message": message, **extra})
        _save(LOG, events[:1000])
    job.update(agent_status=status, agent_message=message, agent_updated_at=datetime.now().astimezone().isoformat(), **extra)
    jobright._write_store([job])
    operational_store.save_checkpoint(
        str(job.get("id") or ""), status=status,
        step=status, message=message,
        url=str(job.get("application_url") or job.get("url") or ""),
        resume_path=str(extra.get("resume_path") or job.get("agent_resume_path") or ""),
        evidence_path=str(extra.get("evidence_path") or ""),
        unanswered_fields=extra.get("unanswered_fields", []),
        completed_fields=extra.get("completed_fields", []),
        answers=extra.get("answers", {}),
    )
    operational_store.add_activity("agent", status, message, str(job.get("id") or ""),
                                   company=job.get("company"), role=job.get("title"), **extra)


def _ensure_description(job: dict[str, Any], browser_context: Any = None) -> dict[str, Any]:
    if len(job.get("description", "")) >= 300 and job.get("application_url"):
        return job
    if browser_context is not None:
        return jobright.load_details_in_context(str(job["id"]), browser_context)
    return jobright.load_details(str(job["id"]))


def _resume_filename(job: dict[str, Any]) -> str:
    """Human-readable, Finder-safe filename: candidate - company - role.pdf."""
    parts = ["Sri Akash Kadali", str(job.get("company") or "Company"),
             str(job.get("title") or "Role")]
    cleaned = [re.sub(r"\s+", " ", re.sub(r'[/\\:*?"<>|]', "-", part)).strip(" .-")
               for part in parts]
    return " - ".join(cleaned)[:220].rstrip(" .-") + ".pdf"


def _generate_resume(job: dict[str, Any]) -> Path:
    RESUMES.mkdir(parents=True, exist_ok=True)
    target = RESUMES / _resume_filename(job)
    if target.exists() and target.stat().st_size > 10000:
        job["agent_resume_path"] = str(target)
        applications.record_generated(
            str(job.get("company", "")), str(job.get("title", "")), target,
            str(job.get("application_url") or job.get("url") or ""), "Jobright", str(job.get("description") or ""))
        return target
    previous = Path(str(job.get("agent_resume_path") or LEGACY_RESUMES / f"{job['id']}.pdf"))
    if previous.exists() and previous.stat().st_size > 10000:
        shutil.copy2(previous, target)
        job["agent_resume_path"] = str(target)
        jobright._write_store([job])
        applications.record_generated(
            str(job.get("company", "")), str(job.get("title", "")), target,
            str(job.get("application_url") or job.get("url") or ""), "Jobright", str(job.get("description") or ""))
        return target
    response = requests.post(
        "http://127.0.0.1:8000/api/optimize/run",
        data={"jd_text": job["description"], "job_description": job["description"],
              "use_humanize": "true", "latex_safe": "true"},
        timeout=900,
    )
    response.raise_for_status()
    result = response.json()
    encoded = result.get("pdf_base64_humanized") or result.get("pdf_base64")
    if not encoded:
        raise RuntimeError("ASTRA did not return a résumé PDF.")
    target.write_bytes(base64.b64decode(encoded.split("base64,")[-1]))
    job["agent_resume_path"] = str(target)
    job["agent_fit_score"] = result.get("rating_score") or result.get("coverage_ratio")
    _event(job, "resume_ready", "Tailored résumé generated.", resume_path=str(target))
    applications.record_generated(
        str(job.get("company", "")), str(job.get("title", "")), target,
        str(job.get("application_url") or job.get("url") or ""), "Jobright", str(job.get("description") or ""))
    return target


def _field_value(label: str, profile: dict[str, Any], job: dict[str, Any]) -> str:
    key = label.casefold()
    rules = (
        (r"preferred.*first", "first_name"), (r"first.?name", "first_name"),
        (r"preferred.*(?:last|surname)", "last_name"), (r"last.?name|surname", "last_name"),
        (r"e.?mail", "email"), (r"phone|mobile", "phone"), (r"city|location|address", "location"),
        (r"linkedin", "linkedin"), (r"portfolio|website|github", "portfolio"),
        (r"years?.*experience", "years_experience"),
        (r"current (?:company|employer)", "current_employer"),
        (r"current (?:job )?title|current position", "current_title"),
        (r"country.*citizenship|citizenship.*country", "citizenship"),
        (r"pronouns?", "pronouns"),
        (r"how did you hear|how.*hear about", "job_source"),
        (r"earliest.*start|available.*start|start date", "available_start"),
    )
    for pattern, name in rules:
        if re.search(pattern, key):
            return str(profile.get(name, ""))
    if re.search(r"salary expectation|desired (?:base )?salary|compensation", key):
        location = f"{job.get('location', '')} {job.get('card_text', '')}".casefold()
        return "150000" if re.search(r"california|\bca\b", location) else "130000"
    if re.search(r"zip|postal", key):
        match = re.search(r"\b\d{5}(?:-\d{4})?\b", str(profile.get("location", "")))
        return match.group(0) if match else ""
    return ""


def _select_candidates(label: str, profile: dict[str, Any]) -> list[str]:
    """Return preferred visible option labels for an ATS combobox question."""
    key = " ".join(label.casefold().split())
    rules: tuple[tuple[str, list[str]], ...] = (
        (r"country", ["United States", "United States of America"]),
        (r"location|city", ["Palo Alto, California, United States", "Palo Alto, CA", "Palo Alto"]),
        (r"eligible to work|authori[sz]ed to work", ["Yes"] if profile.get("authorized_to_work") else []),
        (r"sponsor|immigration.*support", ["Yes"] if profile.get("requires_sponsorship") else []),
        (r"currently an employee|currently.*contractor", ["No"]),
        (r"previously worked", ["No"]),
        (r"gender", ["Male"] if str(profile.get("gender", "")).casefold() in {"male", "man", "boy"} else []),
        (r"hispanic|latino", ["No", "No, not Hispanic or Latino"]),
        (r"race|ethnicity", ["Asian"] if re.search(r"asian|south asian|indian", str(profile.get("race_ethnicity", "")), re.I) else []),
        (r"veteran", ["I am not a protected veteran", "Not a protected veteran", "No"]),
        (r"disabil", ["No, I do not have a disability", "No"]),
    )
    for pattern, candidates in rules:
        if re.search(pattern, key):
            return candidates
    return []


def _question_for(node: Any) -> str:
    return str(node.evaluate("""el => {
      const explicit = el.labels ? [...el.labels].map(x => x.innerText).join(' ') : '';
      const labelledBy = (el.getAttribute('aria-labelledby') || '').split(/\\s+/)
        .map(id => document.getElementById(id)?.innerText || '').join(' ');
      const aria = el.getAttribute('aria-label') || '';
      if (explicit.trim()) return explicit;
      if (aria.trim()) return aria;
      if (labelledBy.trim()) return labelledBy;
      const container = el.closest('fieldset, .application-question, .field, .form-field, [class*=field]');
      return [el.placeholder, container ? container.innerText : '', el.name].filter(Boolean).join(' ');
    }"""))


def _fill_comboboxes(page: Any, profile: dict[str, Any]) -> list[str]:
    """Fill React-style ATS comboboxes and return questions still unanswered."""
    unanswered: list[str] = []
    for node in page.locator('[role="combobox"]:visible').all():
        try:
            question = _question_for(node)
            metadata = node.evaluate("""el => ({
              required: el.required || el.getAttribute('aria-required') === 'true',
              text: (el.parentElement?.parentElement?.innerText || '').trim()
            })""")
            if metadata.get("text") and "select..." not in metadata["text"].casefold():
                continue
            candidates = _select_candidates(question, profile)
            if not candidates:
                if metadata.get("required"):
                    unanswered.append(" ".join(question.split())[:180] or "Unlabelled selection")
                continue
            node.click()
            page.wait_for_timeout(150)
            options = page.get_by_role("option")
            option_texts = [" ".join(text.split()) for text in options.all_inner_texts()]
            match_index = next((i for candidate in candidates for i, text in enumerate(option_texts)
                                if text.casefold() == candidate.casefold()), None)
            if match_index is None:
                match_index = next((i for candidate in candidates for i, text in enumerate(option_texts)
                                    if candidate.casefold() in text.casefold()), None)
            if match_index is None:
                # Autocomplete controls (notably Greenhouse's city field) do
                # not expose options until a search term is entered.
                node.fill(candidates[0])
                page.wait_for_timeout(500)
                option_texts = [" ".join(text.split()) for text in options.all_inner_texts()]
                match_index = next((i for candidate in candidates for i, text in enumerate(option_texts)
                                    if candidate.casefold() in text.casefold() or
                                    text.casefold() in candidate.casefold()), None)
            if match_index is None:
                if metadata.get("required"):
                    unanswered.append(" ".join(question.split())[:180] or "Unlabelled selection")
                node.press("Escape")
                continue
            options.nth(match_index).click()
            page.wait_for_timeout(150)
        except Exception:
            if locals().get("metadata", {}).get("required"):
                unanswered.append(" ".join(locals().get("question", "Selection").split())[:180])
    return unanswered


def _pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
        return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    except Exception:
        return ""


def _hirex_answer(question: str, job: dict[str, Any], resume: Path, profile: dict[str, Any]) -> str:
    """Use Talk to Hirex for open-ended application questions, never identity/legal facts."""
    question = " ".join(question.split()).strip()
    if not question or _FACTUAL_OR_SENSITIVE_QUESTION.search(question):
        return ""
    resume_text = _pdf_text(resume)
    if not resume_text:
        return ""
    cache_key = hashlib.sha256("\x1f".join([
        question.casefold(), str(job.get("description", "")), resume_text,
        json.dumps(profile, sort_keys=True, ensure_ascii=False),
    ]).encode("utf-8")).hexdigest()
    with _file_guard:
        answer_cache = _load(ANSWER_CACHE, {})
        cached = answer_cache.get(cache_key)
    if isinstance(cached, dict) and cached.get("answer"):
        return str(cached["answer"])
    response = requests.post(
        "http://127.0.0.1:8000/api/talk/application-answer",
        json={
            "jd_text": job.get("description", ""),
            "question": question,
            "resume_text": resume_text,
            "profile": profile,
        },
        timeout=240,
    )
    response.raise_for_status()
    answer = " ".join(str(response.json().get("answer", "")).split()).strip()
    if not answer or "NEEDS_USER_INPUT" in answer or len(answer) > 1800:
        return ""
    with _file_guard:
        answer_cache = _load(ANSWER_CACHE, {})
        answer_cache[cache_key] = {
            "answer": answer, "created_at": datetime.now().astimezone().isoformat(),
            "job_id": job.get("id"), "question": question,
        }
        # Bound local storage while retaining recent answers across app restarts.
        if len(answer_cache) > 2000:
            ordered = sorted(answer_cache.items(), key=lambda item: item[1].get("created_at", ""), reverse=True)
            answer_cache = dict(ordered[:2000])
        _save(ANSWER_CACHE, answer_cache)
    return answer


def _apply(job: dict[str, Any], resume: Path, cfg: dict[str, Any], browser_context: Any = None,
           evidence_path: Path | None = None) -> tuple[str, str, dict[str, Any]]:
    from playwright.sync_api import sync_playwright
    profile = cfg["profile"]
    own_playwright = own_context = False
    playwright = None
    context = browser_context
    if context is None:
        playwright = sync_playwright().start()
        own_playwright = own_context = True
        context = playwright.chromium.launch_persistent_context(
            str(jobright.PROFILE), channel="chrome", headless=True)
    page = context.new_page()
    completed_fields: list[str] = []
    generated_answers: dict[str, str] = {}
    try:
        page.goto(job["application_url"], wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(3500)
        for node in page.locator('input:not([type=hidden]), textarea').all():
            try:
                if not node.is_visible() or node.is_disabled():
                    continue
                kind = (node.get_attribute("type") or "text").lower()
                if node.get_attribute("role") == "combobox":
                    continue
                if kind == "file":
                    node.set_input_files(str(resume))
                    completed_fields.append(_question_for(node) or "Résumé upload")
                    continue
                if kind in {"checkbox", "radio", "submit", "button"} or node.input_value().strip():
                    continue
                label = _question_for(node)
                value = _field_value(label, profile, job)
                if value:
                    node.fill(value)
                    completed_fields.append(label[:180])
            except Exception:
                continue
        if cfg.get("company_question_ai"):
            for node in page.locator('textarea[required]:visible, input[required][type=text]:visible').all():
                try:
                    if node.input_value().strip() or node.is_disabled():
                        continue
                    question = _question_for(node)
                    answer = _hirex_answer(question, job, resume, profile)
                    if answer:
                        node.fill(answer)
                        completed_fields.append(question[:180])
                        generated_answers[question[:300]] = answer
                except Exception:
                    continue
        unresolved = _fill_comboboxes(page, profile)
        if unresolved:
            # Selecting an option rerenders Greenhouse's React control. A
            # second pass resolves any controls whose locator was detached by
            # an earlier selection in the same form.
            page.wait_for_timeout(250)
            unresolved = _fill_comboboxes(page, profile)
        unresolved.extend(page.locator('input[required]:visible:not([role=combobox]), textarea[required]:visible, select[required]:visible').evaluate_all(
            """els => els.filter(e => {
              if (e.type==='checkbox' || e.type==='radio') return !e.checked;
              return !(e.value || '').trim();
            }).map(e => {
              const labels=e.labels ? [...e.labels].map(x=>x.innerText.trim()).filter(Boolean) : [];
              const box=e.closest('fieldset, .application-question, .field, .form-field');
              return labels.join(' ') || e.getAttribute('aria-label') || e.placeholder ||
                     (box ? box.innerText.trim().slice(0,180) : '') || e.name || e.type;
            })"""
        ))
        if evidence_path is not None:
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(evidence_path), full_page=True)
        if unresolved:
            return "needs_attention", "Required questions need answers: " + ", ".join(unresolved[:6]), {
                "completed_fields": completed_fields, "unanswered_fields": unresolved,
                "answers": generated_answers,
            }
        if not cfg.get("auto_submit"):
            return "ready_to_submit", "Application is filled and waiting for final review.", {
                "completed_fields": completed_fields, "unanswered_fields": [],
                "answers": generated_answers,
            }
        buttons = page.get_by_role("button", name=re.compile(r"submit application|submit|apply now", re.I))
        if not buttons.count():
            return "needs_attention", "The ATS submit button was not recognized.", {
                "completed_fields": completed_fields, "unanswered_fields": ["Submit button was not recognized"],
                "answers": generated_answers,
            }
        buttons.last.click()
        page.wait_for_timeout(6000)
        text = page.locator("body").inner_text()[:8000]
        if re.search(r"thank you|application (?:was )?submitted|successfully applied|we received your application", text, re.I):
            return "applied", "Application submitted successfully.", {"completed_fields": completed_fields, "answers": generated_answers}
        return "needs_attention", "Submission was attempted but confirmation was not detected.", {
            "completed_fields": completed_fields, "unanswered_fields": ["Submission confirmation was not detected"],
            "answers": generated_answers,
        }
    finally:
        try:
            page.close()
        except Exception:
            pass
        if own_context:
            try:
                context.close()
            finally:
                if own_playwright and playwright is not None:
                    playwright.stop()


def run_agent(target_job_id: str | None = None, dry_run: bool = False) -> None:
    with _guard:
        if _runtime["running"]:
            return
        mode = "dry run" if dry_run else "daily run"
        _runtime.update(running=True, message=f"Preparing {mode}", current_job="", mode=mode)
    try:
        cfg = settings()
        if dry_run:
            cfg = {**cfg, "auto_submit": False, "daily_limit": 1}
        if not cfg["profile"].get("profile_complete"):
            raise RuntimeError("Complete the audited application profile before the agent can apply.")
        jobs = []
        from backend.api import decision_center
        decision_state = decision_center._load()
        for raw in jobright._read_store().get("items", []):
            _, relevant = jobright._classify_role(str(raw.get("title", "")))
            actionable = decision_center._decorate(raw, decision_state).get("decision") == "actionable"
            if relevant and actionable and not jobright._education_ineligible(raw) and raw.get("agent_status") not in {"applied", "applying"}:
                if target_job_id is None or str(raw.get("id")) == target_job_id:
                    jobs.append(raw)
        if target_job_id is not None and not jobs:
            raise RuntimeError("The selected job was not found or is not eligible for this run.")
        completed = 0
        attempted = 0
        limit = int(cfg["daily_limit"])
        from playwright.sync_api import sync_playwright
        with sync_playwright() as playwright:
            browser_context = playwright.chromium.launch_persistent_context(
                str(jobright.PROFILE), channel="chrome", headless=True)
            try:
                for job in jobs:
                    if _stop.is_set() or (limit > 0 and attempted >= limit):
                        break
                    attempted += 1
                    _runtime.update(current_job=job.get("title", ""), message=f"Preparing {job.get('company')}")
                    try:
                        _event(job, "preparing", "Loading the full job description.")
                        job = _ensure_description(job, browser_context)
                        resume = _generate_resume(job)
                        _event(job, "applying", "Opening employer application and attaching tailored résumé.")
                        evidence = DRY_RUNS / f"{job['id']}.png" if dry_run else None
                        status, message, review = _apply(job, resume, cfg, browser_context, evidence)
                        _event(job, status, message, **review,
                               **({"evidence_path": str(evidence)} if evidence else {}))
                        if status == "applied":
                            completed += 1
                    except Exception as exc:
                        _event(job, "error", str(exc))
            finally:
                browser_context.close()
        _runtime.update(message=f"{mode.title()} finished: {attempted} attempted, {completed} submitted.",
                        attempted=attempted, submitted=completed)
    except Exception as exc:
        _runtime.update(message=str(exc))
    finally:
        _runtime.update(running=False, current_job="", last_run=datetime.now().astimezone().isoformat())
        _stop.clear()


def _scheduler() -> None:
    last_date = ""
    while True:
        cfg = settings()
        now = datetime.now().astimezone()
        if cfg.get("enabled") and now.strftime("%H:%M") == cfg.get("run_at") and now.date().isoformat() != last_date:
            last_date = now.date().isoformat()
            threading.Thread(target=run_agent, daemon=True, name="astra-apply-run").start()
        time.sleep(30)


threading.Thread(target=_scheduler, daemon=True, name="astra-apply-scheduler").start()


@router.get("")
def get_agent() -> dict[str, Any]:
    return {"settings": settings(), "runtime": dict(_runtime), "events": _load(LOG, [])[:100]}


@router.patch("")
def update_agent(patch: dict[str, Any] = Body(...)) -> dict[str, Any]:
    current = settings()
    if "profile" in patch:
        patch["profile"] = {**current["profile"], **patch["profile"]}
    merged = {**current, **patch}
    merged["daily_limit"] = max(0, min(10000, int(merged["daily_limit"])))
    _save(SETTINGS, merged)
    return {"settings": merged, "runtime": dict(_runtime)}


@router.post("/run")
def run_now() -> dict[str, Any]:
    if _runtime["running"]:
        return dict(_runtime)
    threading.Thread(target=run_agent, daemon=True, name="astra-apply-run").start()
    return {"started": True}


@router.post("/dry-run/{job_id}")
def dry_run(job_id: str) -> dict[str, Any]:
    """Fill exactly one selected job and capture evidence, never submit it."""
    if _runtime["running"]:
        return dict(_runtime)
    threading.Thread(target=run_agent, args=(job_id, True), daemon=True,
                     name="astra-apply-dry-run").start()
    return {"started": True, "job_id": job_id, "auto_submit": False}


@router.post("/stop")
def stop() -> dict[str, Any]:
    _stop.set()
    return {"stopping": True}
