"""Local, user-authorized Jobright web-app importer.

The importer uses a dedicated Chrome profile under ASTRA's data directory.  It
never reads the user's normal browser cookies or credentials.  The user signs
in in the visible window, and ASTRA only stores structured job records.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.core.config import DATA_DIR

router = APIRouter(prefix="/api/jobright", tags=["jobright"])

STORE = Path(DATA_DIR) / "jobright_jobs.json"
PROFILE = Path(DATA_DIR) / "jobright_browser"
FEED_URL = "https://jobright.ai/jobs/recommend"
_lock = threading.Lock()
_state: dict[str, Any] = {"running": False, "phase": "idle", "error": "", "message": ""}

_EXCLUDED_ROLE = re.compile(
    r"\b(cyber|cybersecurity|security|soc analyst|penetration|identity access|iam|"
    r"full[ -]?stack|front[ -]?end|frontend|ui engineer|mobile engineer|ios engineer|android engineer|"
    r"qa engineer|quality assurance|test engineer|devops|site reliability|\bsre\b|"
    r"network engineer|systems administrator|it support|help desk|salesforce|firmware|embedded|hardware)\b",
    re.I,
)


def _classify_role(title: str) -> tuple[str, bool]:
    value = re.sub(r"\s+", " ", title).strip()
    if not value or _EXCLUDED_ROLE.search(value):
        return "Other", False
    categories = (
        ("Computer Vision", r"computer vision|machine vision|perception|image processing|visual ai|slam|mapping engineer"),
        ("LLM & Inference", r"\bllm\b|large language|inference engineer|generative ai|genai|\bnlp\b|language model|ai agent|agentic"),
        ("Machine Learning & AI", r"machine learning|deep learning|\bai engineer\b|artificial intelligence|applied scientist|research scientist|ml engineer|mlops|model training"),
        ("Data", r"data scientist|data engineer|analytics engineer|decision scientist|applied data|data platform"),
        ("Software Engineering", r"software engineer|software developer|backend engineer|back-end engineer|platform engineer|infrastructure engineer|python engineer|c\+\+ engineer"),
    )
    for category, pattern in categories:
        if re.search(pattern, value, re.I):
            return category, True
    return "Other", False


def _description_sections(text: str) -> dict[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    names = ["Responsibilities", "Qualification", "Required", "Preferred", "Benefits", "Company"]
    indexes = {name: lines.index(name) for name in names if name in lines}
    def section(name: str, stops: list[str]) -> str:
        start = indexes.get(name)
        if start is None:
            return ""
        ends = [indexes[x] for x in stops if x in indexes and indexes[x] > start]
        end = min(ends) if ends else len(lines)
        return "\n".join(lines[start + 1:end]).strip()
    return {
        "responsibilities": section("Responsibilities", ["Qualification", "Required", "Preferred", "Benefits", "Company"]),
        "required": section("Required", ["Preferred", "Benefits", "Company"]),
        "preferred": section("Preferred", ["Benefits", "Company"]),
        "benefits": section("Benefits", ["Company"]),
    }


def _read_store() -> dict[str, Any]:
    if not STORE.exists():
        return {"account": "kadali18@umd.edu", "updated_at": None, "items": []}
    try:
        data = json.loads(STORE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"items": []}
    except Exception:
        return {"account": "kadali18@umd.edu", "updated_at": None, "items": []}


def _write_store(items: list[dict[str, Any]]) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    existing = {str(x.get("id")): x for x in _read_store().get("items", [])}
    for item in items:
        previous = existing.get(str(item.get("id")), {})
        existing[str(item["id"])] = {**previous, **item}
    payload = {
        "account": "kadali18@umd.edu",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "items": sorted(existing.values(), key=lambda x: str(x.get("captured_at", "")), reverse=True),
    }
    STORE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalized_url(value: str) -> str:
    """Normalize a job URL for comparisons without tracking/query noise."""
    value = str(value or "").strip().casefold()
    return re.sub(r"[?#].*$", "", value).rstrip("/")


def _normalized_name(value: str) -> str:
    """Make company/title matching resilient to punctuation and spacing."""
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _normalized_role(value: str) -> str:
    """Normalize a role while ignoring non-role workplace qualifiers."""
    text = str(value or "").casefold().strip()
    text = re.sub(r"^\s*[\[(]\s*(?:remote|hybrid|onsite|on[- ]site)\s*[\])]\s*", "", text)
    text = re.sub(r"\s*[-–—]\s*(?:remote|hybrid|onsite|on[- ]site)\s*$", "", text)
    return _normalized_name(text)


def _posted_age_minutes(value: str) -> int:
    """Convert Jobright's relative posted label to an age for chronological sorting."""
    text = str(value or "").strip().casefold()
    if not text:
        return 10**12
    if "just now" in text or re.search(r"\bnow\b", text):
        return 0
    match = re.search(r"(\d+)\s*(minute|hour|day|week|month|year)s?\b", text)
    if not match:
        return 10**12
    amount = int(match.group(1))
    multiplier = {
        "minute": 1,
        "hour": 60,
        "day": 24 * 60,
        "week": 7 * 24 * 60,
        "month": 30 * 24 * 60,
        "year": 365 * 24 * 60,
    }[match.group(2)]
    return amount * multiplier


def _timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(str(value or "").replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return 0.0


_JD_STOPWORDS = {
    "and", "the", "with", "for", "you", "your", "our", "are", "this", "that",
    "will", "from", "have", "has", "job", "role", "work", "team", "years", "about",
}


def _text_tokens(value: str) -> frozenset[str]:
    return frozenset(
        word for word in re.findall(r"[a-z0-9+#.]{3,}", str(value or "").casefold())
        if word not in _JD_STOPWORDS
    )


def _substring_match(left: str, right: str, minimum: int, coverage: float) -> bool:
    if not left or not right:
        return False
    shorter, longer = sorted((left, right), key=len)
    return len(shorter) >= minimum and shorter in longer and len(shorter) / len(longer) >= coverage


def _company_match(left: str, right: str) -> bool:
    def core(value: str) -> str:
        words = value.split()
        suffixes = {"inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited", "company", "co"}
        while words and words[-1] in suffixes:
            words.pop()
        return " ".join(words)
    left_core, right_core = core(left), core(right)
    return left_core == right_core or _substring_match(left_core, right_core, 4, 0.6)


def _role_match(left: str, right: str) -> bool:
    if left == right or _substring_match(left, right, 10, 0.65):
        return True
    left_tokens, right_tokens = set(left.split()), set(right.split())
    if min(len(left_tokens), len(right_tokens)) < 2:
        return False
    overlap = len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))
    length_coverage = min(len(left), len(right)) / max(len(left), len(right))
    return overlap >= 0.8 and length_coverage >= 0.65


def _requires_phd(job: dict[str, Any]) -> bool:
    """Identify roles whose required eligibility is specifically doctoral-level."""
    title = str(job.get("title") or "")
    title_has_phd = bool(re.search(r"\b(?:ph\.?d\.?|doctoral)\b", title, re.I))
    title_allows_others = bool(re.search(
        r"\b(?:welcome|preferred|optional|bachelor'?s?|master'?s?)\b", title, re.I,
    ))
    if title_has_phd and not title_allows_others:
        return True
    sections = job.get("description_sections") or {}
    required = str(sections.get("required") or "").strip()
    if not required:
        description = str(job.get("description") or "")
        match = re.search(
            r"(?:required qualifications?|minimum qualifications?|requirements?)\s*\n(.*?)(?=\n\s*(?:preferred|benefits?|company)\b|\Z)",
            description, re.I | re.S,
        )
        required = match.group(1).strip() if match else ""
    if not required:
        return False
    # If the same degree requirement explicitly accepts a bachelor's or
    # master's, it is not PhD-specific. A PhD mentioned only under preferred
    # qualifications never reaches this required-only check.
    accepts_lower_degree = bool(re.search(
        r"\b(?:bachelor'?s?|b\.?s\.?|master'?s?|m\.?s\.?)\b.{0,80}\b(?:or|and/or)\b.{0,80}\b(?:ph\.?d\.?|doctoral)\b|"
        r"\b(?:ph\.?d\.?|doctoral)\b.{0,80}\b(?:or|and/or)\b.{0,80}\b(?:bachelor'?s?|b\.?s\.?|master'?s?|m\.?s\.?)\b",
        required, re.I | re.S,
    ))
    if accepts_lower_degree:
        return False
    return bool(re.search(
        r"\b(?:pursuing|current|enrolled|completed|earned|hold|have|requires?|required|minimum|must)\b.{0,100}\b(?:ph\.?d\.?|doctoral(?: degree)?)\b|"
        r"\b(?:ph\.?d\.?|doctoral(?: degree)?)\b.{0,80}\b(?:required|minimum|candidate|student|graduate|or equivalent experience)\b",
        required, re.I | re.S,
    ))


def _requires_current_student(job: dict[str, Any]) -> bool:
    """Identify jobs limited to candidates who are still enrolled or pursuing a degree."""
    title = str(job.get("title") or "")
    if re.search(r"\bco[- ]?op\b|\bstudent (?:intern|researcher|worker)\b", title, re.I):
        return True
    sections = job.get("description_sections") or {}
    required = str(sections.get("required") or "").strip()
    if not required:
        description = str(job.get("description") or "")
        match = re.search(
            r"(?:required qualifications?|minimum qualifications?|requirements?|eligibility)\s*\n(.*?)(?=\n\s*(?:preferred|benefits?|company)\b|\Z)",
            description, re.I | re.S,
        )
        required = match.group(1).strip() if match else description
    if not required:
        return False
    degree = r"(?:associate'?s?|bachelor'?s?|master'?s?|ph\.?d\.?|doctoral)(?:\s+degree)?"
    patterns = (
        rf"\b(?:working|work)\s+(?:toward|towards)\s+(?:an?\s+)?{degree}\b",
        rf"\b(?:currently\s+)?pursuing\s+(?:an?\s+)?{degree}\b",
        rf"\bcurrently\s+enrolled\b.{{0,120}}\b{degree}\b",
        rf"\b{degree}\b.{{0,120}}\bcurrently\s+enrolled\b",
        r"\bmust\s+(?:be\s+)?(?:currently\s+)?enrolled\b",
        r"\breturn(?:ing)?\s+to\s+(?:school|university|college)\b",
        r"\bgraduat(?:e|ing|ion)\s+(?:date\s+)?(?:between|in|after)\b",
        r"\bexpected\s+graduat(?:e|ion)\b",
    )
    return any(re.search(pattern, required, re.I | re.S) for pattern in patterns)


def _education_ineligible(job: dict[str, Any]) -> bool:
    return _requires_phd(job) or _requires_current_student(job)


def _application_match_keys() -> tuple[set[str], set[tuple[str, str]], list[tuple[str, frozenset[str]]]]:
    """Return URL, name, and saved-JD evidence represented by PDF resumes."""
    try:
        from backend.api import applications
        records = applications._load().get("applications", {}).values()
    except Exception:
        return set(), set(), []
    urls: set[str] = set()
    names: set[tuple[str, str]] = set()
    descriptions: list[tuple[str, frozenset[str]]] = []
    for app in records:
        # The resume library is the source of truth requested by the user. Ignore
        # tracker-only rows that have never had a PDF associated with them.
        if not any(str(f.get("path", "")).casefold().endswith(".pdf") for f in app.get("files", [])):
            continue
        if url := _normalized_url(str(app.get("job_url") or "")):
            urls.add(url)
        company = _normalized_name(str(app.get("company") or ""))
        role = _normalized_role(str(app.get("role") or ""))
        if company and role and role != "role not identified":
            names.add((company, role))
        jd_tokens = _text_tokens(str(app.get("jd_text") or ""))
        if company and len(jd_tokens) >= 30:
            descriptions.append((company, jd_tokens))
    return urls, names, descriptions


def _already_in_resume_library(
    job: dict[str, Any], urls: set[str], names: set[tuple[str, str]],
    descriptions: list[tuple[str, frozenset[str]]] | None = None,
) -> bool:
    job_urls = {
        value for value in (
            _normalized_url(str(job.get("url") or "")),
            _normalized_url(str(job.get("application_url") or "")),
        ) if value
    }
    if job_urls & urls:
        return True
    company = _normalized_name(str(job.get("company") or ""))
    role = _normalized_role(str(job.get("title") or ""))
    if any(_company_match(company, saved_company) and _role_match(role, saved_role)
           for saved_company, saved_role in names):
        return True
    job_jd = _text_tokens(str(job.get("description") or job.get("card_text") or ""))
    if len(job_jd) >= 30:
        for saved_company, saved_jd in descriptions or []:
            if not _company_match(company, saved_company):
                continue
            # Containment similarity handles descriptions saved from either a
            # complete page or a slightly shorter imported Jobright section.
            if len(job_jd & saved_jd) / min(len(job_jd), len(saved_jd)) >= 0.85:
                return True
    return False


def _parse_card(url: str, text: str) -> dict[str, Any] | None:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    timed = next((i for i, x in enumerate(lines) if re.search(r"\s[·•]\s.*(?:ago|now)|(?:minute|hour|day|week|month)s? ago|just now", x, re.I)), -1)
    if timed >= 0 and re.search(r"[·•]", lines[timed]):
        company, posted = [x.strip() for x in re.split(r"[·•]", lines[timed], maxsplit=1)]
        title = lines[timed + 1] if timed + 1 < len(lines) else ""
    elif timed >= 0:
        posted = lines[timed]
        cursor = timed + 1
        while cursor < len(lines) and (
            re.search(r"early applicant|school alumni|^company-logo$|^banner$", lines[cursor], re.I)
            or re.fullmatch(r"\d+ school alumni", lines[cursor], re.I)
        ):
            cursor += 1
        title = lines[cursor] if cursor < len(lines) else ""
        company = lines[cursor + 1] if cursor + 1 < len(lines) else ""
    else:
        title = next((x for x in lines if 4 < len(x) < 180 and not re.search(r"ago$|match$|apply|save", x, re.I)), lines[0])
        company = lines[lines.index(title) + 1] if lines.index(title) + 1 < len(lines) else ""
        posted = next((x for x in lines if re.search(r"minute|hour|day|week|month|just now", x, re.I)), "")
    if not title or title.upper() in {"SIGN IN", "JOIN NOW", "TRY FOR FREE"}:
        return None
    match = re.search(r"/jobs/info/([A-Za-z0-9_-]+)", url)
    key = match.group(1) if match else hashlib.sha256(url.encode()).hexdigest()[:20]
    return {
        "id": key,
        "title": title,
        "company": company,
        "location": next((x for x in lines if re.search(r"Remote|Hybrid|Onsite|, [A-Z]{2}\b", x)), ""),
        "posted": posted,
        "match": next((x for x in lines if "%" in x), ""),
        "url": url,
        "card_text": "\n".join(lines)[:4000],
        "summary": lines[lines.index("Why this job is a match") + 1] if "Why this job is a match" in lines and lines.index("Why this job is a match") + 1 < len(lines) else "",
        "description": "",
        "description_status": "not_loaded",
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def _collect_cards(page: Any) -> list[dict[str, Any]]:
    page.goto(FEED_URL, wait_until="domcontentloaded", timeout=90000)
    page.wait_for_timeout(5000)
    stable = 0
    previous = 0
    records: dict[str, dict[str, Any]] = {}
    for _ in range(100):
        cards = page.locator('a[href*="/jobs/info/"]').evaluate_all("""
          els => els.map(a => {
            let box = a;
            for (let i=0; i<6 && box.parentElement; i++) {
            const text=(box.innerText || '').trim(), size=text.length;
            if ((/Why this job is a match/i.test(text) && size <= 3500) || (size >= 160 && size <= 1800)) break;
              box = box.parentElement;
            }
            return {url: a.href, text: (box.innerText || a.innerText || '').trim()};
          })
        """)
        for raw in cards:
            item = _parse_card(raw.get("url", ""), raw.get("text", ""))
            if item:
                records[item["id"]] = item
        if len(records) == previous:
            stable += 1
        else:
            stable = 0
            previous = len(records)
        if stable >= 8:
            break
        page.evaluate("""() => {
          const nodes=[...document.querySelectorAll('*')].filter(x=>x.scrollHeight>x.clientHeight+300);
          const target=nodes.sort((a,b)=>b.scrollHeight-a.scrollHeight)[0];
          if(target) target.scrollTop=target.scrollHeight; else window.scrollTo(0,document.body.scrollHeight);
        }""")
        page.wait_for_timeout(1200)
    return list(records.values())


def _run_import() -> None:
    with _lock:
        _state.update(running=True, phase="opening", error="", message="Opening a private Jobright window…")
    try:
        from playwright.sync_api import sync_playwright
        PROFILE.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                str(PROFILE), channel="chrome", headless=False,
                viewport={"width": 1440, "height": 940},
                args=["--disable-blink-features=AutomationControlled"],
            )
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(FEED_URL, wait_until="domcontentloaded", timeout=90000)
            with _lock:
                _state.update(phase="waiting_for_login", message="Sign in to Jobright in the opened window. ASTRA will import after the job feed appears.")
            page.wait_for_function("""() => {
              const text=document.body.innerText || '';
              return /Your Saved Filters|Job Recommendations/i.test(text) &&
                     /Why this job is a match/i.test(text) &&
                     !/SIGN IN\\s+JOIN NOW/i.test(text.slice(0,2500));
            }""", timeout=600000)
            with _lock:
                _state.update(phase="importing", message="Reading every job currently available in your Jobright feed…")
            items = _collect_cards(page)
            _write_store(items)
            context.close()
            with _lock:
                _state.update(running=False, phase="complete", message=f"Imported {len(items)} jobs from Jobright.")
    except Exception as exc:
        with _lock:
            _state.update(running=False, phase="error", error=str(exc), message="Jobright import stopped.")


@router.get("")
def list_jobs(q: str = "", category: str = "all", relevant_only: bool = True, limit: int = 5000) -> dict[str, Any]:
    data = _read_store()
    applied_urls, applied_names, applied_descriptions = _application_match_keys()
    try:
        from backend.api import decision_center
        decision_state = decision_center._load()
    except Exception:
        decision_state = {"overrides": {}, "blocked_companies": []}
    blocked_companies = set(decision_state.get("blocked_companies", []))
    items = []
    for raw in data.get("items", []):
        item = dict(raw)
        item["category"], item["relevant"] = _classify_role(str(item.get("title", "")))
        item["already_applied"] = _already_in_resume_library(
            item, applied_urls, applied_names, applied_descriptions,
        )
        item["phd_specific"] = _requires_phd(item)
        item["education_ineligible"] = _education_ineligible(item)
        override = (decision_state.get("overrides", {}) or {}).get(str(item.get("id")), {})
        if override.get("action") == "not_applied":
            item["already_applied"] = False
        item["user_hidden"] = (
            override.get("action") in {"applied", "hide", "review_rejected"}
            or _normalized_name(str(item.get("company", ""))) in blocked_companies
        )
        items.append(item)
    # All user-facing counts describe the actionable, unapplied queue.
    unapplied = [x for x in items if not x["already_applied"] and not x["education_ineligible"] and not x["user_hidden"]]
    counts: dict[str, int] = {}
    for item in unapplied:
        if item["relevant"]:
            counts[item["category"]] = counts.get(item["category"], 0) + 1
    relevant_total = sum(counts.values())
    if relevant_only:
        items = [x for x in items if x["relevant"]]
    # Jobs with a matching PDF in the user's resume/application library are
    # already handled and should not return to the discovery queue.
    items = [x for x in items if not x["already_applied"]]
    items = [x for x in items if not x["education_ineligible"]]
    items = [x for x in items if not x["user_hidden"]]
    if category != "all":
        items = [x for x in items if x["category"] == category]
    if q.strip():
        needle = q.casefold()
        items = [x for x in items if needle in f"{x.get('title','')} {x.get('company','')} {x.get('location','')}".casefold()]
    # Jobright's visible posted/reposted age is the primary chronology. Recent
    # views and imports only break ties, so an old opened job cannot jump above a
    # newly posted role and make the feed look zig-zagged.
    items.sort(
        key=lambda x: (
            _posted_age_minutes(str(x.get("posted") or "")),
            -_timestamp(str(x.get("last_opened_at") or x.get("captured_at") or "")),
        )
    )
    return {**data, "items": items[: max(1, min(limit, 10000))], "total": len(items), "all_total": len(data.get("items", [])), "relevant_total": relevant_total, "category_counts": counts, "state": dict(_state)}


@router.post("/{job_id}/view")
def record_job_view(job_id: str) -> dict[str, str]:
    data = _read_store()
    job = next((x for x in data.get("items", []) if str(x.get("id")) == job_id), None)
    if not job:
        raise HTTPException(404, "Job not found")
    job["last_opened_at"] = datetime.now(timezone.utc).isoformat()
    _write_store([job])
    return {"id": job_id, "last_opened_at": str(job["last_opened_at"])}


@router.get("/status")
def status() -> dict[str, Any]:
    data = _read_store()
    return {**dict(_state), "total": len(data.get("items", [])), "updated_at": data.get("updated_at")}


@router.post("/connect")
def connect() -> dict[str, Any]:
    with _lock:
        if _state["running"]:
            return dict(_state)
        _state.update(running=True, phase="starting", error="", message="Starting Jobright connection…")
    threading.Thread(target=_run_import, name="astra-jobright-import", daemon=True).start()
    return dict(_state)


def load_details_in_context(job_id: str, context: Any) -> dict[str, Any]:
    """Load one job using an existing Playwright context.

    The application agent already owns a synchronous Playwright session.  Reusing
    that context avoids starting a nested sync_playwright instance in the same
    asyncio worker thread, which Playwright rejects.
    """
    data = _read_store()
    job = next((x for x in data.get("items", []) if str(x.get("id")) == job_id), None)
    if not job:
        raise HTTPException(404, "Job not found")
    job["category"], job["relevant"] = _classify_role(str(job.get("title", "")))
    if len(job.get("description", "")) >= 300 and job.get("description_sections") and job.get("application_url"):
        return job
    page = context.new_page()
    try:
        page.goto(job["url"], wait_until="domcontentloaded", timeout=90000)
        page.wait_for_timeout(3500)
        page_text = page.locator("main").first.inner_text()
        original = page.locator('a:has-text("Original Job Post")')
        application_url = original.first.get_attribute("href") if original.count() else ""
        sections = _description_sections(page_text)
        description = "\n\n".join(
            f"{label}\n{sections[key]}" for key, label in (
                ("responsibilities", "Responsibilities"),
                ("required", "Required qualifications"),
                ("preferred", "Preferred qualifications"),
                ("benefits", "Benefits"),
            ) if sections[key]
        )
        if len(description.strip()) < 300:
            raise RuntimeError("The full description was not visible. Reconnect Jobright and try again.")
        job["description"] = description.strip()[:50000]
        job["description_sections"] = sections
        job["application_url"] = application_url or job.get("application_url", "")
        job["description_status"] = "complete"
        job["phd_specific"] = _requires_phd(job)
        job["education_ineligible"] = _education_ineligible(job)
        _write_store([job])
        return job
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        page.close()


@router.post("/{job_id}/details")
def load_details(job_id: str) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(str(PROFILE), channel="chrome", headless=True)
        try:
            return load_details_in_context(job_id, context)
        finally:
            context.close()


@router.post("/{job_id}/open")
def open_job(job_id: str, target: str = "jobright") -> dict[str, str]:
    data = _read_store()
    job = next((x for x in data.get("items", []) if str(x.get("id")) == job_id), None)
    if not job:
        raise HTTPException(404, "Job not found")
    url = job.get("application_url") if target == "application" else job.get("url")
    if not url:
        raise HTTPException(422, "Load the complete job description first to discover the original application page.")
    if not webbrowser.open(str(url), new=2):
        raise HTTPException(500, "Could not open the browser.")
    return {"url": str(url), "target": target}
