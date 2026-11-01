"""Compliant internet job discovery using public employer and job APIs."""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from fastapi import APIRouter, HTTPException, Query

from backend.api import jobright
from backend.core import config

router = APIRouter(prefix="/api/internet-jobs", tags=["internet-jobs"])
STORE = Path(config.DATA_DIR) / "internet_jobs.json"
_LOCK = threading.RLock()
_runtime: dict[str, Any] = {"running": False, "message": "Ready", "error": "", "last_refresh": None}

# Public company feeds. Invalid or unavailable boards are skipped without
# interrupting the rest of the refresh.
GREENHOUSE = {
    "Anthropic": "anthropic", "Scale AI": "scaleai", "Databricks": "databricks",
    "Waymo": "waymo", "Nuro": "nuro", "Together AI": "togetherai",
}
ASHBY = {"OpenAI": "openai", "Cohere": "cohere", "Cursor": "cursor", "Perplexity": "perplexity"}
LEVER = {"Zoox": "zoox", "Shield AI": "shieldai"}

ROLE_TERMS = re.compile(
    r"machine learning|\bml\b|artificial intelligence|\bai\b|deep learning|"
    r"large language model|\bllm\b|inference|computer vision|vision engineer|"
    r"research engineer|applied scientist|data scientist|software engineer|"
    r"ai infrastructure|ml infrastructure|ml platform|generative ai|robotics",
    re.I,
)
DOMAIN_TITLE = re.compile(
    r"machine learning|\bml\b|artificial intelligence|\bai\b|deep learning|"
    r"large language model|\bllm\b|inference|computer vision|vision|perception|"
    r"research (?:engineer|scientist)|applied scientist|data scientist|"
    r"generative|foundation model|robotics|autonomy|cuda|gpu|neural|nlp",
    re.I,
)
EXCLUDED_TITLE = re.compile(
    r"cyber|security engineer|frontend|front[- ]end|full[- ]stack|sales|marketing|"
    r"account executive|product manager|recruiter|attorney|counsel|designer|"
    r"\bmanager\b|\barchitect\b|\bstrategist\b|\badvisory\b|"
    r"\bsenior\b|\bsr\.?\b|\bstaff\b|\bprincipal\b|\blead\b|\bdirector\b|\bhead\b|distinguished|"
    r"\bpolicy\b|\bgovernance\b|\boperations\b|\bpeople\b|\bfellow(?:ship)?\b|"
    r"\bprogram\b|\bconsultant\b|\bcustomer\b|\bsolutions?\b",
    re.I,
)
CURRENT_STUDENT_REQUIRED = re.compile(
    r"currently (?:enrolled|pursuing|a student)|must be (?:enrolled|pursuing)|"
    r"pursuing (?:a |an )?(?:bachelor|master|ph\.?d|doctoral|degree)|"
    r"working toward (?:a |an )?(?:bachelor|master|ph\.?d|doctoral|degree)|"
    r"graduat(?:e|ing) (?:between|in) (?:spring|summer|fall|winter|20\d{2})",
    re.I,
)
CLEARANCE_REQUIRED = re.compile(
    r"(?:active|current|required|must (?:have|hold|possess|maintain)|ability to obtain).{0,35}"
    r"(?:security clearance|top secret|ts/sci|secret clearance|public trust)|"
    r"(?:security clearance|top secret|ts/sci|secret clearance|public trust).{0,35}required",
    re.I,
)
SPONSORSHIP_UNAVAILABLE = re.compile(
    r"(?:no|not|unable to|cannot|can't|will not).{0,40}(?:sponsor|sponsorship|visa)|"
    r"(?:sponsor|sponsorship|visa).{0,40}(?:not available|unavailable|not offered)",
    re.I,
)
CITIZENSHIP_REQUIRED = re.compile(
    r"(?:must (?:be|hold)|requires?).{0,35}(?:u\.?s\.? citizen|united states citizen|green card|permanent resident)|"
    r"(?:u\.?s\.? citizenship|united states citizenship).{0,25}required|"
    r"must (?:be|qualify as).{0,20}(?:u\.?s\.? person|united states person)",
    re.I,
)
US_HINT = re.compile(
    r"united states|\busa\b|\bu\.s\.\b|remote.*us|california|\bca\b|"
    r"new york|washington|texas|massachusetts|virginia|maryland|colorado|"
    r"pennsylvania|oregon|illinois|georgia|arizona|north carolina",
    re.I,
)
BAY_AREA = re.compile(
    r"san francisco|south san francisco|bay area|palo alto|mountain view|"
    r"sunnyvale|santa clara|san jose|redwood city|menlo park|fremont|"
    r"berkeley|oakland|hayward|cupertino|san mateo|los altos",
    re.I,
)
CALIFORNIA = re.compile(r"california|\bca\b|los angeles|san diego|irvine|sacramento", re.I)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_html(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<br\s*/?>|</p>|</li>|</h\d>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def _id(source: str, external_id: Any, company: str, title: str) -> str:
    raw = f"{source}|{external_id}|{company}|{title}".casefold()
    return hashlib.sha1(raw.encode()).hexdigest()[:24]


def _region(location: str) -> tuple[str, int]:
    if BAY_AREA.search(location):
        return "Bay Area", 3
    if CALIFORNIA.search(location):
        return "California", 2
    return "United States", 1


def _required_lines(description: str) -> list[str]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in description.splitlines() if line.strip()]
    units = []
    for line in lines:
        units.extend(x.strip() for x in re.split(r"(?<=[.!?;])\s+|\s+[•·]\s+", line) if x.strip())
    preferred = re.compile(r"preferred|nice to have|bonus|plus|ideally|desirable", re.I)
    return [line for line in units if not preferred.search(line)]


def _experience_minimum(description: str) -> int:
    requirements = []
    word_number = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                   "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    # Treat sentences/bullets independently. Some feeds flatten an entire job
    # description into one line; in that case a later mention of a master's
    # degree must not turn an earlier "4+ years total, including 2+ years..."
    # requirement into a two-year requirement.
    units = []
    for line in _required_lines(description):
        units.extend(x.strip() for x in re.split(r"(?<=[.!?;])\s+|\s+[•·]\s+", line) if x.strip())
    for line in units:
        if not re.search(r"experience|worked|background", line, re.I):
            continue
        values = []
        for match in re.finditer(r"\b(\d{1,2})\s*(?:\+|plus)?\s*(?:(?:-|to)\s*\d{1,2}\s*)?(?:years?|yrs?)\b", line, re.I):
            values.append(int(match.group(1)))
        for match in re.finditer(r"(?:minimum of|at least)\s+(\d{1,2})\s+(?:years?|yrs?)", line, re.I):
            values.append(int(match.group(1)))
        for match in re.finditer(r"(?:minimum of|at least)?\s*"
                                 r"(one|two|three|four|five|six|seven|eight|nine|ten)"
                                 r"(?:\s*\+|\s+plus)?\s+(?:years?|yrs?)", line, re.I):
            values.append(word_number[match.group(1).casefold()])
        if values:
            # Only use a master's-specific lower threshold when the same
            # sentence explicitly expresses degree alternatives. A generic
            # master's mention elsewhere is not an experience equivalency.
            degree_alternative = bool(
                re.search(r"(?:bachelor|b\.?s\.?).{0,80}(?:or|/)\s*(?:a\s+)?(?:master|m\.?s\.?)", line, re.I)
                or re.search(r"(?:master|m\.?s\.?).{0,80}(?:or|/)\s*(?:a\s+)?(?:bachelor|b\.?s\.?)", line, re.I)
            )
            requirements.append(min(values) if degree_alternative else max(values))
    return max(requirements) if requirements else 0


def _accept(title: str, location: str, description: str) -> bool:
    if not DOMAIN_TITLE.search(title) or EXCLUDED_TITLE.search(title):
        return False
    location_text = location or description[:800]
    if not location_text or not US_HINT.search(location_text):
        return False
    required_text = "\n".join(_required_lines(description))
    probe = {"title": title, "description": required_text, "description_sections": {"required": required_text}}
    from backend.api import jobright_agent
    profile = jobright_agent.settings().get("profile", {})
    years_match = re.search(r"\d+", str(profile.get("years_experience") or "2"))
    max_experience = int(years_match.group(0)) if years_match else 2
    currently_student = str(profile.get("currently_student") or "no").casefold() == "yes"
    has_clearance = str(profile.get("security_clearance") or "no").casefold() == "yes"
    needs_sponsorship = str(profile.get("requires_sponsorship") or "yes").casefold() == "yes"
    citizenship = str(profile.get("citizenship") or "India").casefold()
    if jobright._requires_phd(probe):
        return False
    if not currently_student and (jobright._requires_current_student(probe) or CURRENT_STUDENT_REQUIRED.search(required_text)):
        return False
    if _experience_minimum(required_text) > max_experience:
        return False
    if not has_clearance and CLEARANCE_REQUIRED.search(required_text):
        return False
    if needs_sponsorship and SPONSORSHIP_UNAVAILABLE.search(required_text):
        return False
    if citizenship not in {"us", "usa", "united states", "united states of america"} and CITIZENSHIP_REQUIRED.search(required_text):
        return False
    return True


def _finish(item: dict[str, Any]) -> dict[str, Any] | None:
    title, location, description = item["title"], item.get("location", ""), item.get("description", "")
    if not _accept(title, location, description):
        return None
    category, relevant = jobright._classify_role(title)
    if not relevant and not ROLE_TERMS.search(title):
        return None
    region, priority = _region(location)
    item.update(category=category, relevant=True, region=region, region_priority=priority,
                captured_at=_now(), description_status="complete" if len(description) >= 300 else "partial")
    return item


def _greenhouse(company: str, token: str) -> list[dict[str, Any]]:
    response = requests.get(f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true", timeout=35)
    response.raise_for_status()
    results = []
    for raw in response.json().get("jobs", []):
        title = str(raw.get("title") or "").strip()
        location = str((raw.get("location") or {}).get("name") or "").strip()
        description = _clean_html(raw.get("content"))
        url = str(raw.get("absolute_url") or "")
        item = _finish({"id": _id("Greenhouse", raw.get("id"), company, title), "external_id": str(raw.get("id") or ""),
            "requisition_id": str(raw.get("requisition_id") or ""), "title": title, "company": company,
            "location": location, "description": description, "url": url, "application_url": url,
            "source": "Greenhouse", "posted_at": raw.get("first_published"), "updated_at": raw.get("updated_at"),
            "salary": raw.get("pay_input_ranges") or []})
        if item:
            results.append(item)
    return results


def _ashby(company: str, board: str) -> list[dict[str, Any]]:
    response = requests.get(f"https://api.ashbyhq.com/posting-api/job-board/{board}",
                            headers={"User-Agent": "Mozilla/5.0 ASTRA Job Discovery"}, timeout=35)
    response.raise_for_status()
    results = []
    for raw in response.json().get("jobs", []):
        title = str(raw.get("title") or "").strip()
        location = str(raw.get("location") or "").strip()
        description = _clean_html(raw.get("descriptionHtml") or raw.get("descriptionPlain") or raw.get("description"))
        url = str(raw.get("jobUrl") or raw.get("applyUrl") or "")
        item = _finish({"id": _id("Ashby", raw.get("id"), company, title), "external_id": str(raw.get("id") or ""),
            "requisition_id": str(raw.get("jobCode") or ""), "title": title, "company": company,
            "location": location, "description": description, "url": url,
            "application_url": str(raw.get("applyUrl") or url), "source": "Ashby",
            "posted_at": raw.get("publishedAt"), "updated_at": raw.get("publishedAt"),
            "workplace_type": raw.get("workplaceType"), "salary": raw.get("compensation")})
        if item:
            results.append(item)
    return results


def _lever(company: str, site: str) -> list[dict[str, Any]]:
    response = requests.get(f"https://api.lever.co/v0/postings/{site}?mode=json", timeout=35)
    response.raise_for_status()
    results = []
    for raw in response.json():
        title = str(raw.get("text") or "").strip()
        categories = raw.get("categories") or {}
        location = str(categories.get("location") or ", ".join(categories.get("allLocations") or [])).strip()
        description = "\n".join(filter(None, [
            _clean_html(raw.get("descriptionPlain") or raw.get("description")),
            *[f"{str(x.get('text') or '')}\n{_clean_html(x.get('content'))}" for x in raw.get("lists", [])],
            _clean_html(raw.get("additionalPlain") or raw.get("additional")),
        ]))
        item = _finish({"id": _id("Lever", raw.get("id"), company, title), "external_id": str(raw.get("id") or ""),
            "requisition_id": "", "title": title, "company": company, "location": location,
            "description": description, "url": str(raw.get("hostedUrl") or ""),
            "application_url": str(raw.get("applyUrl") or raw.get("hostedUrl") or ""), "source": "Lever",
            "posted_at": None, "updated_at": None, "workplace_type": raw.get("workplaceType"),
            "salary": raw.get("salaryRange") or raw.get("salaryDescriptionPlain")})
        if item:
            results.append(item)
    return results


def _adzuna(_company: str, _key: str) -> list[dict[str, Any]]:
    app_id, app_key = os.getenv("ADZUNA_APP_ID", "").strip(), os.getenv("ADZUNA_APP_KEY", "").strip()
    if not app_id or not app_key:
        return []
    results = []
    queries = ("machine learning", "artificial intelligence", "LLM inference", "computer vision")
    for query in queries:
        response = requests.get("https://api.adzuna.com/v1/api/jobs/us/search/1", params={
            "app_id": app_id, "app_key": app_key, "what": query, "where": "United States",
            "results_per_page": 50, "sort_by": "date", "max_days_old": 30,
            "content-type": "application/json",
        }, timeout=35)
        response.raise_for_status()
        for raw in response.json().get("results", []):
            title = str(raw.get("title") or "").strip()
            company = str((raw.get("company") or {}).get("display_name") or "Unknown employer")
            location = str((raw.get("location") or {}).get("display_name") or "United States")
            item = _finish({"id": _id("Adzuna", raw.get("id"), company, title), "external_id": str(raw.get("id") or ""),
                "requisition_id": "", "title": title, "company": company, "location": location,
                "description": _clean_html(raw.get("description")), "url": str(raw.get("redirect_url") or ""),
                "application_url": str(raw.get("redirect_url") or ""), "source": "Adzuna",
                "posted_at": raw.get("created"), "updated_at": raw.get("created"),
                "salary": {"min": raw.get("salary_min"), "max": raw.get("salary_max")}})
            if item:
                results.append(item)
    return results


def _usajobs(_company: str, _key: str) -> list[dict[str, Any]]:
    api_key = os.getenv("USAJOBS_API_KEY", "").strip()
    email = os.getenv("USAJOBS_EMAIL", "").strip()
    if not api_key or not email:
        return []
    results = []
    headers = {"Authorization-Key": api_key, "User-Agent": email, "Host": "data.usajobs.gov"}
    for keyword in ("machine learning", "artificial intelligence", "computer vision"):
        response = requests.get("https://data.usajobs.gov/api/search", headers=headers, params={
            "Keyword": keyword, "LocationName": "United States", "ResultsPerPage": 100,
            "SortField": "opendate", "SortDirection": "Desc",
        }, timeout=35)
        response.raise_for_status()
        for wrapper in response.json().get("SearchResult", {}).get("SearchResultItems", []):
            raw = wrapper.get("MatchedObjectDescriptor") or {}
            title = str(raw.get("PositionTitle") or "").strip()
            locations = raw.get("PositionLocation") or []
            location = "; ".join(str(x.get("LocationName") or "") for x in locations) or "United States"
            details = raw.get("UserArea", {}).get("Details", {})
            description = _clean_html("\n".join(str(details.get(key) or "") for key in (
                "JobSummary", "MajorDuties", "Requirements", "Qualifications")))
            url = str(raw.get("PositionURI") or raw.get("ApplyURI", [""])[0] or "")
            company = str(raw.get("OrganizationName") or raw.get("DepartmentName") or "US Federal Government")
            item = _finish({"id": _id("USAJOBS", raw.get("PositionID"), company, title),
                "external_id": str(raw.get("PositionID") or ""), "requisition_id": str(raw.get("PositionID") or ""),
                "title": title, "company": company, "location": location, "description": description,
                "url": url, "application_url": str((raw.get("ApplyURI") or [url])[0]), "source": "USAJOBS",
                "posted_at": raw.get("PublicationStartDate"), "updated_at": raw.get("PublicationStartDate"),
                "salary": raw.get("PositionRemuneration") or []})
            if item:
                results.append(item)
    return results


def _write(items: list[dict[str, Any]], sources: list[dict[str, Any]]) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    pending = STORE.with_suffix(".tmp")
    pending.write_text(json.dumps({"updated_at": _now(), "items": items, "sources": sources}, ensure_ascii=False), encoding="utf-8")
    os.replace(pending, STORE)


def _sort_key(item: dict[str, Any]) -> tuple[int, float, str]:
    stamp = jobright._timestamp(str(item.get("updated_at") or item.get("posted_at") or ""))
    return (-int(item.get("region_priority", 0)), -stamp, str(item.get("company") or ""))


def _read() -> dict[str, Any]:
    try:
        value = json.loads(STORE.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"items": [], "sources": []}
    except Exception:
        return {"items": [], "sources": [], "updated_at": None}


def refresh() -> None:
    with _LOCK:
        if _runtime["running"]:
            return
        _runtime.update(running=True, message="Searching public employer job feeds", error="")
    feeds = [("Greenhouse", company, token, _greenhouse) for company, token in GREENHOUSE.items()]
    feeds += [("Ashby", company, board, _ashby) for company, board in ASHBY.items()]
    feeds += [("Lever", company, site, _lever) for company, site in LEVER.items()]
    if os.getenv("ADZUNA_APP_ID") and os.getenv("ADZUNA_APP_KEY"):
        feeds.append(("Adzuna", "USA aggregate", "configured", _adzuna))
    if os.getenv("USAJOBS_API_KEY") and os.getenv("USAJOBS_EMAIL"):
        feeds.append(("USAJOBS", "Federal jobs", "configured", _usajobs))
    items, source_status = [], []
    try:
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="astra-internet-jobs") as pool:
            futures = {pool.submit(fn, company, key): (source, company) for source, company, key, fn in feeds}
            for future in as_completed(futures):
                source, company = futures[future]
                try:
                    found = future.result()
                    items.extend(found)
                    source_status.append({"source": source, "company": company, "ok": True, "jobs": len(found)})
                except Exception as exc:
                    source_status.append({"source": source, "company": company, "ok": False, "jobs": 0, "error": str(exc)[:240]})
        unique = {}
        for item in items:
            key = jobright._normalized_url(item.get("application_url") or item.get("url") or "") or item["id"]
            unique[key] = item
        items = list(unique.values())
        items.sort(key=_sort_key)
        _write(items, source_status)
        _runtime.update(message=f"Found {len(items)} relevant US jobs from {sum(x['ok'] for x in source_status)} feeds",
                        last_refresh=_now())
    except Exception as exc:
        _runtime.update(error=str(exc), message="Internet job search failed")
    finally:
        _runtime["running"] = False


@router.post("/refresh")
def start_refresh() -> dict[str, Any]:
    if _runtime["running"]:
        return dict(_runtime)
    threading.Thread(target=refresh, daemon=True, name="astra-internet-job-refresh").start()
    return {"started": True}


@router.get("")
def list_jobs(q: str = Query(""), region: str = Query("all"), category: str = Query("all"),
              hide_applied: bool = Query(True), limit: int = Query(500, ge=1, le=5000)) -> dict[str, Any]:
    store = _read()
    items = [x for x in store.get("items", []) if _accept(
        str(x.get("title") or ""), str(x.get("location") or ""), str(x.get("description") or ""))]
    if hide_applied:
        from backend.api.decision_center import _best_application_match
        items = [x for x in items if not _best_application_match(x)]
    if q.strip():
        needle = q.casefold()
        items = [x for x in items if needle in f"{x.get('title','')} {x.get('company','')} {x.get('location','')} {x.get('description','')}".casefold()]
    if region != "all":
        items = [x for x in items if str(x.get("region")) == region]
    if category != "all":
        items = [x for x in items if str(x.get("category")) == category]
    items.sort(key=_sort_key)
    counts = {name: sum(x.get("region") == name for x in items) for name in ("Bay Area", "California", "United States")}
    from backend.api import jobright_agent
    profile = jobright_agent.settings().get("profile", {})
    return {"items": items[:limit], "total": len(items), "counts": counts, "updated_at": store.get("updated_at"),
            "sources": store.get("sources", []), "runtime": dict(_runtime), "active_filters": {
                "years_experience": str(profile.get("years_experience") or "2"),
                "currently_student": str(profile.get("currently_student") or "no"),
                "security_clearance": str(profile.get("security_clearance") or "no"),
                "requires_sponsorship": str(profile.get("requires_sponsorship") or "yes"),
                "degree": "Master's completed" if str(profile.get("masters_completed") or "yes").casefold() == "yes" else "Degree profile",
            }}


@router.get("/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = next((x for x in _read().get("items", []) if str(x.get("id")) == job_id), None)
    if not job:
        raise HTTPException(404, "Internet job not found")
    return job
