"""Compliant internet job discovery using public employer and job APIs."""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from functools import lru_cache

import requests
from fastapi import APIRouter, Body, HTTPException, Query

from backend.api import jobright
from backend.core import config

router = APIRouter(prefix="/api/internet-jobs", tags=["internet-jobs"])
STORE = Path(config.DATA_DIR) / "internet_jobs.json"
SETTINGS_STORE = Path(config.DATA_DIR) / "internet_jobs_settings.json"
SPONSOR_REGISTRY = Path(__file__).resolve().parent.parent / "data" / "top_tech_sponsors.json"
_LOCK = threading.RLock()
_runtime: dict[str, Any] = {"running": False, "message": "Ready", "error": "", "last_refresh": None}

# Public company feeds. Invalid or unavailable boards are skipped without
# interrupting the rest of the refresh.
GREENHOUSE = {
    "Anthropic": "anthropic", "Scale AI": "scaleai", "Databricks": "databricks",
    "Waymo": "waymo", "Nuro": "nuro", "Together AI": "togetherai",
    "Figure AI": "figureai", "Anduril": "andurilindustries",
}
ASHBY = {"OpenAI": "openai", "Cohere": "cohere", "Cursor": "cursor", "Perplexity": "perplexity",
         "Harvey": "harvey", "Pika": "pika", "Decagon": "decagon", "Baseten": "baseten"}
LEVER = {"Zoox": "zoox", "Shield AI": "shieldai", "Highspot": "highspot"}
MAJOR_CAREERS = {
    "google": "https://www.google.com/about/careers/applications/jobs/results/",
    "microsoft": "https://jobs.careers.microsoft.com/global/en/search",
    "apple": "https://jobs.apple.com/en-us/search",
    "amazon": "https://www.amazon.jobs/en/search",
    "amazon web services": "https://www.amazon.jobs/content/en/teams/amazon-web-services",
    "meta platforms": "https://www.metacareers.com/jobs/",
    "nvidia": "https://jobs.nvidia.com/",
    "oracle": "https://careers.oracle.com/jobs/",
}


@lru_cache(maxsize=1)
def _sponsor_registry() -> dict[str, Any]:
    try:
        value = json.loads(SPONSOR_REGISTRY.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"companies": []}
    except Exception:
        return {"companies": []}


@lru_cache(maxsize=1)
def _sponsor_map() -> dict[str, dict[str, Any]]:
    return {jobright._normalized_name(str(x.get("company") or "")): x
            for x in _sponsor_registry().get("companies", []) if isinstance(x, dict)}


@lru_cache(maxsize=512)
def _sponsor_for_company(company: str) -> dict[str, Any]:
    normalized = jobright._normalized_name(company)
    direct = _sponsor_map().get(normalized)
    if direct:
        return direct
    matches = [value for key, value in _sponsor_map().items()
               if jobright._company_match(normalized, key) or normalized in key or key in normalized]
    return max(matches, key=lambda x: int(x.get("tech_lca_positions", 0))) if matches else {}


def _safe_date(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text or re.fullmatch(r"\d+\s+(?:days?|hours?|months?)", text, re.I):
        return None
    return text if jobright._timestamp(text) else None


def _sponsorship_assessment(description: str, sponsor: dict[str, Any]) -> dict[str, Any]:
    if SPONSORSHIP_UNAVAILABLE.search(description):
        return {"status": "unavailable", "confidence": 0, "reason": "This posting explicitly says sponsorship is unavailable."}
    explicit = re.search(r"(?:visa|immigration|h-?1b).{0,50}(?:sponsor|support|available)|"
                         r"(?:sponsor|support).{0,50}(?:visa|immigration|h-?1b)", description, re.I)
    if explicit:
        return {"status": "posting_supports", "confidence": 100, "reason": explicit.group(0)[:180]}
    filings = int(sponsor.get("tech_lca_positions", 0))
    if filings >= 20:
        return {"status": "strong_history", "confidence": 80,
                "reason": f"Employer had {filings} recent certified technical LCA positions."}
    if filings:
        return {"status": "company_history", "confidence": 55,
                "reason": f"Employer had {filings} recent certified technical LCA positions; this role is unconfirmed."}
    return {"status": "unknown", "confidence": 25, "reason": "No role-specific sponsorship evidence was found."}


def _application_policy(description: str) -> dict[str, Any]:
    cooldown = re.search(r"(?:wait(?:ing)? period|reapply|reconsider).{0,50}(\d{2,3})[- ]day|"
                         r"(\d{2,3})[- ]day.{0,50}(?:wait|reapply|reconsider)", description, re.I)
    days = int(next((x for x in cooldown.groups() if x), 0)) if cooldown else 0
    return {"cooldown_days": days, "duplicate_family": bool(re.search(
        r"multiple applications|one application|same role|duplicate application", description, re.I))}


def _configured_feeds() -> list[tuple[str, str, str, Any]]:
    feeds: dict[tuple[str, str], tuple[str, str, str, Any]] = {}
    functions = {"greenhouse": _greenhouse, "ashby": _ashby, "lever": _lever}
    for item in _sponsor_registry().get("companies", []):
        ats, token = str(item.get("ats") or "").casefold(), str(item.get("token") or "")
        if item.get("direct_feed") and ats in functions and token:
            feeds[(ats, token)] = (ats.title(), str(item.get("company") or token), token, functions[ats])
    # Preserve locally configured boards even when they are outside the top-300 registry.
    for company, token in GREENHOUSE.items(): feeds.setdefault(("greenhouse", token), ("Greenhouse", company, token, _greenhouse))
    for company, token in ASHBY.items(): feeds.setdefault(("ashby", token), ("Ashby", company, token, _ashby))
    for company, token in LEVER.items(): feeds.setdefault(("lever", token), ("Lever", company, token, _lever))
    feeds[("amazon", "amazon.jobs")] = ("Amazon Jobs", "Amazon", "amazon.jobs", _amazon)
    feeds[("workday", "nvidia")] = ("Workday", "NVIDIA", "nvidia", _nvidia)
    return list(feeds.values())

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
    r"cyber|security engineer|frontend|front[- ]end|full[- ]?stack|sales|marketing|"
    r"account executive|product manager|product management|product builder|recruiter|attorney|counsel|designer|"
    r"developer advocate|developer relations|office of the ceo|\bintern(?:ship)?\b|vice president|\bvp\b|"
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
OPAQUE_OR_ADVANCED_RESEARCH = re.compile(
    r"minimum years of experience.{0,100}(?:internal job level|correlate)|"
    r"strong track record of (?:scientific |independent )?research|"
    r"track record of original (?:technical|scientific|research) work|"
    r"published research.{0,100}(?:major|top[- ]tier|neurips|icml|iclr|acl|emnlp|cvpr)|"
    r"(?:interpretability|alignment|transformer circuits?).{0,80}(?:publication|research output|relevant work)",
    re.I,
)
QUALITATIVE_SENIORITY = re.compile(
    r"\b(?:extensive|substantial|significant) experience\b|\bdeep expertise\b|"
    r"\bproven track record\b|\bexpert(?:ise)? in\b|"
    r"\bown and pursue a research agenda\b|\bhave built demanding (?:ml|ai|distributed) systems\b|"
    r"\btraining runs (?:working )?at scale\b|\bleading large[- ]scale .{0,35}(?:project|initiative)",
    re.I,
)
EARLY_CAREER_RESEARCH = re.compile(
    r"new (?:college )?grad|recent graduate|entry[- ]level|early career|"
    r"university graduate|graduate program|\bjunior\b|\bassociate\b|0\s*(?:-|to)\s*2 years",
    re.I,
)
US_HINT = re.compile(
    r"united states|\busa\b|\bus\b|\bu\.s\.\b|remote.*us|california|\bca\b|"
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
TECH_SKILLS = ("python", "pytorch", "tensorflow", "sql", "aws", "gcp", "azure", "docker",
               "kubernetes", "spark", "ray", "cuda", "c++", "opencv", "nlp", "llm", "rag",
               "transformer", "computer vision", "machine learning", "deep learning", "retrieval")
PROFILE_SKILLS = {"python", "pytorch", "tensorflow", "sql", "aws", "docker", "opencv", "nlp", "llm",
                  "rag", "transformer", "computer vision", "machine learning", "deep learning", "retrieval"}


def _settings() -> dict[str, Any]:
    defaults = {"enabled": True, "interval_hours": 6, "minimum_score": 60,
                "last_scheduled_refresh": None, "seen_ids": [], "corrections": [], "quality": {},
                "saved_searches": [], "source_health": {}, "alerts": [], "company_rules": {}}
    try:
        value = json.loads(SETTINGS_STORE.read_text(encoding="utf-8"))
        return {**defaults, **value} if isinstance(value, dict) else defaults
    except Exception:
        return defaults


def _save_settings(value: dict[str, Any]) -> None:
    SETTINGS_STORE.parent.mkdir(parents=True, exist_ok=True)
    pending = SETTINGS_STORE.with_suffix(".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(pending, SETTINGS_STORE)


@lru_cache(maxsize=4)
def _profile_for_stamp(stamp: int) -> tuple[dict[str, Any], str]:
    from backend.api import jobright_agent
    profile = jobright_agent.settings().get("profile", {})
    fingerprint = hashlib.sha1(json.dumps(profile, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return profile, fingerprint


def _profile_context() -> tuple[dict[str, Any], str]:
    from backend.api import jobright_agent
    try:
        stamp = jobright_agent.SETTINGS.stat().st_mtime_ns
    except OSError:
        stamp = 0
    return _profile_for_stamp(stamp)


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
        if not re.search(r"experience|worked|background|\b\d{1,2}\s*(?:\+|plus)?\s*years?\b|"
                         r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:plus\s+)?years?\b", line, re.I):
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


def _assessment(title: str, location: str, description: str) -> dict[str, Any]:
    reasons: list[dict[str, str]] = []
    if not DOMAIN_TITLE.search(title):
        reasons.append({"code": "unrelated_title", "message": "The title is outside your selected AI, ML, LLM, vision and related software domains.", "evidence": title})
    if EXCLUDED_TITLE.search(title):
        reasons.append({"code": "seniority_or_domain", "message": "The title indicates an excluded seniority level or unrelated function.", "evidence": title})
    location_text = location or description[:800]
    if not location_text or not US_HINT.search(location_text):
        reasons.append({"code": "outside_usa", "message": "The posting does not identify a United States work location.", "evidence": location or "No US location found"})
    required_text = "\n".join(_required_lines(description))
    probe = {"title": title, "description": required_text, "description_sections": {"required": required_text}}
    profile, _ = _profile_context()
    years_match = re.search(r"\d+", str(profile.get("years_experience") or "2"))
    max_experience = int(years_match.group(0)) if years_match else 2
    currently_student = str(profile.get("currently_student") or "no").casefold() == "yes"
    has_clearance = str(profile.get("security_clearance") or "no").casefold() == "yes"
    needs_sponsorship = str(profile.get("requires_sponsorship") or "yes").casefold() == "yes"
    citizenship = str(profile.get("citizenship") or "India").casefold()
    if jobright._requires_phd(probe):
        reasons.append({"code": "phd_required", "message": "The role requires PhD or doctoral eligibility.", "evidence": next((x for x in _required_lines(description) if re.search(r"ph\.?d|doctoral", x, re.I)), "PhD requirement detected")})
    if not currently_student and (jobright._requires_current_student(probe) or CURRENT_STUDENT_REQUIRED.search(required_text)):
        reasons.append({"code": "current_student", "message": "The role requires current enrolment or a future graduation window.", "evidence": next((x for x in _required_lines(description) if CURRENT_STUDENT_REQUIRED.search(x)), "Current-student requirement detected")})
    experience = _experience_minimum(required_text)
    if experience > max_experience:
        reasons.append({"code": "experience", "message": f"The role requires at least {experience} years; your profile allows up to {max_experience}.", "evidence": next((x for x in _required_lines(description) if _experience_minimum(x) == experience), f"{experience}+ years required")})
    if re.search(r"\bresearch scientist\b", title, re.I) and not experience and not EARLY_CAREER_RESEARCH.search(
        f"{title}\n{required_text}"
    ):
        reasons.append({"code": "research_seniority_unspecified", "message": "The Research Scientist role does not provide a clear 0–2 year or early-career pathway.", "evidence": title})
    # Research titles frequently omit a numeric requirement while describing a
    # senior frontier-research profile. Keep them only when the posting gives
    # explicit early-career evidence; do not interpret an unspecified internal
    # level or a publication track record as suitable for 0–2 years.
    research_title = bool(re.search(r"research (?:engineer|scientist)(?:\s*/\s*(?:research )?(?:engineer|scientist))?", title, re.I))
    if research_title and OPAQUE_OR_ADVANCED_RESEARCH.search(required_text) and not EARLY_CAREER_RESEARCH.search(
        f"{title}\n{required_text}"
    ):
        reasons.append({"code": "advanced_research", "message": "The role expects an advanced or specialised research record without clear early-career eligibility.", "evidence": next((x for x in _required_lines(description) if OPAQUE_OR_ADVANCED_RESEARCH.search(x)), "Advanced research track record required")})
    qualitative = QUALITATIVE_SENIORITY.search(required_text)
    if qualitative and not EARLY_CAREER_RESEARCH.search(f"{title}\n{required_text}"):
        reasons.append({"code": "qualitative_seniority", "message": "The role uses senior-level experience language without a 0–2 year pathway.", "evidence": next((x for x in _required_lines(description) if QUALITATIVE_SENIORITY.search(x)), qualitative.group(0))})
    if not has_clearance and CLEARANCE_REQUIRED.search(required_text):
        reasons.append({"code": "clearance", "message": "The role requires or expects eligibility for a security clearance.", "evidence": next((x for x in _required_lines(description) if CLEARANCE_REQUIRED.search(x)), "Security clearance required")})
    if needs_sponsorship and SPONSORSHIP_UNAVAILABLE.search(required_text):
        reasons.append({"code": "sponsorship", "message": "The posting says visa or immigration sponsorship is unavailable.", "evidence": next((x for x in _required_lines(description) if SPONSORSHIP_UNAVAILABLE.search(x)), "Sponsorship unavailable")})
    if citizenship not in {"us", "usa", "united states", "united states of america"} and CITIZENSHIP_REQUIRED.search(required_text):
        reasons.append({"code": "citizenship", "message": "The role requires US citizenship, permanent residency or US-person status.", "evidence": next((x for x in _required_lines(description) if CITIZENSHIP_REQUIRED.search(x)), "US citizenship restriction")})
    required_skills = sorted(skill for skill in TECH_SKILLS if re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", required_text, re.I))
    matched_skills = sorted(set(required_skills) & PROFILE_SKILLS)
    missing_skills = sorted(set(required_skills) - PROFILE_SKILLS)
    technical_fit = round(100 * len(matched_skills) / len(required_skills)) if required_skills else 75
    score = 90
    score -= 35 if experience == max_experience and experience else 0
    score -= 10 if len(description) < 500 else 0
    score += round((technical_fit - 50) * .2)
    score += 5 if BAY_AREA.search(location) else 3 if CALIFORNIA.search(location) else 0
    score = max(0, min(100, score if not reasons else 0))
    return {"eligible": not reasons, "score": score, "reasons": reasons,
            "experience_required": experience, "profile_max_experience": max_experience,
            "technical_fit": technical_fit, "matched_skills": matched_skills, "missing_skills": missing_skills}


def _accept(title: str, location: str, description: str) -> bool:
    return bool(_assessment(title, location, description)["eligible"])


def _finish(item: dict[str, Any], retain_filtered: bool = False) -> dict[str, Any] | None:
    title = re.sub(r"\s+", " ", str(item.get("title") or "")).strip()
    location = re.sub(r"\s+", " ", str(item.get("location") or "")).strip()
    description = str(item.get("description") or "").strip()
    if not title or not str(item.get("company") or "").strip():
        return None
    url_valid = str(item.get("url") or item.get("application_url") or "").startswith("http")
    item.update(title=title, location=location, description=description,
                posted_at=_safe_date(item.get("posted_at")), updated_at=_safe_date(item.get("updated_at")))
    # Store domain-related US postings even when profile filters reject them so
    # the UI can explain the decision and allow a correction. Truly unrelated
    # titles and non-US locations are discarded before persistence.
    if not DOMAIN_TITLE.search(title) or not (location or description[:800]) or not US_HINT.search(location or description[:800]):
        return None
    category, relevant = jobright._classify_role(title)
    if not relevant and not ROLE_TERMS.search(title):
        return None
    region, priority = _region(location)
    assessment = _assessment(title, location, description)
    if not assessment["eligible"] and not retain_filtered:
        return None
    sponsor = _sponsor_for_company(str(item.get("company") or ""))
    sponsorship = _sponsorship_assessment(description, sponsor)
    _, profile_fingerprint = _profile_context()
    item.update(category=category, relevant=True, region=region, region_priority=priority,
                captured_at=_now(), description_status="complete" if len(description) >= 300 else "partial",
                filter_decision=assessment, sponsor_history=bool(sponsor),
                tech_lca_positions=int(sponsor.get("tech_lca_positions", 0)),
                sponsorship_evidence=sponsor.get("evidence", ""), sponsorship=sponsorship,
                application_policy=_application_policy(description), profile_fingerprint=profile_fingerprint,
                data_quality={"valid_url": url_valid, "complete_description": len(description) >= 300,
                              "valid_date": bool(item.get("posted_at") or item.get("updated_at"))})
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
            "salary": raw.get("pay_input_ranges") or []}, retain_filtered=True)
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
            "workplace_type": raw.get("workplaceType"), "salary": raw.get("compensation")}, retain_filtered=True)
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
            "salary": raw.get("salaryRange") or raw.get("salaryDescriptionPlain")}, retain_filtered=True)
        if item:
            results.append(item)
    return results


def _amazon(_company: str, _key: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in ("machine learning", "artificial intelligence", "LLM inference", "computer vision"):
        for offset in (0, 100):
            response = requests.get("https://www.amazon.jobs/en/search.json", params={
                "base_query": query, "loc_query": "United States", "offset": offset, "result_limit": 100,
            }, headers={"User-Agent": "Mozilla/5.0 ASTRA Job Discovery"}, timeout=35)
            response.raise_for_status()
            jobs = response.json().get("jobs", [])
            for raw in jobs:
                external = str(raw.get("id_icims") or raw.get("id") or "")
                if not external or external in seen or str(raw.get("country_code") or "").upper() != "USA":
                    continue
                seen.add(external)
                title = str(raw.get("title") or "").strip()
                location = str(raw.get("normalized_location") or raw.get("location") or "United States")
                description = _clean_html("\n".join(str(raw.get(k) or "") for k in (
                    "description", "basic_qualifications", "preferred_qualifications")))
                url = "https://www.amazon.jobs" + str(raw.get("job_path") or "")
                item = _finish({"id": _id("Amazon Jobs", external, "Amazon", title), "external_id": external,
                    "requisition_id": external, "title": title, "company": "Amazon", "location": location,
                    "description": description, "url": url, "application_url": str(raw.get("url_next_step") or url),
                    "source": "Amazon Jobs", "posted_at": raw.get("posted_date"), "updated_at": None,
                    "salary": []}, retain_filtered=True)
                if item:
                    results.append(item)
            if len(jobs) < 100:
                break
    return results


def _nvidia(_company: str, _key: str) -> list[dict[str, Any]]:
    base = "https://nvidia.wd5.myworkdayjobs.com"
    board = "/wday/cxs/nvidia/NVIDIAExternalCareerSite"
    postings: dict[str, dict[str, Any]] = {}
    for query in ("machine learning", "artificial intelligence", "inference", "computer vision"):
        for offset in (0, 20, 40, 60, 80):
            response = requests.post(base + board + "/jobs", json={
                "limit": 20, "offset": offset, "searchText": query, "appliedFacets": {},
            }, headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 ASTRA Job Discovery"}, timeout=35)
            response.raise_for_status()
            batch = response.json().get("jobPostings", [])
            for raw in batch:
                location = str(raw.get("locationsText") or "")
                path = str(raw.get("externalPath") or "")
                title = str(raw.get("title") or "")
                if path and US_HINT.search(location) and DOMAIN_TITLE.search(title) and not EXCLUDED_TITLE.search(title):
                    postings[path] = raw
            if len(batch) < 20:
                break
    def load_detail(pair: tuple[str, dict[str, Any]]) -> dict[str, Any] | None:
        path, raw = pair
        try:
            detail = requests.get(base + board + path, headers={"User-Agent": "Mozilla/5.0 ASTRA Job Discovery"}, timeout=18)
            detail.raise_for_status()
            info = detail.json().get("jobPostingInfo") or {}
            title = str(info.get("title") or raw.get("title") or "").strip()
            location = str(info.get("location") or raw.get("locationsText") or "")
            description = _clean_html(info.get("jobDescription"))
            external = str((raw.get("bulletFields") or [""])[0] or path.rsplit("_", 1)[-1])
            url = str(info.get("externalUrl") or f"{base}/en-US/NVIDIAExternalCareerSite{path}")
            return _finish({"id": _id("Workday", external, "NVIDIA", title), "external_id": external,
                "requisition_id": external, "title": title, "company": "NVIDIA", "location": location,
                "description": description, "url": url, "application_url": url, "source": "Workday",
                "posted_at": info.get("startDate") or raw.get("postedOn"), "updated_at": None,
                "salary": []}, retain_filtered=True)
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=16, thread_name_prefix="astra-nvidia-detail") as pool:
        return [item for item in pool.map(load_detail, list(postings.items())[:160]) if item]


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
                "salary": {"min": raw.get("salary_min"), "max": raw.get("salary_max")}}, retain_filtered=True)
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
                "salary": raw.get("PositionRemuneration") or []}, retain_filtered=True)
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
    return (-int(item.get("rank_score", item.get("filter_decision", {}).get("score", 0))), -stamp,
            str(item.get("company") or ""))


def _canonical_key(item: dict[str, Any]) -> str:
    company = jobright._normalized_name(str(item.get("company") or ""))
    company = re.sub(r"\s+(?:inc|incorporated|llc|ltd|limited|corp|corporation)$", "", company).strip()
    role = jobright._normalized_role(str(item.get("title") or ""))
    requisition = re.sub(r"\W+", "", str(item.get("requisition_id") or "")).casefold()
    if requisition:
        return f"req:{company}:{requisition}"
    return f"role:{company}:{role}"


def _rank(item: dict[str, Any]) -> int:
    decision = item.get("filter_decision") or {}
    if not decision.get("eligible"):
        item["rank_components"] = {"eligibility": 0}
        return 0
    title = str(item.get("title") or "")
    experience_required = int(decision.get("experience_required", 0))
    experience_fit = 100 if experience_required <= 1 else 80 if experience_required == 2 else 40
    role_fit = 95 if re.search(r"llm|inference|machine learning|computer vision|\bai\b", title, re.I) else 75
    if re.search(r"\b(?:junior|associate|entry[- ]level|new grad|early career|engineer i|engineer 1)\b", title, re.I):
        role_fit = 100
    technical = int(decision.get("technical_fit", 75))
    sponsorship = int((item.get("sponsorship") or {}).get("confidence", 25))
    location = {3: 100, 2: 85, 1: 65}.get(int(item.get("region_priority", 0)), 50)
    stamp = jobright._timestamp(str(item.get("updated_at") or item.get("posted_at") or ""))
    age_days = max(0, int((time.time() - stamp) / 86400)) if stamp else 90
    freshness = 100 if age_days <= 3 else 90 if age_days <= 14 else 75 if age_days <= 30 else 55
    components = {"eligibility": 100, "experience": experience_fit, "technical": technical,
                  "sponsorship": sponsorship, "location": location, "freshness": freshness,
                  "role": role_fit, "interview_history": int(item.get("interview_history_score", 50))}
    score = round(components["eligibility"] * .15 + experience_fit * .18 + technical * .18 +
                  sponsorship * .17 + location * .10 + freshness * .10 + role_fit * .09 +
                  components["interview_history"] * .03)
    item["rank_components"] = components
    return max(0, min(100, score))


def _validate_link(item: dict[str, Any]) -> dict[str, Any]:
    url = str(item.get("application_url") or item.get("url") or "")
    if not url:
        item.update(link_status="missing", link_checked_at=_now())
        return item
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 ASTRA Link Validator"},
                                timeout=10, allow_redirects=True, stream=True)
        status = response.status_code
        final_url = response.url
        response.close()
        stamp = jobright._timestamp(str(item.get("updated_at") or item.get("posted_at") or ""))
        stale = bool(stamp and time.time() - stamp > 180 * 86400)
        item.update(link_status="stale" if status < 400 and stale else "open" if status < 400 else "closed", link_http_status=status,
                    final_url=final_url, link_checked_at=_now())
    except Exception as exc:
        item.update(link_status="unknown", link_error=str(exc)[:160], link_checked_at=_now())
    return item


def _read() -> dict[str, Any]:
    try:
        value = json.loads(STORE.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"items": [], "sources": []}
    except Exception:
        return {"items": [], "sources": [], "updated_at": None}


def refresh(force: bool = False) -> None:
    with _LOCK:
        if _runtime["running"]:
            return
        _runtime.update(running=True, message="Searching public employer job feeds", error="")
    feeds = _configured_feeds()
    if os.getenv("ADZUNA_APP_ID") and os.getenv("ADZUNA_APP_KEY"):
        feeds.append(("Adzuna", "USA aggregate", "configured", _adzuna))
    if os.getenv("USAJOBS_API_KEY") and os.getenv("USAJOBS_EMAIL"):
        feeds.append(("USAJOBS", "Federal jobs", "configured", _usajobs))
    prior = _read()
    settings = _settings()
    health = dict(settings.get("source_health") or {})
    prior_by_company: dict[str, list[dict[str, Any]]] = {}
    for item in prior.get("items", []):
        prior_by_company.setdefault(jobright._normalized_name(str(item.get("company") or "")), []).append(item)
    active_feeds, cached_feeds = [], []
    interval = max(1, int(settings.get("interval_hours", 6))) * 3600
    for feed in feeds:
        source, company, _, _ = feed
        key = f"{source}|{company}"
        last = jobright._timestamp(str((health.get(key) or {}).get("last_success") or ""))
        reusable = prior_by_company.get(jobright._normalized_name(company), [])
        if not force and last and time.time() - last < interval and reusable:
            cached_feeds.append((feed, reusable))
        else:
            active_feeds.append(feed)
    items = [dict(item) for _, reusable in cached_feeds for item in reusable]
    source_status = [{"source": feed[0], "company": feed[1], "ok": True, "jobs": len(reusable),
                      "cached": True, "duration_ms": 0,
                      "last_success": (health.get(f"{feed[0]}|{feed[1]}") or {}).get("last_success")}
                     for feed, reusable in cached_feeds]
    try:
        with ThreadPoolExecutor(max_workers=20, thread_name_prefix="astra-internet-jobs") as pool:
            started: dict[Any, float] = {}
            futures = {}
            for source, company, key, fn in active_feeds:
                future = pool.submit(fn, company, key)
                futures[future] = (source, company)
                started[future] = time.perf_counter()
            for future in as_completed(futures):
                source, company = futures[future]
                duration_ms = round((time.perf_counter() - started[future]) * 1000)
                health_key = f"{source}|{company}"
                try:
                    found = future.result()
                    items.extend(found)
                    health[health_key] = {"last_success": _now(), "last_failure": (health.get(health_key) or {}).get("last_failure"),
                                          "consecutive_failures": 0, "duration_ms": duration_ms, "jobs": len(found)}
                    source_status.append({"source": source, "company": company, "ok": True, "jobs": len(found),
                                          "duration_ms": duration_ms, **health[health_key]})
                except Exception as exc:
                    previous_health = health.get(health_key) or {}
                    health[health_key] = {**previous_health, "last_failure": _now(),
                                          "consecutive_failures": int(previous_health.get("consecutive_failures", 0)) + 1,
                                          "duration_ms": duration_ms, "error": str(exc)[:240]}
                    # Preserve the last successful snapshot through transient failures.
                    items.extend(prior_by_company.get(jobright._normalized_name(company), []))
                    source_status.append({"source": source, "company": company, "ok": False, "jobs": 0,
                                          "duration_ms": duration_ms, **health[health_key]})
        unique: dict[str, dict[str, Any]] = {}
        for item in items:
            key = _canonical_key(item)
            if key in unique:
                existing = unique[key]
                sources = set(existing.get("source_names") or [existing.get("source")])
                sources.add(item.get("source"))
                existing["source_names"] = sorted(str(x) for x in sources if x)
                existing["duplicate_count"] = int(existing.get("duplicate_count", 1)) + 1
                if len(str(item.get("description") or "")) > len(str(existing.get("description") or "")):
                    preserved = {"source_names": existing["source_names"], "duplicate_count": existing["duplicate_count"]}
                    existing.update(item)
                    existing.update(preserved)
            else:
                item.update(source_names=[item.get("source")], duplicate_count=1)
                unique[key] = item
        items = list(unique.values())
        eligible_items = [x for x in items if (x.get("filter_decision") or {}).get("eligible")]
        filtered_items = [x for x in items if not (x.get("filter_decision") or {}).get("eligible")]
        for item in filtered_items:
            item.setdefault("link_status", "not_checked")
        with ThreadPoolExecutor(max_workers=16, thread_name_prefix="astra-link-check") as pool:
            eligible_items = list(pool.map(_validate_link, eligible_items))
        items = eligible_items + filtered_items
        items = [x for x in items if x.get("link_status") not in {"closed", "stale"}]
        for item in items:
            item["rank_score"] = _rank(item)
        items.sort(key=_sort_key)
        _write(items, source_status)
        previous = set(settings.get("seen_ids") or [])
        eligible_ids = {str(x.get("id")) for x in items if (x.get("filter_decision") or {}).get("eligible")}
        new_ids = eligible_ids - previous
        settings["new_jobs"] = len(new_ids)
        alerts = list(settings.get("alerts") or [])
        for saved in settings.get("saved_searches", []):
            matches = []
            for item in items:
                if str(item.get("id")) not in new_ids or not (item.get("filter_decision") or {}).get("eligible"):
                    continue
                text = f"{item.get('title','')} {item.get('company','')} {item.get('description','')}".casefold()
                if saved.get("q") and str(saved["q"]).casefold() not in text: continue
                if saved.get("region", "all") != "all" and item.get("region") != saved.get("region"): continue
                if saved.get("category", "all") != "all" and item.get("category") != saved.get("category"): continue
                if int(item.get("rank_score", 0)) < int(saved.get("minimum_score", 60)): continue
                matches.append(str(item.get("id")))
            if matches:
                alerts.append({"id": hashlib.sha1(f"{saved.get('id')}|{_now()}".encode()).hexdigest()[:16],
                               "saved_search_id": saved.get("id"), "name": saved.get("name"),
                               "job_ids": matches, "count": len(matches), "created_at": _now(), "read": False})
        settings["alerts"] = alerts[-500:]
        settings["seen_ids"] = sorted(previous | eligible_ids)[-10000:]
        settings["last_scheduled_refresh"] = _now()
        settings["source_health"] = health
        _save_settings(settings)
        _runtime.update(message=f"Found {len(items)} relevant US jobs from {sum(x['ok'] for x in source_status)} feeds",
                        last_refresh=_now())
    except Exception as exc:
        _runtime.update(error=str(exc), message="Internet job search failed")
    finally:
        _runtime["running"] = False


@router.post("/refresh")
def start_refresh(force: bool = Query(True)) -> dict[str, Any]:
    if _runtime["running"]:
        return dict(_runtime)
    threading.Thread(target=refresh, kwargs={"force": force}, daemon=True, name="astra-internet-job-refresh").start()
    return {"started": True, "force": force}


@router.get("")
def list_jobs(q: str = Query(""), region: str = Query("all"), category: str = Query("all"),
              hide_applied: bool = Query(True), view: str = Query("eligible"),
              limit: int = Query(500, ge=1, le=5000)) -> dict[str, Any]:
    store = _read()
    state = _settings()
    _, current_profile_fingerprint = _profile_context()
    corrections = {str(x.get("job_id")): x for x in state.get("corrections", [])}
    try:
        from backend.api.decision_center import _application_candidates
        historical = _application_candidates()
    except Exception:
        historical = []
    outcomes: dict[str, list[str]] = {}
    for candidate in historical:
        app = candidate.get("app") or {}
        outcomes.setdefault(jobright._normalized_name(str(app.get("company") or "")), []).append(
            str(app.get("status") or "").casefold())
    items = []
    for raw in store.get("items", []):
        x = dict(raw)
        assessment = dict(x.get("filter_decision") or {}) if (not x.get("profile_fingerprint") or x.get("profile_fingerprint") == current_profile_fingerprint) else _assessment(
            str(x.get("title") or ""), str(x.get("location") or ""), str(x.get("description") or ""))
        correction = corrections.get(str(x.get("id")))
        if correction and correction.get("action") == "restore":
            assessment = {**assessment, "eligible": True, "score": max(60, int(assessment.get("score", 0))),
                          "overridden": True}
        elif correction and correction.get("action") == "hide":
            assessment = {**assessment, "eligible": False, "reasons": [{"code": "user_hidden", "message": "You hid this posting.", "evidence": correction.get("note", "Manual correction")}], "overridden": True}
        company_outcomes = outcomes.get(jobright._normalized_name(str(x.get("company") or "")), [])
        x["interview_history_score"] = 85 if any(re.search(r"interview|recruiter|assessment|offer", s) for s in company_outcomes) else 35 if any("reject" in s for s in company_outcomes) else 50
        x["filter_decision"] = assessment
        x["rank_score"] = _rank(x)
        if (not correction or correction.get("action") != "restore") and assessment.get("eligible") and x["rank_score"] < int(state.get("minimum_score", 60)):
            assessment = {**assessment, "eligible": False, "reasons": [{
                "code": "low_fit", "message": f"The fit score is below your {state.get('minimum_score', 60)}% recommendation threshold.",
                "evidence": f"Calculated fit: {x['rank_score']}%",
            }]}
            x["filter_decision"] = assessment
        items.append(x)
    evaluated_items = list(items)
    if view == "eligible":
        items = [x for x in items if (x.get("filter_decision") or {}).get("eligible")]
    elif view == "filtered":
        items = [x for x in items if not (x.get("filter_decision") or {}).get("eligible")]
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
    eligible_total = sum(bool((x.get("filter_decision") or {}).get("eligible")) for x in evaluated_items)
    filtered_total = len(evaluated_items) - eligible_total
    reason_counts: dict[str, int] = {}
    for x in evaluated_items:
        for reason in (x.get("filter_decision") or {}).get("reasons", []):
            reason_counts[reason["code"]] = reason_counts.get(reason["code"], 0) + 1
    registry = _sponsor_registry()
    targets = registry.get("companies", [])
    connected = {jobright._normalized_name(str(x.get("company") or ""))
                 for x in store.get("sources", []) if x.get("ok")}
    return {"items": items[:limit], "total": len(items), "eligible_total": eligible_total,
            "filtered_total": filtered_total, "reason_counts": reason_counts, "new_jobs": int(state.get("new_jobs", 0)),
            "alerts": [x for x in state.get("alerts", []) if not x.get("read")],
            "counts": counts, "updated_at": store.get("updated_at"),
            "sources": store.get("sources", []), "runtime": dict(_runtime), "sponsor_registry": {
                "targets": len(targets),
                "direct_feeds": sum(bool(x.get("direct_feed")) for x in targets),
                "connected_feeds": sum(jobright._normalized_name(str(x.get("company") or "")) in connected
                                       for x in targets if x.get("direct_feed")),
                "evidence": registry.get("generated_from"), "source_url": registry.get("source_url"),
                "disclaimer": registry.get("disclaimer"),
            }, "active_filters": {
                "years_experience": str(profile.get("years_experience") or "2"),
                "currently_student": str(profile.get("currently_student") or "no"),
                "security_clearance": str(profile.get("security_clearance") or "no"),
                "requires_sponsorship": str(profile.get("requires_sponsorship") or "yes"),
                "degree": "Master's completed" if str(profile.get("masters_completed") or "yes").casefold() == "yes" else "Degree profile",
            }}


@router.get("/settings")
def get_settings() -> dict[str, Any]:
    return _settings()


@router.patch("/settings")
def update_settings(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    value = _settings()
    if "enabled" in payload:
        value["enabled"] = bool(payload["enabled"])
    if "interval_hours" in payload:
        value["interval_hours"] = max(1, min(168, int(payload["interval_hours"])))
    if "minimum_score" in payload:
        value["minimum_score"] = max(0, min(100, int(payload["minimum_score"])))
    _save_settings(value)
    return value


@router.post("/{job_id}/feedback")
def feedback(job_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    action = str(payload.get("action") or "")
    if action not in {"restore", "hide", "correct"}:
        raise HTTPException(400, "Unsupported correction action")
    value = _settings()
    corrections = [x for x in value.get("corrections", []) if str(x.get("job_id")) != job_id]
    corrections.append({"job_id": job_id, "action": action, "note": str(payload.get("note") or ""), "created_at": _now()})
    value["corrections"] = corrections[-5000:]
    quality = dict(value.get("quality") or {})
    quality[action] = int(quality.get(action, 0)) + 1
    value["quality"] = quality
    _save_settings(value)
    return {"ok": True, "settings": value}


@router.post("/feedback/bulk")
def bulk_feedback(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    ids = [str(x) for x in payload.get("job_ids", [])][:1000]
    action = str(payload.get("action") or "")
    if action not in {"restore", "hide"}:
        raise HTTPException(400, "Unsupported bulk action")
    value = _settings()
    corrections = [x for x in value.get("corrections", []) if str(x.get("job_id")) not in set(ids)]
    corrections.extend({"job_id": job_id, "action": action, "note": str(payload.get("note") or "Bulk review"),
                        "created_at": _now()} for job_id in ids)
    value["corrections"] = corrections[-5000:]
    _save_settings(value)
    return {"ok": True, "updated": len(ids)}


@router.get("/saved-searches")
def saved_searches() -> dict[str, Any]:
    value = _settings()
    return {"items": value.get("saved_searches", []),
            "alerts": [x for x in value.get("alerts", []) if not x.get("read")]}


@router.post("/saved-searches")
def save_search(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    value = _settings()
    item = {"id": str(payload.get("id") or hashlib.sha1(f"{_now()}|{payload}".encode()).hexdigest()[:16]),
            "name": str(payload.get("name") or "Saved search")[:80], "q": str(payload.get("q") or "")[:160],
            "region": str(payload.get("region") or "all"), "category": str(payload.get("category") or "all"),
            "minimum_score": max(0, min(100, int(payload.get("minimum_score", 60)))), "created_at": _now()}
    searches = [x for x in value.get("saved_searches", []) if str(x.get("id")) != item["id"]]
    searches.append(item)
    value["saved_searches"] = searches[-100:]
    _save_settings(value)
    return item


@router.delete("/saved-searches/{search_id}")
def delete_search(search_id: str) -> dict[str, Any]:
    value = _settings()
    before = len(value.get("saved_searches", []))
    value["saved_searches"] = [x for x in value.get("saved_searches", []) if str(x.get("id")) != search_id]
    _save_settings(value)
    return {"deleted": before - len(value["saved_searches"])}


@router.post("/alerts/{alert_id}/read")
def read_alert(alert_id: str) -> dict[str, Any]:
    value = _settings()
    for item in value.get("alerts", []):
        if str(item.get("id")) == alert_id:
            item["read"] = True
    _save_settings(value)
    return {"ok": True}


@router.get("/source-health")
def source_health() -> dict[str, Any]:
    value = _settings()
    rows = [{"key": key, **health} for key, health in (value.get("source_health") or {}).items()]
    rows.sort(key=lambda x: (-int(x.get("consecutive_failures", 0)), -int(x.get("duration_ms", 0))))
    return {"items": rows, "total": len(rows),
            "healthy": sum(int(x.get("consecutive_failures", 0)) == 0 for x in rows),
            "failing": sum(int(x.get("consecutive_failures", 0)) > 0 for x in rows)}


@router.get("/analytics")
def analytics() -> dict[str, Any]:
    store, state = _read(), _settings()
    items = store.get("items", [])
    eligible = [x for x in items if _accept(str(x.get("title") or ""), str(x.get("location") or ""), str(x.get("description") or ""))]
    return {"discovered": len(items), "eligible": len(eligible), "filtered": len(items) - len(eligible),
            "new_jobs": int(state.get("new_jobs", 0)), "quality": state.get("quality", {}),
            "duplicates_merged": sum(max(0, int(x.get("duplicate_count", 1)) - 1) for x in items),
            "closed_removed": sum(int(not s.get("ok", True)) for s in store.get("sources", []))}


@router.get("/companies")
def companies(q: str = Query("")) -> dict[str, Any]:
    registry = _sponsor_registry()
    rows = []
    for raw in registry.get("companies", []):
        item = dict(raw)
        name = str(item.get("company") or "")
        lowered = name.casefold()
        career_url = ""
        for key, url in MAJOR_CAREERS.items():
            if key in lowered:
                career_url = url
                break
        if not career_url and item.get("direct_feed"):
            ats, token = str(item.get("ats") or ""), str(item.get("token") or "")
            if ats == "greenhouse": career_url = f"https://job-boards.greenhouse.io/{token}"
            elif ats == "lever": career_url = f"https://jobs.lever.co/{token}"
            elif ats == "ashby": career_url = f"https://jobs.ashbyhq.com/{token}"
        item["career_url"] = career_url
        item["live_in_astra"] = bool(item.get("direct_feed")) or any(k in lowered for k in ("amazon", "nvidia"))
        rows.append(item)
    if q.strip():
        needle = q.casefold()
        rows = [x for x in rows if needle in f"{x.get('company','')} {x.get('legal_employer','')}".casefold()]
    rows.sort(key=lambda x: (-int(x.get("tech_lca_positions", 0)), str(x.get("company") or "")))
    return {"items": rows, "total": len(rows), "source_url": registry.get("source_url"),
            "disclaimer": registry.get("disclaimer")}


@router.get("/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = next((x for x in _read().get("items", []) if str(x.get("id")) == job_id), None)
    if not job:
        raise HTTPException(404, "Internet job not found")
    return job


def _discovery_scheduler() -> None:
    """Refresh quietly on the configured cadence while ASTRA is running."""
    while True:
        try:
            value = _settings()
            if value.get("enabled") and not _runtime.get("running"):
                last = jobright._timestamp(str(value.get("last_scheduled_refresh") or ""))
                interval = max(1, int(value.get("interval_hours", 6))) * 3600
                if not last or time.time() - last >= interval:
                    refresh()
        except Exception:
            pass
        time.sleep(60)


if "pytest" not in sys.modules and "PYTEST_CURRENT_TEST" not in os.environ:
    threading.Thread(target=_discovery_scheduler, daemon=True, name="astra-internet-discovery").start()
