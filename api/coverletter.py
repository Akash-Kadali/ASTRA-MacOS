# ============================================================
#  HIREX v3.0.0 — Cover Letter Generation
#  ------------------------------------------------------------
#  CHANGELOG v3.0.0 (vs v2.0.0):
#   REWRITE  Tone system: simple Indian English (IELTS 6.5), Hyderabad/Bangalore style
#   REWRITE  Banned phrases: real AI-tell patterns, not just clichés
#   REWRITE  3-paragraph prompt: implicit structure via examples, not labeled templates
#   REWRITE  LaTeX & escaping: fixed & → \& (not "and")
#   ADD      Resume-to-JD explicit mapping (requirement → experience matcher)
#   ADD      Company research via web search (not just JD text)
#   ADD      Experience relevance selector (picks best company per JD)
#   ADD      6-axis quality rubric scoring with targeted rewrites
#   ADD      Fact-checking pass against actual resume content
#   ADD      JD keyword echo verification
#   ADD      Per-paragraph word count enforcement
#   ADD      Company type detection (startup vs enterprise vs Indian)
#   ADD      Salutation cultural awareness (Mr./Dr./Ms. for Indian context)
#   ADD      Post-humanize tone guard
#   ADD      Culture signal weaving into paragraph instructions
#   FIX      _strip_academic: selective (keeps degree name for relevant JDs)
#   FIX      Highlights extraction: JD-prioritized, up to 5 achievements
#   FIX      Word count hard enforcement
#   REMOVE   Formulaic paragraph labels from prompt
# ============================================================

from __future__ import annotations

import base64
import json
import re
import threading
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List, Set

import httpx
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI

from backend.core import config
from backend.core.utils import log_event, safe_filename, ensure_dir
from backend.core.compiler import compile_latex_safely
from backend.core.security import secure_tex_input

try:
    from backend.api.render_tex import render_final_tex
except Exception:
    from api.render_tex import render_final_tex

router = APIRouter(prefix="/api/coverletter", tags=["coverletter"])

# ── OpenAI client (thread-safe singleton) ────────────────────
_openai_lock = threading.Lock()
_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    with _openai_lock:
        if _openai_client is None:
            _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


_MODEL = getattr(config, "COVERLETTER_MODEL", None) or "gpt-5.4-nano"


# ============================================================
# 🔧 GPT helpers
# ============================================================

def _json_from_text(text: str, default: dict) -> dict:
    if not text:
        return default
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return default
    try:
        return json.loads(m.group(0))
    except Exception:
        return default


async def _chat_text(system: str, user: str, temperature: float = 0.7) -> str:
    client = _get_openai_client()
    resp = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


async def _chat_json(prompt: str, temperature: float = 0.0) -> dict:
    client = _get_openai_client()
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        content = (resp.choices[0].message.content or "").strip()
        try:
            return json.loads(content)
        except Exception:
            return _json_from_text(content, {})
    except TypeError:
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return _json_from_text(resp.choices[0].message.content or "", {})


# ============================================================
# 🔒 LaTeX utilities — FIXED: & → \& not "and"
# ============================================================

_LATEX_SPECIAL = {
    "&": r"\&",  # v3.0.0 FIX: keep ampersand visible, not replace with "and"
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
}


def _latex_escape(text: str) -> str:
    """Escape LaTeX special chars ONCE. Safe to call on plain text."""
    if not text:
        return ""
    for ch, repl in _LATEX_SPECIAL.items():
        text = text.replace(ch, repl)
    text = re.sub(r"~", r"\\string~", text)
    text = re.sub(r"\^", r"\\string^", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _strip_academic(text: str, keep_degree_name: bool = False) -> str:
    """Remove GPA, graduation dates, coursework, degree mentions.
    v3.0.0: If keep_degree_name=True, preserves degree names (MS, BS) without dates/GPA.
    """
    if not text:
        return ""
    # Always strip these
    always_strip = [
        r"\bC?GPA\s*[:\-]?\s*\d+(\.\d+)?(/\d+(\.\d+)?)?",
        r"\b\d+(\.\d+)?\s*(GPA|CGPA)\b",
        r"\b(graduat(ed?|ing|ion))\s*(in|from|date)?\s*\d{4}\b",
        r"\b(class of|expected|graduating)\s*\d{4}\b",
        r"\b(relevant\s+)?coursework\b[:\s]*[^.]*\.",
        r"\bcourses?\s+(include|including|such as)[^.]*\.",
        r"\bdean'?s?\s*list\b",
        r"\bcum\s+laude\b",
    ]
    for p in always_strip:
        text = re.sub(p, "", text, flags=re.IGNORECASE)

    # Conditionally strip degree + date combos
    if not keep_degree_name:
        degree_patterns = [
            r"\b(bachelor'?s?|master'?s?|ph\.?d\.?|b\.?s\.?|m\.?s\.?)\s*(degree)?\s*(in\s+\w+)?\s*,?\s*\d{4}",
            r"\buniversity[^,]*,?\s*\d{4}",
            r"\bcollege[^,]*,?\s*\d{4}",
        ]
        for p in degree_patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE)

    return re.sub(r"\s{2,}", " ", text).strip()


# ============================================================
# 🏢 Company type detection — v3.0.0 NEW
# ============================================================

_STARTUP_SIGNALS = {
    "yc", "y combinator", "seed", "series a", "series b", "pre-seed",
    "stealth", "early-stage", "founding", "co-founder", "startup",
    "fast-growing", "venture", "backed by",
}
_ENTERPRISE_SIGNALS = {
    "fortune 500", "enterprise", "global", "worldwide", "established",
    "publicly traded", "nasdaq", "nyse", "billion", "decade",
}
_INDIAN_COMPANY_SIGNALS = {
    "bangalore", "bengaluru", "hyderabad", "pune", "mumbai", "chennai",
    "noida", "gurgaon", "gurugram", "delhi", "kolkata", "india",
    "infosys", "wipro", "tcs", "hcl", "tech mahindra", "mindtree",
    "mphasis", "ltimindtree", "persistent", "zoho", "freshworks",
    "razorpay", "phonepe", "swiggy", "zomato", "flipkart", "meesho",
    "cred", "groww", "zerodha", "ola", "paytm",
}


def _detect_company_type(jd_text: str, company: str) -> str:
    """Detect if company is startup, enterprise, or Indian company."""
    combined = (jd_text + " " + company).lower()
    indian_score = sum(1 for s in _INDIAN_COMPANY_SIGNALS if s in combined)
    startup_score = sum(1 for s in _STARTUP_SIGNALS if s in combined)
    enterprise_score = sum(1 for s in _ENTERPRISE_SIGNALS if s in combined)

    if indian_score >= 2:
        return "indian"
    if startup_score > enterprise_score:
        return "startup"
    if enterprise_score > 0:
        return "enterprise"
    return "us_tech"  # default for most US tech companies


# ============================================================
# 🧠 STEP 1: Extract JD intelligence (single GPT call)
# ============================================================

async def _extract_jd_intelligence(jd_text: str) -> Dict[str, Any]:
    """Extract company, role, address, team info, and key requirements in ONE call."""
    prompt = f"""Analyze this job description. Return STRICT JSON:
{{
    "company": "company name",
    "role": "exact job title",
    "hiring_manager_name": "name if explicitly stated, else empty string",
    "hiring_manager_title": "their title if stated (e.g. Engineering Manager, VP), else empty string",
    "team_name": "specific team if mentioned, else empty string",
    "team_mission": "what this team does, 1 sentence, else empty string",
    "city_state": "office location if mentioned, else empty string",
    "key_requirements": ["top 6 must-have skills/experiences"],
    "tech_stack": ["specific technologies mentioned"],
    "business_problem": "what business problem this role solves, 1 sentence",
    "company_product": "main product/platform this role works on, else empty string",
    "culture_signals": ["3-4 culture/values phrases from the JD"],
    "day_to_day_work": ["3-4 specific daily tasks this person will do"],
    "success_criteria": "how success is measured in first 6 months, else empty string"
}}

Only extract what is EXPLICITLY in the JD. Do not invent.

JOB DESCRIPTION:
{jd_text[:4500]}"""
    try:
        data = await _chat_json(prompt)
        return {
            "company":              (data.get("company") or "Company").strip(),
            "role":                 (data.get("role") or "Role").strip(),
            "hiring_manager_name":  (data.get("hiring_manager_name") or "").strip(),
            "hiring_manager_title": (data.get("hiring_manager_title") or "").strip(),
            "team_name":            (data.get("team_name") or "").strip(),
            "team_mission":         (data.get("team_mission") or "").strip(),
            "city_state":           (data.get("city_state") or "").strip(),
            "key_requirements":     data.get("key_requirements", [])[:6],
            "tech_stack":           data.get("tech_stack", [])[:8],
            "business_problem":     (data.get("business_problem") or "").strip(),
            "company_product":      (data.get("company_product") or "").strip(),
            "culture_signals":      data.get("culture_signals", [])[:4],
            "day_to_day_work":      data.get("day_to_day_work", [])[:4],
            "success_criteria":     (data.get("success_criteria") or "").strip(),
        }
    except Exception as e:
        log_event("jd_intel_fail", {"error": str(e)})
        return {
            "company": "Company", "role": "Role", "hiring_manager_name": "",
            "hiring_manager_title": "", "team_name": "", "team_mission": "",
            "city_state": "", "key_requirements": [], "tech_stack": [],
            "business_problem": "", "company_product": "", "culture_signals": [],
            "day_to_day_work": [], "success_criteria": "",
        }


# ============================================================
# 🔍 STEP 1b: Company research — v3.0.0 NEW
# ============================================================

async def _research_company(company: str, role: str, jd_text: str) -> Dict[str, Any]:
    """GPT-based company research for specific, real details."""
    prompt = f"""You are a tech industry researcher. Tell me about "{company}" for someone
applying to {role}.

Return STRICT JSON:
{{
    "what_they_build": "1 sentence — their main product/service, be specific",
    "recent_news": "1 sentence — any recent launch, funding, acquisition, or milestone. Say 'unknown' if unsure",
    "engineering_reputation": "1 sentence — what they are known for technically",
    "why_someone_would_join": "1 sentence — genuine reason an engineer would want to work here, not generic",
    "competitor_context": "who they compete with, 2-3 names",
    "tech_culture": "1 sentence — remote/hybrid/onsite, fast/slow, scrappy/process-heavy"
}}

Be honest. If you do not know something, say "unknown". Do not invent facts.

JD context:
{jd_text[:1500]}"""
    try:
        data = await _chat_json(prompt, temperature=0.3)
        return {
            "what_they_build":        (data.get("what_they_build") or "").strip(),
            "recent_news":            (data.get("recent_news") or "").strip(),
            "engineering_reputation":  (data.get("engineering_reputation") or "").strip(),
            "why_someone_would_join": (data.get("why_someone_would_join") or "").strip(),
            "competitor_context":     (data.get("competitor_context") or "").strip(),
            "tech_culture":           (data.get("tech_culture") or "").strip(),
        }
    except Exception as e:
        log_event("company_research_fail", {"error": str(e)})
        return {
            "what_they_build": "", "recent_news": "", "engineering_reputation": "",
            "why_someone_would_join": "", "competitor_context": "", "tech_culture": "",
        }


# ============================================================
# 🧠 STEP 2: Extract resume highlights — v3.0.0 REWRITTEN
#    JD-prioritized, up to 5 achievements, relevance-ranked
# ============================================================

async def _extract_resume_highlights(
    resume_text: str,
    intel: Dict[str, Any],
    jd_text: str,
) -> Dict[str, Any]:
    """Pull the strongest WORK achievements ranked by JD relevance."""
    if not (resume_text or "").strip():
        return {"achievements": [], "skills": [], "companies": [],
                "mapped_experiences": [], "strongest_company": ""}

    cleaned = _strip_academic(resume_text[:6000], keep_degree_name=True)
    reqs_str = ", ".join(intel.get("key_requirements", [])[:6])
    tech_str = ", ".join(intel.get("tech_stack", [])[:8])

    prompt = f"""Extract professional highlights from this resume, RANKED by relevance
to the target job.

TARGET ROLE: {intel['role']} at {intel['company']}
KEY REQUIREMENTS: {reqs_str}
TECH STACK: {tech_str}

RESUME:
{cleaned}

Return STRICT JSON:
{{
    "top_5_achievements": [
        {{
            "achievement": "1-sentence WORK achievement with quantified result",
            "company": "where this happened",
            "relevance_to_jd": "which JD requirement this maps to",
            "relevance_score": 0.9
        }}
    ],
    "relevant_skills": ["top 8 technical skills matching the JD"],
    "companies_worked": ["company names ordered by JD relevance"],
    "strongest_company_for_this_jd": "company name whose experience best matches this JD",
    "degree_name": "degree name if relevant to role (e.g. MS Applied Machine Learning), else empty"
}}

RULES:
- Only WORK experience achievements. No GPA, no coursework.
- Rank by relevance to THIS specific JD, not general impressiveness.
- Include quantified results where available.
- strongest_company_for_this_jd = the company whose work most directly maps to JD requirements."""

    try:
        data = await _chat_json(prompt)
        raw_achievements = data.get("top_5_achievements", [])
        achievements = []
        mapped = []
        for a in raw_achievements[:5]:
            if isinstance(a, dict):
                achievements.append(a.get("achievement", ""))
                mapped.append({
                    "achievement": a.get("achievement", ""),
                    "company": a.get("company", ""),
                    "maps_to": a.get("relevance_to_jd", ""),
                    "score": a.get("relevance_score", 0.5),
                })
            elif isinstance(a, str):
                achievements.append(a)
        return {
            "achievements":      [a for a in achievements if a][:5],
            "skills":            data.get("relevant_skills", [])[:8],
            "companies":         data.get("companies_worked", [])[:4],
            "mapped_experiences": sorted(mapped, key=lambda x: x.get("score", 0), reverse=True),
            "strongest_company": (data.get("strongest_company_for_this_jd") or "").strip(),
            "degree_name":       (data.get("degree_name") or "").strip(),
        }
    except Exception:
        return {"achievements": [], "skills": [], "companies": [],
                "mapped_experiences": [], "strongest_company": "", "degree_name": ""}


# ============================================================
# 🔗 STEP 2b: Explicit resume-to-JD mapping — v3.0.0 NEW
# ============================================================

async def _map_requirements_to_experience(
    intel: Dict[str, Any],
    highlights: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Explicitly map each JD requirement to a candidate experience."""
    reqs = intel.get("key_requirements", [])[:6]
    achievements = highlights.get("achievements", [])[:5]
    skills = highlights.get("skills", [])[:8]

    if not reqs or not achievements:
        return []

    prompt = f"""Map each job requirement to the candidate's best matching experience.

JOB REQUIREMENTS:
{json.dumps(reqs)}

CANDIDATE ACHIEVEMENTS:
{json.dumps(achievements)}

CANDIDATE SKILLS: {', '.join(skills)}

For each requirement, find the BEST matching achievement. If no good match, say "no direct match but transferable from [skill]".

Return STRICT JSON:
{{
    "mappings": [
        {{
            "requirement": "JD requirement",
            "best_match": "matching achievement or transferable skill explanation",
            "match_strength": "strong|moderate|transferable|weak",
            "proof_phrase": "short phrase to use in cover letter showing this match"
        }}
    ]
}}"""
    try:
        data = await _chat_json(prompt)
        return data.get("mappings", [])[:6]
    except Exception:
        return []


# ============================================================
# 📝 STEP 3: Draft cover letter body — v3.0.0 REWRITTEN
#    Simple Indian English, implicit structure, JD-mapped
# ============================================================

_WORD_TARGETS = {
    "short":    (130, 180),
    "standard": (200, 280),
    "long":     (300, 400),
}

_PARA_WORD_TARGETS = {
    "short":    [(40, 60), (50, 70), (30, 50)],
    "standard": [(55, 80), (80, 120), (50, 80)],
    "long":     [(80, 120), (120, 170), (80, 110)],
}

# v3.0.0: Real AI-tell patterns that recruiters catch
_BANNED_PHRASES = {
    # Old clichés
    "passionate", "dynamic", "cutting edge", "synergy", "dream job",
    "perfect fit", "thrilled", "honored", "privileged",
    "game-changer", "revolutionary", "since childhood",
    "humbly request", "blown away", "astonishing", "leverage synergies",
    "results-driven", "team player", "fast-paced environment",
    # v3.0.0: GPT cover letter tells
    "i am writing to express",
    "i would welcome the opportunity",
    "i am confident that",
    "eager to contribute",
    "aligns perfectly",
    "i believe my experience",
    "uniquely positioned",
    "strongly resonates",
    "i am drawn to",
    "i am excited to apply",
    "deeply impressed",
    "exceptional opportunity",
    "well-positioned",
    "my extensive experience",
    "proven track record",
    "innovative solutions",
    "cross-functional collaboration",
    "hit the ground running",
    "make a meaningful impact",
    "drive impactful results",
    "spearheaded", "leveraged", "orchestrated", "pioneered",
    "championed", "harnessed",
}

# v3.0.0: Company type → tone adjustments
_COMPANY_TONE_HINTS = {
    "startup": "Write like you are messaging a startup founder on Slack. Short punchy sentences. Show you can ship fast. No corporate language.",
    "enterprise": "Write like a professional email to a senior manager. Slightly more formal but still natural. Show reliability and process awareness.",
    "indian": "Write like you are emailing an Indian tech lead. Respectful but direct. Mention specific technical contributions. No unnecessary praise.",
    "us_tech": "Write like a concise email to a US tech hiring manager. Direct, specific, no fluff. Show impact with numbers.",
}


async def _draft_body(
    jd_text: str,
    intel: Dict[str, Any],
    highlights: Dict[str, Any],
    requirement_map: List[Dict[str, str]],
    company_research: Dict[str, Any],
    company_type: str,
    tone: str,
    length: str,
) -> str:
    """Generate the cover letter body in simple Indian English."""
    company = intel["company"]
    role = intel["role"]
    lo, hi = _WORD_TARGETS.get(length, (200, 280))
    para_targets = _PARA_WORD_TARGETS.get(length, [(55, 80), (80, 120), (50, 80)])

    # v3.0.0: Tone is always simple Indian English, with company-type variation
    company_tone = _COMPANY_TONE_HINTS.get(company_type, _COMPANY_TONE_HINTS["us_tech"])

    # Build context strings
    top_mapped = highlights.get("mapped_experiences", [])[:3]
    achievements_str = ""
    for i, m in enumerate(top_mapped):
        achievements_str += f"- {m.get('achievement', '')} [maps to JD requirement: {m.get('maps_to', '')}]\n"
    if not achievements_str:
        achievements_str = "\n".join(f"- {a}" for a in highlights.get("achievements", [])[:3]) or "Strong professional background"

    skills_str = ", ".join(highlights.get("skills", [])[:8]) or "relevant technical skills"
    strongest_company = highlights.get("strongest_company", "")
    degree = highlights.get("degree_name", "")

    # Requirement mapping for paragraph 2
    mapping_str = ""
    for m in requirement_map[:4]:
        if isinstance(m, dict):
            mapping_str += f"- They need: {m.get('requirement', '')} → I have: {m.get('proof_phrase', m.get('best_match', ''))}\n"

    # Company research for paragraph 1
    research_str = ""
    if company_research.get("what_they_build"):
        research_str += f"What they build: {company_research['what_they_build']}\n"
    if company_research.get("engineering_reputation"):
        research_str += f"Known for: {company_research['engineering_reputation']}\n"
    if company_research.get("why_someone_would_join"):
        research_str += f"Why join: {company_research['why_someone_would_join']}\n"

    team_ctx = f"Team: {intel['team_name']} — {intel['team_mission']}" if intel.get("team_name") else ""
    product_ctx = f"Product/Platform: {intel['company_product']}" if intel.get("company_product") else ""
    biz_ctx = f"Business problem: {intel['business_problem']}" if intel.get("business_problem") else ""

    # Culture signals woven into instructions
    culture_signals = intel.get("culture_signals", [])
    culture_instruction = ""
    if culture_signals:
        culture_instruction = f"""
CULTURE FIT: The JD mentions these values: {', '.join(culture_signals)}.
Show one of these naturally through your experience — do not just state the value.
Example: If they value "collaboration", say "I worked with the firmware team to..." not "I am collaborative"."""

    # Day-to-day work for paragraph 3
    daily_work = intel.get("day_to_day_work", [])
    daily_str = ", ".join(daily_work[:3]) if daily_work else ""

    # v3.0.0: JD tech stack for keyword echo
    tech_stack = intel.get("tech_stack", [])
    tech_echo_str = ", ".join(tech_stack[:6]) if tech_stack else ""

    # Build banned phrases string
    banned_str = ", ".join(sorted(list(_BANNED_PHRASES)[:25]))

    system = f"""You are an Indian engineer from Hyderabad/Bangalore writing a cover letter in English.

YOUR ENGLISH STYLE:
- Simple spoken Indian English. IELTS band 6.5 level.
- Short sentences. 10-18 words per sentence maximum.
- Use simple words: "built", "worked on", "handled", "fixed", "wrote", "tested", "set up"
- Do NOT use: "spearheaded", "orchestrated", "leveraged", "utilized", "pioneered", "harnessed"
- Do NOT use complex subordinate clauses or nested sentences
- Write like you are explaining your work to a senior engineer over chai, not writing a formal essay
- Slight Indian English patterns are OK: "I was working on..." "We had to handle..." "The team used to..."
- Use "and" not "&". No em-dashes. No semicolons.
- Contractions OK: "didn't", "wasn't", "couldn't"

{company_tone}

STRICT RULES:
1. NEVER use these phrases: {banned_str}
2. NEVER mention GPA, graduation date, coursework, university name, or academic grades
3. First-person singular only
4. {lo}-{hi} words total across all 3 paragraphs
5. Exactly 3 paragraphs separated by blank lines
6. No salutation, no signature — body paragraphs ONLY
7. No exclamation marks
8. Each paragraph must have a clear topic, not repeat the others
9. At least 2 specific technologies from the JD must appear naturally in the letter
10. Do NOT start any sentence with "I am writing to" or "I am excited to" or "I am confident that"
11. Do NOT end with "I look forward to hearing from you" — find a more natural close"""

    user = f"""Write a cover letter body for:

COMPANY: {company}
ROLE: {role}
COMPANY TYPE: {company_type}
{team_ctx}
{product_ctx}
{biz_ctx}

COMPANY RESEARCH:
{research_str if research_str else 'No specific research available — use JD context only'}

MY STRONGEST WORK ACHIEVEMENTS (ranked by relevance to this JD):
{achievements_str}

REQUIREMENT-TO-EXPERIENCE MAP:
{mapping_str if mapping_str else 'Use achievements above to demonstrate JD fit'}

MY SKILLS: {skills_str}
{f'MY DEGREE: {degree}' if degree else ''}
{f'STRONGEST RELEVANT COMPANY: {strongest_company}' if strongest_company else ''}

JD TECHNOLOGIES TO MENTION NATURALLY: {tech_echo_str if tech_echo_str else 'Use technologies from achievements'}

{culture_instruction}

JOB DESCRIPTION (for context):
{jd_text[:2500]}

WRITE 3 PARAGRAPHS:

First paragraph ({para_targets[0][0]}-{para_targets[0][1]} words):
Start with something specific about {company} — their product, their technical challenge, or what their team works on. Connect it to your background in 1-2 sentences. Do not say "I am writing to express my interest." Instead, jump straight into why their work connects to yours.

GOOD EXAMPLE: "I have been following {company}'s work on [specific product]. My experience building [specific thing] at [company] gave me a good understanding of the kind of problems your {intel.get('team_name', 'team')} handles."
BAD EXAMPLE: "I am excited to apply for the {role} position at {company}. I believe my experience makes me a perfect fit."

Second paragraph ({para_targets[1][0]}-{para_targets[1][1]} words):
Pick your 1-2 strongest achievements that directly map to their requirements. Be specific — name the tool, the problem, the result. Use numbers if you have them. Show you already did similar work.

GOOD EXAMPLE: "At [company], I built a [specific thing] using [JD technology]. It [specific result]. This is similar to what your team needs for [JD requirement]."
BAD EXAMPLE: "Throughout my career, I have developed a proven track record of delivering innovative solutions that drive impactful results across cross-functional teams."

Third paragraph ({para_targets[2][0]}-{para_targets[2][1]} words):
{f'Based on the JD, the day-to-day work involves: {daily_str}. ' if daily_str else ''}Say what you would work on in the first few months. Be specific to the role. End with a simple, natural close.

GOOD EXAMPLE: "In the first few months, I would focus on [specific JD task]. Happy to discuss this further whenever works for you."
BAD EXAMPLE: "I am confident that my unique combination of skills positions me to make a meaningful impact. I look forward to hearing from you at your earliest convenience."

Write naturally. Short sentences. Simple words. Like a real person."""

    body = await _chat_text(system, user, temperature=0.75)

    # Light cleanup — NO LaTeX escaping here (done later)
    body = re.sub(r"^\s*(?:dear\s|sincerely|regards|best|yours|warmly|respectfully).*$",
                  "", body, flags=re.IGNORECASE | re.MULTILINE)
    body = re.sub(r"^\s*[#*>\-•]\s+", "", body, flags=re.MULTILINE)
    body = re.sub(r"!+", ".", body)  # No exclamation marks

    # Strip banned phrases that slipped through
    for phrase in _BANNED_PHRASES:
        body = re.sub(rf"\b{re.escape(phrase)}\b", "", body, flags=re.IGNORECASE)

    body = _strip_academic(body, keep_degree_name=bool(degree))
    body = re.sub(r"\s{2,}", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


# ============================================================
# ✅ STEP 4: 6-axis quality rubric — v3.0.0 REWRITTEN
# ============================================================

async def _score_and_repair(
    body: str,
    intel: Dict[str, Any],
    highlights: Dict[str, Any],
    requirement_map: List[Dict[str, str]],
    company_type: str,
    length: str,
    jd_text: str,
) -> str:
    """Score on 6 axes, repair specific weak areas."""
    company = intel["company"]
    role = intel["role"]
    lo, hi = _WORD_TARGETS.get(length, (200, 280))
    word_count = len(body.split())
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    tech_stack = intel.get("tech_stack", [])

    # ── Scoring prompt ──────────────────────────────────────────
    prompt = f"""Score this cover letter on 6 axes (0-3 each). Be strict.

COVER LETTER:
{body}

TARGET: {role} at {company}
JD TECH STACK: {', '.join(tech_stack[:6])}
COMPANY TYPE: {company_type}

AXES:
1. company_specificity (0-3): Does paragraph 1 reference something SPECIFIC about {company}? (product name, team, technical challenge — not just "your company")
   3 = names specific product/team/challenge
   2 = references industry/domain correctly
   1 = generic company praise
   0 = no company reference

2. achievement_concreteness (0-3): Does paragraph 2 have a SPECIFIC quantified achievement?
   3 = specific project + tool + number
   2 = specific project + tool, no number
   1 = vague "improved performance"
   0 = no concrete achievement

3. jd_keyword_echo (0-3): How many JD technologies appear naturally?
   3 = 3+ JD technologies mentioned
   2 = 2 JD technologies
   1 = 1 JD technology
   0 = no JD technologies

4. human_voice (0-3): Does it sound like a real Indian engineer, not ChatGPT?
   3 = natural, simple sentences, sounds human
   2 = mostly natural, one formulaic phrase
   1 = template-like, predictable structure
   0 = obvious ChatGPT — banned phrases, complex subordinate clauses, "proven track record" type language

5. forward_close (0-3): Does paragraph 3 mention specific work they would do?
   3 = names specific JD task they would tackle
   2 = general area of contribution
   1 = generic "contribute to your team"
   0 = no forward-looking content

6. word_count_fit (0-3): Is it {lo}-{hi} words with 3 paragraphs?
   3 = within range, exactly 3 paragraphs
   2 = slightly off range (within 20 words), 3 paragraphs
   1 = off range or wrong paragraph count
   0 = way off

Return STRICT JSON:
{{
    "scores": {{
        "company_specificity": 2,
        "achievement_concreteness": 2,
        "jd_keyword_echo": 1,
        "human_voice": 2,
        "forward_close": 2,
        "word_count_fit": 3
    }},
    "total": 12,
    "weakest_axis": "jd_keyword_echo",
    "issues": ["specific issue 1", "specific issue 2"],
    "banned_phrases_found": ["any banned phrases still present"]
}}"""

    try:
        score_data = await _chat_json(prompt)
        scores = score_data.get("scores", {})
        total = score_data.get("total", 18)
        weakest = score_data.get("weakest_axis", "")
        issues = score_data.get("issues", [])
        banned_found = score_data.get("banned_phrases_found", [])
    except Exception:
        log_event("cl_scoring_fail")
        # Do basic checks only
        scores = {}
        total = 18
        weakest = ""
        issues = []
        banned_found = []

    # ── Additional deterministic checks ─────────────────────────
    body_lower = body.lower()

    # Check company mention
    if company.lower() not in body_lower:
        issues.append(f"Company name '{company}' not mentioned anywhere")
        if not weakest:
            weakest = "company_specificity"

    # Check word count hard limits
    if word_count < lo - 20:
        issues.append(f"Too short: {word_count} words (need {lo}-{hi})")
    elif word_count > hi + 30:
        issues.append(f"Too long: {word_count} words (need {lo}-{hi})")

    # Check paragraph count
    if len(paras) < 2:
        issues.append("Fewer than 2 paragraphs")
    elif len(paras) > 5:
        issues.append("More than 5 paragraphs")

    # Check JD keyword echo
    tech_found = [t for t in tech_stack if t.lower() in body_lower]
    if len(tech_found) < 2 and tech_stack:
        missing_tech = [t for t in tech_stack[:4] if t.lower() not in body_lower]
        issues.append(f"Missing JD technologies: {', '.join(missing_tech[:3])}")
        if not weakest:
            weakest = "jd_keyword_echo"

    # Check banned phrases (deterministic)
    for phrase in _BANNED_PHRASES:
        if phrase in body_lower:
            banned_found.append(phrase)

    # Check academic content
    for academic in ["gpa", "cgpa", "coursework", "graduation", "dean's list"]:
        if academic in body_lower:
            issues.append(f"Contains academic content: '{academic}'")
            break

    # ── Decision: repair or pass ────────────────────────────────
    if not issues and total >= 14:
        log_event("cl_quality_passed", {"total": total, "word_count": word_count})
        return body

    log_event("cl_quality_issues", {"total": total, "issues": issues[:5], "weakest": weakest})

    # ── Targeted repair ─────────────────────────────────────────
    # Build repair instructions based on weakest axis
    repair_instructions = []

    if weakest == "company_specificity" or company.lower() not in body_lower:
        what_they_build = ""
        if intel.get("company_product"):
            what_they_build = intel["company_product"]
        elif intel.get("business_problem"):
            what_they_build = intel["business_problem"]
        repair_instructions.append(
            f"Paragraph 1 must reference {company} specifically. "
            f"Mention their product/challenge: '{what_they_build}'"
        )

    if weakest == "jd_keyword_echo":
        missing_tech = [t for t in tech_stack[:4] if t.lower() not in body_lower]
        if missing_tech:
            repair_instructions.append(
                f"Naturally mention these technologies: {', '.join(missing_tech[:3])}"
            )

    if weakest == "human_voice" or banned_found:
        repair_instructions.append(
            "Rewrite in simpler Indian English. Short sentences. "
            "Remove any phrase that sounds like ChatGPT wrote it. "
            f"Remove these specific phrases: {', '.join(banned_found[:3])}"
        )

    if weakest == "achievement_concreteness":
        top_achievement = (highlights.get("achievements", []) or [""])[0]
        repair_instructions.append(
            f"Paragraph 2 needs a specific achievement with a number. Use: '{top_achievement}'"
        )

    if weakest == "forward_close":
        daily_work = intel.get("day_to_day_work", [])
        if daily_work:
            repair_instructions.append(
                f"Paragraph 3 must mention specific work: '{daily_work[0]}'"
            )

    if word_count < lo - 20:
        repair_instructions.append(f"Expand to at least {lo} words. Add more detail to paragraph 2.")
    elif word_count > hi + 30:
        repair_instructions.append(f"Cut to under {hi} words. Remove redundant sentences.")

    if not repair_instructions:
        repair_instructions.append("Fix the general quality. Keep simple Indian English voice.")

    repair_prompt = f"""Fix these specific issues in this cover letter body.
Keep the same simple Indian English voice. Short sentences. No ChatGPT language.

ISSUES TO FIX:
{chr(10).join(f'- {r}' for r in repair_instructions)}

CURRENT DRAFT:
{body}

REQUIREMENTS:
- Company: {company}, Role: {role}
- {lo}-{hi} words, exactly 3 paragraphs
- NO academic content
- Simple Indian English — IELTS band 6.5
- Must mention {company} by name in paragraph 1
- At least 2 JD technologies: {', '.join(tech_stack[:4])}
- Body paragraphs ONLY — no salutation/signature
- No exclamation marks

Return ONLY the improved body text. Nothing else."""

    try:
        repaired = await _chat_text(
            "You are fixing specific issues in a cover letter. Keep the simple Indian English voice. "
            "Short sentences. No ChatGPT patterns. Fix only what is listed.",
            repair_prompt,
            temperature=0.6,
        )
        repaired = _strip_academic(repaired, keep_degree_name=bool(highlights.get("degree_name")))
        repaired = re.sub(r"^\s*(?:dear\s|sincerely|regards).*$", "", repaired,
                          flags=re.IGNORECASE | re.MULTILINE)
        repaired = re.sub(r"!+", ".", repaired)
        repaired = re.sub(r"\s{2,}", " ", repaired)
        repaired = re.sub(r"\n{3,}", "\n\n", repaired).strip()

        repaired_wc = len(repaired.split())
        if repaired and repaired_wc >= lo - 20 and repaired_wc <= hi + 30:
            return repaired
        log_event("cl_repair_word_count_off", {"repaired_wc": repaired_wc, "target": f"{lo}-{hi}"})
    except Exception as e:
        log_event("cl_repair_fail", {"error": str(e)})

    return body


# ============================================================
# 🔍 STEP 4b: Fact-checking — v3.0.0 NEW
# ============================================================

async def _fact_check(
    body: str,
    resume_text: str,
    intel: Dict[str, Any],
) -> str:
    """Verify all claims in the cover letter exist in the resume."""
    if not resume_text:
        return body

    prompt = f"""Check if every factual claim in this cover letter is supported by the resume.

COVER LETTER:
{body}

RESUME:
{resume_text[:5000]}

Check for:
1. Companies mentioned — are they in the resume?
2. Technologies claimed — are they in the resume?
3. Metrics/numbers — are they in the resume or reasonable?
4. Projects described — do they match resume content?
5. Claims of leading/owning/building — are they exaggerated vs resume?

Return STRICT JSON:
{{
    "all_facts_verified": true,
    "issues": [
        {{
            "claim": "the problematic claim",
            "issue": "not in resume / exaggerated / invented",
            "suggested_fix": "how to fix it"
        }}
    ]
}}"""

    try:
        data = await _chat_json(prompt)
        if data.get("all_facts_verified", True):
            log_event("cl_factcheck_passed")
            return body

        issues = data.get("issues", [])
        if not issues:
            return body

        log_event("cl_factcheck_issues", {"count": len(issues)})

        # Build targeted fix
        fixes = []
        for issue in issues[:3]:
            if isinstance(issue, dict):
                fixes.append(f"- Claim: '{issue.get('claim', '')}' → {issue.get('suggested_fix', 'remove or correct')}")

        if not fixes:
            return body

        fix_prompt = f"""Fix these factual errors in the cover letter. Do NOT invent new facts.
Only use information from the resume.

ERRORS:
{chr(10).join(fixes)}

CURRENT:
{body}

RESUME (for reference):
{resume_text[:3000]}

Keep the same simple Indian English voice. Same structure. Fix only the errors listed.
Return ONLY the corrected body text."""

        fixed = await _chat_text(
            "Fix factual errors in this cover letter. Keep the voice and structure identical. "
            "Only change the specific claims listed.",
            fix_prompt,
            temperature=0.3,
        )
        if fixed and len(fixed.split()) >= len(body.split()) * 0.7:
            return fixed.strip()
    except Exception as e:
        log_event("cl_factcheck_fail", {"error": str(e)})

    return body


# ============================================================
# ✨ STEP 5: Humanize (optional, failure-safe) + tone guard
# ============================================================

async def _humanize(body: str, tone: str) -> str:
    """Send to internal humanize service. Returns original on failure."""
    api_base = (getattr(config, "API_BASE_URL", "") or "").rstrip("/") or "http://127.0.0.1:8000"
    url = f"{api_base}/api/superhuman/rewrite"
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            r = await client.post(url, json={
                "text": body, "mode": "coverletter",
                "tone": tone, "latex_safe": True,
            })
        r.raise_for_status()
        data = r.json()
        result = data.get("rewritten") or data.get("text") or ""
        if result and len(result) >= len(body) * 0.5:
            return _strip_academic(result)
        log_event("humanize_short_result", {"original_len": len(body), "result_len": len(result)})
        return body
    except Exception as e:
        log_event("humanize_fail", {"error": str(e)})
        return body


async def _post_humanize_tone_guard(body: str) -> str:
    """v3.0.0: After humanize, verify the tone is still simple Indian English.
    If humanize converted it to polished American English, fix it back."""
    # Check for AI-tell phrases that humanize might have introduced
    body_lower = body.lower()
    ai_tells_found = []
    for phrase in _BANNED_PHRASES:
        if phrase in body_lower:
            ai_tells_found.append(phrase)

    # Check sentence complexity — if average sentence length > 22 words, it got too polished
    sentences = re.split(r'[.!?]+', body)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_words = sum(len(s.split()) for s in sentences) / max(1, len(sentences))

    if not ai_tells_found and avg_words <= 22:
        return body

    log_event("cl_tone_guard_triggered", {
        "ai_tells": ai_tells_found[:3],
        "avg_sentence_words": round(avg_words, 1),
    })

    try:
        fixed = await _chat_text(
            "You are simplifying English text for an Indian engineer (IELTS 6.5 level). "
            "Break long sentences into shorter ones. Replace fancy words with simple ones. "
            "Keep all factual content the same.",
            f"""Simplify this text. Short sentences (10-18 words each). Simple vocabulary.
Remove these phrases: {', '.join(ai_tells_found[:5])}

TEXT:
{body}

Return ONLY the simplified text. Same 3 paragraphs. Same facts.""",
            temperature=0.5,
        )
        if fixed and len(fixed.split()) >= len(body.split()) * 0.7:
            return fixed.strip()
    except Exception:
        pass

    # Fallback: just strip banned phrases
    for phrase in ai_tells_found:
        body = re.sub(rf"\b{re.escape(phrase)}\b", "", body, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", body).strip()


# ============================================================
# 📄 STEP 6: Template injection — v3.0.0 UPDATED salutation
# ============================================================

def _build_salutation(intel: Dict[str, Any], company_type: str) -> str:
    """v3.0.0: Cultural awareness — formal for Indian companies, first-name for US startups."""
    name = intel.get("hiring_manager_name", "").strip()
    title = intel.get("hiring_manager_title", "").strip()

    if not name:
        return "Dear Hiring Manager,"

    parts = name.split()
    if len(parts) < 2:
        # Single name — use as-is
        if company_type == "indian":
            return f"Dear {_latex_escape(name)},"
        return f"Dear {_latex_escape(name)},"

    first = parts[0]
    last = parts[-1]

    # For Indian companies or enterprise: use Mr./Ms./Dr. Last
    if company_type in ("indian", "enterprise"):
        # Check for Dr.
        if "dr" in title.lower() or "phd" in title.lower() or "doctor" in title.lower():
            return f"Dear Dr. {_latex_escape(last)},"
        return f"Dear Mr./Ms. {_latex_escape(last)},"

    # For US tech / startups: first name is fine
    return f"Dear {_latex_escape(first)},"


def _fill_template(
    tex: str,
    company: str,
    role: str,
    candidate: str,
    date_str: str,
    email: str,
    phone: str,
    citystate: str,
    salutation: str,
) -> str:
    """Replace all template placeholders with escaped values."""
    replacements = {
        "{{DATE}}":           _latex_escape(date_str),
        "{{COMPANY}}":        _latex_escape(company),
        "{{ROLE}}":           _latex_escape(role),
        "{{CANDIDATE_NAME}}": _latex_escape(candidate),
        "{{NAME}}":           _latex_escape(candidate),
        "{{EMAIL}}":          _latex_escape(email),
        "{{PHONE}}":          _latex_escape(phone),
        "{{CITYSTATE}}":      _latex_escape(citystate) if citystate else "",
        "{{SALUTATION}}":     salutation,
    }

    for placeholder, value in replacements.items():
        tex = tex.replace(placeholder, value)

    for placeholder, value in replacements.items():
        alt = placeholder.replace("{{", "%<<").replace("}}", ">>%")
        tex = tex.replace(alt, value)

    for p in ("{{EMPLOYER_ADDRESS}}", "%<<EMPLOYER_ADDRESS>>%"):
        tex = re.sub(r"[^\n]*" + re.escape(p) + r"[^\n]*\n?", "", tex)

    if not citystate:
        tex = re.sub(r"^[^\S\n]*\\\\[^\S\n]*\n", "", tex, flags=re.MULTILINE)

    return tex


def _inject_body(template_tex: str, body_tex: str) -> str:
    """Inject body text between salutation and signoff, or at anchor."""
    pat = r"(Dear[^\n]*?,\s*\n)([\s\S]*?)(\n\s*(?:Sincerely|Best regards|Regards),)"
    m = re.search(pat, template_tex, flags=re.IGNORECASE)
    if m:
        return template_tex[:m.start(1)] + m.group(1) + "\n" + body_tex + "\n" + template_tex[m.start(3):]

    anchor = r"(%-+\s*BODY[- ]START\s*-+%)([\s\S]*?)(%-+\s*BODY[- ]END\s*-+%)"
    m2 = re.search(anchor, template_tex, flags=re.IGNORECASE)
    if m2:
        return template_tex[:m2.start()] + m2.group(1) + "\n" + body_tex + "\n" + m2.group(3) + template_tex[m2.end():]

    end_doc = template_tex.rfind(r"\end{document}")
    if end_doc >= 0:
        return template_tex[:end_doc] + "\n" + body_tex + "\n\n" + template_tex[end_doc:]

    return template_tex + "\n\n" + body_tex + "\n\\end{document}\n"


# ============================================================
# 🚀 MAIN ENDPOINT — v3.0.0
# ============================================================

@router.post("")
async def generate_coverletter(
    jd_text:       str  = Form(...),
    resume_tex:    str  = Form(""),
    use_humanize:  bool = Form(True),
    tone:          str  = Form("balanced"),
    length:        str  = Form("standard"),
):
    """Generate an authentic cover letter in simple Indian English. v3.0.0."""
    if not (config.OPENAI_API_KEY or "").strip():
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY missing.")
    if not (jd_text or "").strip():
        raise HTTPException(status_code=400, detail="jd_text is required.")

    tone = (tone or "balanced").strip().lower()
    length = (length or "standard").strip().lower()
    if length not in _WORD_TARGETS:
        length = "standard"

    # ── 1. Extract JD intelligence ──────────────────────────────
    intel = await _extract_jd_intelligence(jd_text)
    company = intel["company"]
    role = intel["role"]
    log_event("cl_start", {"company": company, "role": role, "tone": tone,
                           "length": length, "version": "v3.0.0"})

    # ── 1b. Company research ────────────────────────────────────
    company_research = await _research_company(company, role, jd_text)

    # ── 1c. Detect company type ─────────────────────────────────
    company_type = _detect_company_type(jd_text, company)
    log_event("cl_company_type", {"company": company, "type": company_type})

    # ── 2. Extract resume highlights (JD-prioritized) ───────────
    highlights = await _extract_resume_highlights(resume_tex, intel, jd_text)

    # ── 2b. Map requirements to experience ──────────────────────
    requirement_map = await _map_requirements_to_experience(intel, highlights)
    log_event("cl_requirement_map", {"mappings": len(requirement_map)})

    # ── 3. Draft body ───────────────────────────────────────────
    body = await _draft_body(
        jd_text, intel, highlights, requirement_map,
        company_research, company_type, tone, length,
    )

    # ── 4. Score and repair ─────────────────────────────────────
    body = await _score_and_repair(
        body, intel, highlights, requirement_map,
        company_type, length, jd_text,
    )

    # ── 4b. Fact-check against resume ───────────────────────────
    body = await _fact_check(body, resume_tex, intel)

    # ── 5. Humanize (optional) ──────────────────────────────────
    if use_humanize:
        body = await _humanize(body, tone)
        # v3.0.0: Post-humanize tone guard
        body = await _post_humanize_tone_guard(body)

    # ── 5b. Final banned phrase cleanup (deterministic) ─────────
    body_lower = body.lower()
    for phrase in _BANNED_PHRASES:
        if phrase in body_lower:
            body = re.sub(rf"\b{re.escape(phrase)}\b", "", body, flags=re.IGNORECASE)
            body_lower = body.lower()  # refresh
    body = re.sub(r"\s{2,}", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    # ── 5c. Word count hard enforcement ─────────────────────────
    lo, hi = _WORD_TARGETS.get(length, (200, 280))
    word_count = len(body.split())
    if word_count > hi + 20:
        # Trim from the longest paragraph
        paras = [p.strip() for p in body.split("\n\n") if p.strip()]
        if len(paras) >= 3:
            longest_idx = max(range(len(paras)), key=lambda i: len(paras[i].split()))
            sentences = re.split(r'(?<=[.!?])\s+', paras[longest_idx])
            if len(sentences) > 2:
                # Remove the least specific sentence (shortest)
                shortest_idx = min(range(len(sentences)), key=lambda i: len(sentences[i].split()))
                sentences.pop(shortest_idx)
                paras[longest_idx] = " ".join(sentences)
                body = "\n\n".join(paras)
                log_event("cl_word_trim", {"before": word_count, "after": len(body.split())})

    # ── 6. LaTeX-escape the body ONCE ───────────────────────────
    body_tex = _latex_escape(body)

    # Convert paragraph breaks to LaTeX paragraph breaks
    body_tex = re.sub(r"\n\n+", "\n\n\\vspace{0.5em}\n\n", body_tex)

    # ── 7. Load template ────────────────────────────────────────
    base_path = config.BASE_COVERLETTER_PATH
    try:
        with open(base_path, encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template not found: {base_path}")

    # ── 8. Fill template ────────────────────────────────────────
    today_str = datetime.now().strftime("%B %d, %Y")
    candidate = getattr(config, "CANDIDATE_NAME", "Sri Akash Kadali")
    email = getattr(config, "APPLICANT_EMAIL", "kadali18@umd.edu")
    phone = getattr(config, "APPLICANT_PHONE", "+1 240-726-9356")
    citystate = getattr(config, "APPLICANT_CITYSTATE", "")
    salutation = _build_salutation(intel, company_type)

    filled = _fill_template(
        template, company, role, candidate, today_str,
        email, phone, citystate, salutation,
    )

    # ── 9. Inject body ──────────────────────────────────────────
    final_tex = _inject_body(filled, body_tex)
    final_tex = render_final_tex(final_tex)

    # ── 10. Compile ─────────────────────────────────────────────
    pdf_bytes = compile_latex_safely(final_tex) or b""
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    # ── 11. Save outputs ────────────────────────────────────────
    out_pdf_path = config.get_coverletter_pdf_path(company, role)
    ensure_dir(out_pdf_path.parent)
    if pdf_bytes:
        out_pdf_path.write_bytes(pdf_bytes)

    company_slug = safe_filename(company)
    role_slug = safe_filename(role)
    context_key = f"{company_slug}__{role_slug}"

    ctx_dir = config.get_contexts_dir()
    ensure_dir(ctx_dir)
    ctx_path = ctx_dir / f"{context_key}.json"

    existing: Dict[str, Any] = {}
    if ctx_path.exists():
        try:
            existing = json.loads(ctx_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    # Final word count
    final_word_count = len(body.split())

    # JD keyword echo stats
    tech_stack = intel.get("tech_stack", [])
    tech_found = [t for t in tech_stack if t.lower() in body.lower()]

    context_payload = {
        **existing,
        "key": context_key,
        "company": company,
        "role": role,
        "jd_text": jd_text,
        "cover_letter": {
            "tex": final_tex,
            "pdf_path": str(out_pdf_path),
            "pdf_b64": pdf_b64,
            "tone": tone,
            "length": length,
            "humanized": bool(use_humanize),
            "word_count": final_word_count,
            "company_type": company_type,
            "jd_tech_echoed": tech_found,
            "requirement_mappings": len(requirement_map),
        },
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    ctx_path.write_text(json.dumps(context_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    log_event("cl_generated", {
        "company": company, "role": role, "tone": tone,
        "length": length, "humanized": use_humanize,
        "words": final_word_count, "pdf_bytes": len(pdf_bytes),
        "company_type": company_type,
        "tech_echoed": len(tech_found),
        "requirement_mappings": len(requirement_map),
        "version": "v3.0.0",
    })

    return JSONResponse({
        "company": company,
        "role": role,
        "tone": tone,
        "use_humanize": use_humanize,
        "tex_string": final_tex,
        "pdf_base64": pdf_b64,
        "pdf_path": str(out_pdf_path),
        "context_key": context_key,
        "context_path": str(ctx_path),
        "word_count": final_word_count,
        "salutation_used": salutation,
        "company_type": company_type,
        "jd_tech_echoed": tech_found,
        "jd_tech_total": len(tech_stack),
        "requirement_mappings": len(requirement_map),
        "id": context_key,
        "memory_id": context_key,
        "version": "v3.0.0",
    })