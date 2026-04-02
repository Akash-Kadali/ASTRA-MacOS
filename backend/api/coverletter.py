# ============================================================
#  HIREX v2.0.0 — Cover Letter Generation
#  ------------------------------------------------------------
#  CHANGELOG v2.0.0 (vs v1.0.0):
#   REWRITE  Single focused GPT call replaces scattered extract+draft+repair chain
#   REWRITE  Structured 3-paragraph output with per-paragraph word targets
#   FIX      LaTeX escaping applied ONCE at the end, not layered
#   FIX      Body injection handles all template variants reliably
#   FIX      Humanize failure no longer corrupts output
#   FIX      Academic content stripped from resume BEFORE sending to GPT
#   REMOVE   Redundant _postprocess_body double-escaping
#   ADD      GPT-based company research for ANY company (not just hardcoded)
#   ADD      Quality scoring with targeted rewrite (not full redraft)
#   ADD      Paragraph-level structure enforcement
#   ADD      Word count validation before compile
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


_MODEL = getattr(config, "COVERLETTER_MODEL", None) or "gpt-5.4-mini"


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


async def _chat_text(system: str, user: str) -> str:
    client = _get_openai_client()
    resp = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


async def _chat_json(prompt: str) -> dict:
    client = _get_openai_client()
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
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
        )
        return _json_from_text(resp.choices[0].message.content or "", {})


# ============================================================
# 🔒 LaTeX utilities
# ============================================================

_LATEX_SPECIAL = {
    "&": r" and ", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}",
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


def _strip_academic(text: str) -> str:
    """Remove GPA, graduation dates, coursework, degree mentions."""
    if not text:
        return ""
    patterns = [
        r"\bC?GPA\s*[:\-]?\s*\d+(\.\d+)?(/\d+(\.\d+)?)?",
        r"\b\d+(\.\d+)?\s*(GPA|CGPA)\b",
        r"\b(graduat(ed?|ing|ion))\s*(in|from|date)?\s*\d{4}\b",
        r"\b(class of|expected|graduating)\s*\d{4}\b",
        r"\b(relevant\s+)?coursework\b[:\s]*[^.]*\.",
        r"\bcourses?\s+(include|including|such as)[^.]*\.",
        r"\b(bachelor'?s?|master'?s?|ph\.?d\.?|b\.?s\.?|m\.?s\.?)\s*(degree)?\s*(in\s+\w+)?\s*,?\s*\d{4}",
        r"\buniversity[^,]*,?\s*\d{4}",
        r"\bcollege[^,]*,?\s*\d{4}",
        r"\bdean'?s?\s*list\b",
        r"\bcum\s+laude\b",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()


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
    "team_name": "specific team if mentioned, else empty string",
    "team_mission": "what this team does, 1 sentence, else empty string",
    "city_state": "office location if mentioned, else empty string",
    "key_requirements": ["top 5 must-have skills/experiences"],
    "tech_stack": ["specific technologies mentioned"],
    "business_problem": "what business problem this role solves, 1 sentence",
    "company_product": "main product/platform this role works on, else empty string",
    "culture_signals": ["2-3 culture/values phrases from the JD"]
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
            "team_name":            (data.get("team_name") or "").strip(),
            "team_mission":         (data.get("team_mission") or "").strip(),
            "city_state":           (data.get("city_state") or "").strip(),
            "key_requirements":     data.get("key_requirements", [])[:6],
            "tech_stack":           data.get("tech_stack", [])[:8],
            "business_problem":     (data.get("business_problem") or "").strip(),
            "company_product":      (data.get("company_product") or "").strip(),
            "culture_signals":      data.get("culture_signals", [])[:3],
        }
    except Exception as e:
        log_event("jd_intel_fail", {"error": str(e)})
        return {"company": "Company", "role": "Role", "hiring_manager_name": "",
                "team_name": "", "team_mission": "", "city_state": "",
                "key_requirements": [], "tech_stack": [], "business_problem": "",
                "company_product": "", "culture_signals": []}


# ============================================================
# 🧠 STEP 2: Extract resume highlights (single GPT call)
# ============================================================

async def _extract_resume_highlights(resume_text: str) -> Dict[str, Any]:
    """Pull the 3 strongest WORK achievements from the resume."""
    if not (resume_text or "").strip():
        return {"achievements": [], "skills": [], "companies": []}

    cleaned = _strip_academic(resume_text[:5000])
    prompt = f"""Extract professional highlights from this resume. NO academic content.

RESUME:
{cleaned}

Return STRICT JSON:
{{
    "top_3_achievements": [
        "1-sentence WORK achievement with quantified result",
        "1-sentence WORK achievement with quantified result",
        "1-sentence WORK achievement with quantified result"
    ],
    "relevant_skills": ["top 6 technical skills"],
    "companies_worked": ["company names"]
}}

RULES: Only WORK experience. No GPA, no graduation, no coursework."""
    try:
        data = await _chat_json(prompt)
        return {
            "achievements": data.get("top_3_achievements", [])[:3],
            "skills":       data.get("relevant_skills", [])[:6],
            "companies":    data.get("companies_worked", [])[:4],
        }
    except Exception:
        return {"achievements": [], "skills": [], "companies": []}


# ============================================================
# 📝 STEP 3: Draft cover letter body (single focused GPT call)
# ============================================================

_WORD_TARGETS = {
    "short":    (130, 180),
    "standard": (200, 280),
    "long":     (300, 400),
}

_BANNED_PHRASES = {
    "passionate", "dynamic", "cutting edge", "synergy", "dream job",
    "perfect fit", "excited to apply", "thrilled", "honored", "privileged",
    "game-changer", "revolutionary", "always wanted", "since childhood",
    "humbly request", "blown away", "astonishing", "leverage synergies",
    "results-driven", "team player", "fast-paced environment",
}


async def _draft_body(
    jd_text: str,
    intel: Dict[str, Any],
    highlights: Dict[str, Any],
    tone: str,
    length: str,
) -> str:
    """Generate the cover letter body in 3 structured paragraphs."""
    company = intel["company"]
    role = intel["role"]
    lo, hi = _WORD_TARGETS.get(length, (200, 280))

    tone_map = {
        "confident":     "confident and direct — no hedging, but not arrogant",
        "balanced":      "professional yet warm — like writing to a respected colleague",
        "humble":        "thoughtful and genuine — show respect without being servile",
        "conversational": "natural and direct — like a smart friend explaining why this role fits",
    }
    tone_instruction = tone_map.get(tone, tone_map["balanced"])

    # Build context strings
    achievements_str = "\n".join(f"- {a}" for a in highlights.get("achievements", [])[:3]) or "Strong professional background"
    skills_str = ", ".join(highlights.get("skills", [])[:6]) or "relevant technical skills"
    team_ctx = f"Team: {intel['team_name']} — {intel['team_mission']}" if intel.get("team_name") else ""
    product_ctx = f"Product/Platform: {intel['company_product']}" if intel.get("company_product") else ""
    biz_ctx = f"Business problem: {intel['business_problem']}" if intel.get("business_problem") else ""
    reqs_str = ", ".join(intel.get("key_requirements", [])[:5]) or "the role requirements"
    culture_str = ", ".join(intel.get("culture_signals", [])[:3])

    system = f"""You write cover letters that sound like a real person wrote them.

TONE: {tone_instruction}

STRICT RULES:
1. NEVER use these words/phrases: {', '.join(sorted(_BANNED_PHRASES))}
2. NEVER mention GPA, graduation, coursework, university, or academic achievements
3. First-person singular only
4. Use "and" not "&". No em-dashes (—).
5. No duplicate sentences
6. {lo}-{hi} words total
7. Exactly 3 paragraphs separated by blank lines
8. No salutation, no signature — body paragraphs ONLY"""

    user = f"""Write a cover letter body for:

COMPANY: {company}
ROLE: {role}
{team_ctx}
{product_ctx}
{biz_ctx}
Key requirements: {reqs_str}
{f'Culture: {culture_str}' if culture_str else ''}

MY STRONGEST WORK ACHIEVEMENTS:
{achievements_str}

MY SKILLS: {skills_str}

JOB DESCRIPTION (for context):
{jd_text[:3000]}

STRUCTURE:

PARAGRAPH 1 (3-4 sentences):
Why {company} specifically interests you. Reference their actual product/team/challenge — not generic praise. Connect your background naturally. Be specific.

PARAGRAPH 2 (4-5 sentences):
Your strongest relevant WORK achievement. Map your experience to their requirements. Include a specific quantified result. Show you can do what they need.

PARAGRAPH 3 (2-3 sentences):
What you would contribute in the first months. Confident close — express interest in a conversation.

Write naturally. Be specific where you can, general where you must."""

    body = await _chat_text(system, user)

    # Light cleanup — NO LaTeX escaping here (done later)
    body = re.sub(r"^\s*(?:dear\s|sincerely|regards|best|yours).*$", "", body, flags=re.IGNORECASE | re.MULTILINE)
    body = re.sub(r"^\s*[#*>\-•]\s+", "", body, flags=re.MULTILINE)

    # Strip any banned phrases that slipped through
    for phrase in _BANNED_PHRASES:
        body = re.sub(rf"\b{re.escape(phrase)}\b", "", body, flags=re.IGNORECASE)

    body = _strip_academic(body)
    body = re.sub(r"\s{2,}", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


# ============================================================
# ✅ STEP 4: Validate and targeted repair
# ============================================================

async def _validate_and_repair(
    body: str,
    intel: Dict[str, Any],
    highlights: Dict[str, Any],
    length: str,
) -> str:
    """Score the draft and fix specific issues without full redraft."""
    company = intel["company"]
    lo, hi = _WORD_TARGETS.get(length, (200, 280))
    word_count = len(body.split())
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    issues: List[str] = []

    # Check company mention
    if company.lower() not in body.lower():
        issues.append(f"Company name '{company}' not mentioned")

    # Check paragraph count
    if len(paras) < 2:
        issues.append("Fewer than 2 paragraphs")
    elif len(paras) > 5:
        issues.append("More than 5 paragraphs — consolidate")

    # Check word count
    if word_count < lo - 30:
        issues.append(f"Too short ({word_count} words, need {lo}-{hi})")
    elif word_count > hi + 40:
        issues.append(f"Too long ({word_count} words, need {lo}-{hi})")

    # Check for banned content
    body_lower = body.lower()
    for phrase in _BANNED_PHRASES:
        if phrase in body_lower:
            issues.append(f"Contains banned phrase: '{phrase}'")
            break  # One is enough to trigger repair

    for academic in ["gpa", "cgpa", "coursework", "graduation", "dean's list"]:
        if academic in body_lower:
            issues.append(f"Contains academic content: '{academic}'")
            break

    # Check forward-looking close
    if not re.search(r"\b(contribute|bring|drive|look forward|conversation|discuss|connect)\b", body_lower):
        issues.append("Missing forward-looking close")

    if not issues:
        log_event("cl_validation_passed", {"word_count": word_count, "paragraphs": len(paras)})
        return body

    log_event("cl_validation_issues", {"issues": issues})

    # Targeted repair
    repair_prompt = f"""Fix these specific issues in this cover letter body:

ISSUES:
{chr(10).join(f'- {i}' for i in issues)}

CURRENT DRAFT:
{body}

REQUIREMENTS:
- Company: {company}, Role: {intel['role']}
- {lo}-{hi} words, exactly 3 paragraphs
- NO academic content (GPA, graduation, coursework)
- NO clichés (passionate, excited, dream job, etc.)
- Must mention {company} in paragraph 1
- Must end with forward-looking close
- Body paragraphs ONLY — no salutation/signature

Return ONLY the improved body text."""

    try:
        repaired = await _chat_text(
            "You are fixing specific issues in a cover letter. Keep what works, fix what doesn't.",
            repair_prompt,
        )
        repaired = _strip_academic(repaired)
        repaired = re.sub(r"^\s*(?:dear\s|sincerely|regards).*$", "", repaired, flags=re.IGNORECASE | re.MULTILINE)
        repaired = re.sub(r"\s{2,}", " ", repaired)
        repaired = re.sub(r"\n{3,}", "\n\n", repaired).strip()

        if repaired and len(repaired.split()) >= lo - 30:
            return repaired
    except Exception as e:
        log_event("cl_repair_fail", {"error": str(e)})

    return body


# ============================================================
# ✨ STEP 5: Humanize (optional, failure-safe)
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
        return body  # Graceful fallback — never corrupt


# ============================================================
# 📄 STEP 6: Template injection
# ============================================================

def _build_salutation(intel: Dict[str, Any]) -> str:
    name = intel.get("hiring_manager_name", "").strip()
    if name:
        first = name.split()[0]
        return f"Dear {_latex_escape(first)},"
    return "Dear Hiring Manager,"


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

    # Also handle %<<KEY>>% style
    for placeholder, value in replacements.items():
        alt = placeholder.replace("{{", "%<<").replace("}}", ">>%")
        tex = tex.replace(alt, value)

    # Remove employer address placeholder if present (we don't inject address blocks)
    for p in ("{{EMPLOYER_ADDRESS}}", "%<<EMPLOYER_ADDRESS>>%"):
        tex = re.sub(r"[^\n]*" + re.escape(p) + r"[^\n]*\n?", "", tex)

    # Remove empty city/state line if no citystate
    if not citystate:
        tex = re.sub(r"^[^\S\n]*\\\\[^\S\n]*\n", "", tex, flags=re.MULTILINE)

    return tex


def _inject_body(template_tex: str, body_tex: str) -> str:
    """Inject body text between salutation and signoff, or at anchor."""
    # Method 1: Between "Dear ...," and "Sincerely,"
    pat = r"(Dear[^\n]*?,\s*\n)([\s\S]*?)(\n\s*(?:Sincerely|Best regards|Regards),)"
    m = re.search(pat, template_tex, flags=re.IGNORECASE)
    if m:
        return template_tex[:m.start(1)] + m.group(1) + "\n" + body_tex + "\n" + template_tex[m.start(3):]

    # Method 2: Anchor comments
    anchor = r"(%-+\s*BODY[- ]START\s*-+%)([\s\S]*?)(%-+\s*BODY[- ]END\s*-+%)"
    m2 = re.search(anchor, template_tex, flags=re.IGNORECASE)
    if m2:
        return template_tex[:m2.start()] + m2.group(1) + "\n" + body_tex + "\n" + m2.group(3) + template_tex[m2.end():]

    # Method 3: Before \end{document}
    end_doc = template_tex.rfind(r"\end{document}")
    if end_doc >= 0:
        return template_tex[:end_doc] + "\n" + body_tex + "\n\n" + template_tex[end_doc:]

    return template_tex + "\n\n" + body_tex + "\n\\end{document}\n"


# ============================================================
# 🚀 MAIN ENDPOINT
# ============================================================

@router.post("")
async def generate_coverletter(
    jd_text:       str  = Form(...),
    resume_tex:    str  = Form(""),
    use_humanize:  bool = Form(True),
    tone:          str  = Form("balanced"),
    length:        str  = Form("standard"),
):
    """Generate an authentic cover letter. v2.0.0."""
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
    log_event("cl_start", {"company": company, "role": role, "tone": tone, "length": length})

    # ── 2. Extract resume highlights ────────────────────────────
    highlights = await _extract_resume_highlights(resume_tex)

    # ── 3. Draft body ───────────────────────────────────────────
    body = await _draft_body(jd_text, intel, highlights, tone, length)

    # ── 4. Validate and repair ──────────────────────────────────
    body = await _validate_and_repair(body, intel, highlights, length)

    # ── 5. Humanize (optional) ──────────────────────────────────
    if use_humanize:
        body = await _humanize(body, tone)

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
    salutation = _build_salutation(intel)

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

    # Save to context store
    ctx_dir = config.get_contexts_dir()
    ensure_dir(ctx_dir)
    ctx_path = ctx_dir / f"{context_key}.json"

    existing: Dict[str, Any] = {}
    if ctx_path.exists():
        try:
            existing = json.loads(ctx_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

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
            "word_count": len(body.split()),
        },
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    ctx_path.write_text(json.dumps(context_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    log_event("cl_generated", {
        "company": company, "role": role, "tone": tone,
        "length": length, "humanized": use_humanize,
        "words": len(body.split()), "pdf_bytes": len(pdf_bytes),
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
        "word_count": len(body.split()),
        "salutation_used": salutation,
        "id": context_key,
        "memory_id": context_key,
    })