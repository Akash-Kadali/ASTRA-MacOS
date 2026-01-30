"""
============================================================
 HIREX v2.1.3 — talk.py
 ------------------------------------------------------------
 "Talk to HIREX" conversational endpoint.
 Answers job-application or interview questions using
 JD + resume (+ cover letter if available) from the stable
 (Company__Role) context files saved in /api/context.

 Author: Sri Akash Kadali
============================================================
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# OpenAI SDK (lazy/defensive import so the router still loads even if SDK is missing)
try:
    from openai import AsyncOpenAI  # type: ignore
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

from backend.core import config
from backend.core.utils import log_event, safe_filename, ensure_dir
from backend.core.security import secure_tex_input

# Prefix keeps both /api/talk (default) and /api/talk/answer (alias) working
router = APIRouter(prefix="/api/talk", tags=["talk"])

# Initialize OpenAI client if SDK is available; defer credential checks to request-time
openai_client = AsyncOpenAI(api_key=getattr(config, "OPENAI_API_KEY", "")) if AsyncOpenAI else None

# Stable location where /api/context saves contexts
CONTEXT_DIR: Path = config.get_contexts_dir()
ensure_dir(CONTEXT_DIR)

# Cheap, reliable summarizer / answer model (override-able from env)
SUMMARIZER_MODEL = getattr(config, "TALK_SUMMARY_MODEL", "gpt-5-mini")
ANSWER_MODEL = getattr(config, "TALK_ANSWER_MODEL", getattr(config, "DEFAULT_MODEL", "gpt-5-mini"))

# Chat-safe default if a responses-only or image model is requested
CHAT_SAFE_DEFAULT = getattr(config, "DEFAULT_MODEL", "gpt-4o-mini")

# Regex hints for models that are not served by chat.completions
RESPONSES_ONLY_HINTS = (
    re.compile(r"^gpt-image", re.I),
    re.compile(r"^dall[- ]?e", re.I),
    re.compile(r"^whisper", re.I),
)


# ------------------------------------------------------------
# Small helper to be resilient to different secure_tex_input signatures
# ------------------------------------------------------------
def _tex_safe(s: str) -> str:
    """
    Ensure returned text is safe to embed into TeX (no macros injected).
    Works whether secure_tex_input accepts (text) or (filename, text).
    """
    try:
        return secure_tex_input(s)  # type: ignore[arg-type]
    except TypeError:
        return secure_tex_input("inline.txt", s)  # type: ignore[misc]


def _is_responses_only_model(name: str) -> bool:
    if not name:
        return False
    return any(rx.search(name) for rx in RESPONSES_ONLY_HINTS)


def _is_image_family(name: str) -> bool:
    if not name:
        return False
    return bool(re.match(r"^(gpt-image|dall[- ]?e)", name, flags=re.I))


# ============================================================
# 🧠 REQUEST MODEL (back-compat + new stable key)
# ============================================================
class TalkReq(BaseModel):
    # Primary inputs (optional if pulling from context)
    jd_text: str = ""
    question: str
    resume_tex: Optional[str] = None
    resume_plain: Optional[str] = None

    # Behavior
    tone: str = "balanced"
    humanize: bool = True
    model: str = ANSWER_MODEL

    # Context lookup (new: stable key) — fallbacks preserved
    context_key: Optional[str] = None         # NEW: "{company}__{role}"
    context_id: Optional[str] = None          # legacy "title"/"id" filename stem
    title: Optional[str] = None               # alias for legacy id
    use_latest: bool = True                   # fallback to newest if nothing provided


# ============================================================
# 🧩 CONTEXT HELPERS (stable-key aware)
# ============================================================
def _path_for_key(key: str) -> Path:
    return CONTEXT_DIR / f"{safe_filename(key)}.json"


def _latest_path() -> Optional[Path]:
    files = sorted(CONTEXT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _coerce_key_from_ctx(ctx: Dict[str, Any], fallback_path: Optional[Path]) -> str:
    if ctx.get("key"):
        return str(ctx["key"]).strip()
    c, r = (ctx.get("company") or "").strip(), (ctx.get("role") or "").strip()
    if c and r:
        return f"{safe_filename(c)}__{safe_filename(r)}"
    return fallback_path.stem if fallback_path else ""


def _pick_resume_from_ctx(ctx: Dict[str, Any]) -> str:
    """
    Prefer modern nested structure, then legacy flat.
    """
    # modern nested
    h = (ctx.get("humanized") or {})
    o = (ctx.get("optimized") or {})
    for candidate in (h.get("tex"), o.get("tex"), ctx.get("humanized_tex"), ctx.get("resume_tex")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return ""


def _pick_coverletter_from_ctx(ctx: Dict[str, Any]) -> str:
    cl = (ctx.get("cover_letter") or {})
    v = cl.get("tex")
    return v.strip() if isinstance(v, str) else ""


def _load_context(req: TalkReq) -> Tuple[Dict[str, Any], Optional[Path]]:
    """
    Load context in priority order:
      1) context_key (stable)
      2) legacy id/title
      3) latest
    """
    path: Optional[Path] = None
    if (req.context_key or "").strip():
        path = _path_for_key(req.context_key.strip())
    elif (req.context_id or req.title or "").strip():
        # legacy files can still be addressed directly by their filename stem
        stem = safe_filename((req.context_id or req.title or "").strip())
        path = CONTEXT_DIR / f"{stem}.json"
    elif req.use_latest:
        path = _latest_path()

    ctx = _read_json(path)
    if ctx:
        meta = {
            "key": _coerce_key_from_ctx(ctx, path),
            "company": ctx.get("company"),
            "role": ctx.get("role"),
            "updated_at": ctx.get("updated_at") or ctx.get("saved_at"),
            "title_for_memory": ctx.get("title_for_memory") or ctx.get("title"),
        }
        log_event("talk_context_used", meta)
    return ctx, path


# ============================================================
# 🧩 OPENAI HELPERS — smart routing (Responses vs Chat)
# ============================================================
async def _gen_text_smart(system: str, user: str, model: str) -> str:
    """
    Generate text using the appropriate OpenAI endpoint.
    - If an image/responses-only model is selected, route safely:
        • Image-family models → map to CHAT_SAFE_DEFAULT (text-only task)
        • Other responses-only models → call Responses API
    - Otherwise, use Chat Completions.
    Also retries via Responses API if we detect a "responses-only" server error.
    """
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI SDK not installed.")
    if not (getattr(config, "OPENAI_API_KEY", "") or "").strip():
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY missing in environment.")

    requested_model = (model or "").strip() or CHAT_SAFE_DEFAULT

    # If clearly an image model, map to chat-safe text model (this is a text Q&A endpoint)
    if _is_image_family(requested_model):
        mapped = CHAT_SAFE_DEFAULT
        log_event("talk_model_mapped", {"from": requested_model, "to": mapped, "reason": "image_model_not_text"})
        requested_model = mapped

    # If it's responses-only (but not an image model), prefer Responses API
    if _is_responses_only_model(requested_model):
        try:
            resp = await openai_client.responses.create(
                model=requested_model,
                input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            txt = getattr(resp, "output_text", "") or ""
            if not txt:
                # Best-effort extraction if SDK shape changes
                try:
                    out = []
                    for blk in getattr(resp, "output", []) or []:
                        if getattr(blk, "type", "") == "message":
                            for c in getattr(blk, "content", []) or []:
                                if getattr(c, "type", "") == "text":
                                    out.append(getattr(c, "text", "") or "")
                    txt = "\n".join(out).strip()
                except Exception:
                    txt = ""
            return (txt or "").strip()
        except Exception as e:
            # If Responses fails, last-resort: fall back to chat-safe default on chat API
            fallback = CHAT_SAFE_DEFAULT
            log_event("talk_responses_fail_fallback_chat", {"model": requested_model, "error": str(e), "fallback": fallback})
            r = await openai_client.chat.completions.create(
                model=fallback,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return (r.choices[0].message.content or "").strip()

    # Normal path: Chat Completions
    try:
        r = await openai_client.chat.completions.create(
            model=requested_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        # If server tells us to use Responses, retry there once
        msg = str(getattr(e, "message", None) or e)
        if "only supported in v1/responses" in msg.lower():
            log_event("talk_retry_responses_api", {"model": requested_model})
            try:
                resp = await openai_client.responses.create(
                    model=requested_model,
                    input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                txt = getattr(resp, "output_text", "") or ""
                if not txt:
                    try:
                        out = []
                        for blk in getattr(resp, "output", []) or []:
                            if getattr(blk, "type", "") == "message":
                                for c in getattr(blk, "content", []) or []:
                                    if getattr(c, "type", "") == "text":
                                        out.append(getattr(c, "text", "") or "")
                        txt = "\n".join(out).strip()
                    except Exception:
                        txt = ""
                return (txt or "").strip()
            except Exception as e2:
                # Final fallback to chat-safe default
                fallback = CHAT_SAFE_DEFAULT
                log_event("talk_responses_retry_fail_fallback_chat", {"model": requested_model, "error": str(e2), "fallback": fallback})
                r2 = await openai_client.chat.completions.create(
                    model=fallback,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                return (r2.choices[0].message.content or "").strip()

        # Unknown error — surface
        raise


# ============================================================
# 🔎 RESUME SUMMARY
# ============================================================
async def extract_resume_summary(resume_tex: Optional[str], resume_plain: Optional[str]) -> str:
    """
    Compress resume content into factual bullet points.
    Strips formatting, avoids hallucination.
    """
    if not (resume_tex or resume_plain):
        return "No resume text provided."

    text_input = (resume_plain or resume_tex or "").strip()[:3500]
    sys_prompt = (
        "Summarize this resume into 6–10 concise factual bullet points "
        "about key skills, technologies, and experiences. "
        "Do NOT fabricate or guess. Output plain-text bullets."
    )

    try:
        summary_text = await _gen_text_smart(sys_prompt, text_input, model=SUMMARIZER_MODEL)
        return _tex_safe(summary_text)  # keep LaTeX-safe if UI renders into TeX
    except Exception as e:
        log_event("talk_resume_summary_fail", {"error": str(e)})
        # Fallback: provide trimmed raw text (still TeX-safe)
        return _tex_safe(text_input[:1200])


# ============================================================
# 💬 ANSWER GENERATION — exactly two short, customized paragraphs
# ============================================================
async def generate_answer(
    jd_text: str,
    resume_summary: str,
    question: str,
    model: str,
    cover_letter: str = "",
) -> str:
    """
    Produce a two-paragraph, first-person answer (short but rich).
    Para 1 = proof from resume; Para 2 = how that maps to THIS JD/company.
    Ground ONLY in JD, resume summary, and (if present) cover letter.
    No headings, no bullets, no fluff, no claims beyond the sources.
    """
    sys_prompt = (
        "You are HIREX Assistant, an AI recruiter co-pilot. "
        "Write like a candidate who genuinely did the work and was born to do this job. "
        "STRICT FORMAT: Output EXACTLY TWO PARAGRAPHS, no headings or bullets. "
        'STYLE: First-person, confident, specific, natural; never say "as an AI"; avoid buzzwords and clichés. '
        "GROUNDING: Use ONLY facts present in the Job Description (JD), Resume Summary, and (if provided) Cover Letter. "
        "TRUTH: If a detail isn’t in those sources, don’t invent it; prefer a brief gap-statement instead. "
        "CUSTOMIZATION: Weave 3–6 concrete JD keywords naturally (stack, methods, domains). "
        "LENGTH: Keep it tight—roughly 90–150 words total across both paragraphs."
    )

    cl_block = f"\n\n[Cover Letter]\n{cover_letter[:3000]}" if cover_letter else ""
    user_prompt = (
        "You will answer an application or interview question using the provided sources.\n\n"
        "[Job Description]\n"
        f"{jd_text[:6000]}\n\n"
        "[Resume Summary]\n"
        f"{resume_summary[:3000]}{cl_block}\n\n"
        "[Question]\n"
        f"{question.strip()}\n\n"
        "[Role Context]\n"
        "Respond as the applicant described above, aligning tone and evidence to that company and position.\n\n"
        "CONTENT PLAN:\n"
        "Paragraph 1 — Proof of past impact: choose 2–3 resume-backed accomplishments most relevant to the JD; "
        "quantify outcomes (%/#/$/time), name concrete artifacts (models, datasets, systems), and mention tools/frameworks exactly as written.\n"
        "Paragraph 2 — Forward alignment: map those wins to the company’s JD and mission; mirror 3–4 explicit JD needs "
        "(model types, infrastructure, metrics, or domain), outline your contribution plan, and tie to user or business impact.\n\n"
        "HARD RULES:\n"
        "- Exactly two paragraphs (no more, no less).\n"
        "- 90–150 words total.\n"
        "- No bullet points, lists, or headings.\n"
        "- No invented facts; only use data from the sources.\n"
        "- Include 3–5 exact JD nouns naturally.\n"
        "- Avoid copying JD sentences verbatim.\n"
        "- Use confident, natural tone — no buzzwords or fluff.\n"
        "- Past tense for proof; present/future for mapping."
    )


    start = time.time()
    answer = await _gen_text_smart(sys_prompt, user_prompt, model=model)
    latency = round(time.time() - start, 2)
    tokens = len(answer.split())
    log_event("talk_answer_raw", {"latency": latency, "tokens": tokens, "model": model})

    return _tex_safe(answer)


# ============================================================
# ✨ HUMANIZE (SuperHuman)
# ============================================================
async def humanize_text(answer_text: str, tone: str) -> Tuple[str, bool]:
    """
    Refine the tone and flow via SuperHuman rewrite API.
    Falls back gracefully if unavailable.

    Returns:
        (final_text, was_humanized)
    """
    api_base = (getattr(config, "API_BASE_URL", "") or "").rstrip("/") or "http://127.0.0.1:8000"
    url = f"{api_base}/api/superhuman/rewrite"
    payload = {
        "text": "Rewrite while preserving EXACTLY TWO PARAGRAPHS and ~90–150 words total.\n\n" + answer_text,
        "mode": "paragraph",
        "tone": tone,
        "latex_safe": True,
    }

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        rewritten = data.get("rewritten") or answer_text
        was_humanized = isinstance(rewritten, str) and rewritten.strip() and (rewritten.strip() != answer_text.strip())
        return _tex_safe(rewritten), was_humanized
    except Exception as e:
        log_event("talk_superhuman_fail", {"error": str(e)})
        return answer_text, False


# ============================================================
# 🟢 HEALTH CHECK (compact; returns both epoch and ISO)
# ============================================================
@router.get("/ping")
async def ping():
    now = datetime.now(tz=timezone.utc)
    return {"ok": True, "service": "talk", "epoch": time.time(), "iso": now.isoformat()}


# ============================================================
# 🚀 MAIN ENDPOINT
# ============================================================
@router.post("/answer")
@router.post("")  # compatibility for POST /api/talk
async def talk_to_hirex(req: TalkReq):
    """
    Generate a contextual, factual, optionally humanized answer for
    job-application or interview questions.

    Behavior:
      • If jd_text / resume not provided, pulls from the latest (or specified)
        saved context created by /api/context/save (stable key: {company}__{role}).
      • Prefer humanized→optimized→legacy resume text in that order.
      • If a cover letter exists in context, it is included for added grounding.
      • Returns both 'answer' (final) and 'draft_answer' (pre-humanize).
    """
    # Pull context if needed
    jd_text = (req.jd_text or "").strip()
    resume_tex = (req.resume_tex or "").strip()
    cover_letter_tex = ""
    used_key = ""
    used_company = ""
    used_role = ""
    used_title_for_memory = ""
    used_updated_at = ""

    # Load context if any required piece is missing
    if (not jd_text) or (not resume_tex and not (req.resume_plain or "").strip()):
        ctx, ctx_path = _load_context(req)
        if ctx:
            jd_text = jd_text or (ctx.get("jd_text") or "")
            resume_tex = resume_tex or _pick_resume_from_ctx(ctx)
            cover_letter_tex = _pick_coverletter_from_ctx(ctx)
            used_key = _coerce_key_from_ctx(ctx, ctx_path)
            used_company = (ctx.get("company") or "").strip()
            used_role = (ctx.get("role") or "").strip()
            used_title_for_memory = (ctx.get("title_for_memory") or ctx.get("title") or "").strip()
            used_updated_at = (ctx.get("updated_at") or ctx.get("saved_at") or "").strip()

    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="Job Description missing. Provide jd_text or save a context first.")
    if not (resume_tex or (req.resume_plain or "").strip()):
        raise HTTPException(status_code=400, detail="Resume text missing. Provide resume_tex/plain or save a context first.")

    # 1) Resume summary
    resume_summary = await extract_resume_summary(resume_tex, req.resume_plain)

    # 2) Raw answer generation
    model = (req.model or ANSWER_MODEL).strip() or ANSWER_MODEL
    draft_answer = await generate_answer(jd_text, resume_summary, req.question, model=model, cover_letter=cover_letter_tex)

    # 3) Optional humanization (and truthy flag only if it actually changed)
    if req.humanize:
        final_answer, was_humanized = await humanize_text(draft_answer, req.tone)
    else:
        final_answer, was_humanized = draft_answer, False

    # 4) Log metadata
    log_event(
        "talk_to_hirex",
        {
            "question": req.question,
            "tone": req.tone,
            "humanize_requested": req.humanize,
            "humanize_applied": was_humanized,
            "jd_len": len(jd_text),
            "resume_len": len(resume_tex or req.resume_plain or ""),
            "model": model,
            "context_used": bool(used_key),
            "context_key": used_key,
            "company": used_company,
            "role": used_role,
        },
    )

    # 5) Structured return (include aliases for frontend)
    return {
        "question": req.question.strip(),
        "resume_summary": resume_summary,
        "draft_answer": draft_answer,
        "final_text": final_answer,
        "answer": final_answer,                 # alias for UI compatibility
        "tone": req.tone,
        "humanized": was_humanized,             # true only if rewrite changed output
        "model": model,
        # context meta for UI panes / breadcrumbs
        "context": {
            "key": used_key,
            "company": used_company,
            "role": used_role,
            "title_for_memory": used_title_for_memory,
            "updated_at": used_updated_at,
            "has_cover_letter": bool(cover_letter_tex),
        },
    }
