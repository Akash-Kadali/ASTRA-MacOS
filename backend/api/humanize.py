"""
HIREX • api/humanize.py (v2.1.2)
Integrates with AIHumanize.io for tone-only rewriting of Experience & Project bullets.
Targets only \resumeItem{...} entries, with strong LaTeX sanitization to avoid
preamble duplication or document corruption. Concurrency + retry hardened.

Author: Sri Akash Kadali
"""

from __future__ import annotations

import os
import re
import json
import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.core import config
from backend.core.utils import log_event
from backend.core.security import secure_tex_input


# ============================================================
# ⚙️ Configuration
# ============================================================

AIHUMANIZE_REWRITE_URL = "https://aihumanize.io/api/v1/rewrite"

# Prefer config-provided mappings/defaults (from core/config.py)
_AIH_MODE_ID: Dict[str, str] = getattr(
    config, "AIHUMANIZE_MODE_ID", {"quality": "0", "balance": "1", "enhanced": "2"}
)
_HUMANIZE_MODE_DEFAULT: str = str(getattr(config, "HUMANIZE_MODE_DEFAULT", "balance")).lower()
_HUMANIZE_DEFAULT_ON: bool = bool(getattr(config, "HUMANIZE_DEFAULT_ON", True))
_LOCAL_ENABLED: bool = bool(getattr(config, "SUPERHUMAN_LOCAL_ENABLED", False))

# Client limits & retries
MAX_CONCURRENT = int(os.getenv("AIHUMANIZE_MAX_CONCURRENT", "60"))
TIMEOUT_SEC = float(os.getenv("AIHUMANIZE_TIMEOUT_SEC", "2000.0"))
RETRIES = int(os.getenv("AIHUMANIZE_RETRIES", "2"))

# FastAPI router (optional for direct API usage)
router = APIRouter(prefix="/api/humanize", tags=["humanize"])


# ============================================================
# 🧽 LaTeX Sanitizer
# ============================================================

_BAD_PREAMBLE_PATTERNS = [
    r"(?i)\\documentclass(\[[^\]]*\])?\{[^}]*\}",
    r"(?i)\\usepackage(\[[^\]]*\])?\{[^}]*\}",
    r"(?i)\\begin\{document\}",
    r"(?i)\\end\{document\}",
    r"(?i)\\(new|renew)command\*?\{[^}]*\}\{[^}]*\}",
    r"(?i)\\input\{[^}]*\}",
]
_FALLBACK_TAG_RE = re.compile(r"^\[LOCAL-FALLBACK:[^\]]+\]\s*", re.IGNORECASE)

def _escape_unescaped_percent(s: str) -> str:
    # Turn bare % into \% to avoid commenting out the remainder of the line
    return re.sub(r"(?<!\\)%", r"\\%", s)

def _strip_md_fences(s: str) -> str:
    return s.replace("```latex", "").replace("```", "")

def clean_humanized_text(text: str, *, latex_safe: bool = True) -> str:
    """
    Remove dangerous LaTeX preamble/commands and markdown fences.
    Strip any accidental fallback labels. Keep content intact.
    """
    cleaned = text or ""
    cleaned = _strip_md_fences(cleaned)
    cleaned = _FALLBACK_TAG_RE.sub("", cleaned)

    for pat in _BAD_PREAMBLE_PATTERNS:
        cleaned = re.sub(pat, "", cleaned)

    # Remove leading LaTeX comments or decorative headers commonly injected
    cleaned = re.sub(r"(?m)^\s*%.*$", "", cleaned)

    # Normalize whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # Escape stray %
    if latex_safe:
        cleaned = _escape_unescaped_percent(cleaned)
        cleaned = secure_tex_input(cleaned)

    # Final safety check: if we still see preamble markers, reject
    if re.search(r"\\documentclass|\\usepackage|\\begin\{document\}|\\end\{document\}", cleaned, re.I):
        log_event("humanize_sanitizer_reject", {"reason": "preamble_detected"})
        return ""

    return cleaned


# ============================================================
# 🔎 Bullet Extraction (brace-aware)
# ============================================================

@dataclass
class BulletSpan:
    start: int
    end: int
    content: str

def _find_resume_items(tex: str) -> List[BulletSpan]:
    """
    Find \resumeItem{...} ranges with a simple brace-depth scan,
    so nested braces within the bullet are tolerated.
    """
    key = r"\resumeItem{"
    spans: List[BulletSpan] = []
    i = 0
    n = len(tex)
    while i < n:
        j = tex.find(key, i)
        if j == -1:
            break
        k = j + len(key)  # content starts here
        depth = 1
        p = k
        while p < n and depth > 0:
            ch = tex[p]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            p += 1
        if depth == 0:
            content = tex[k : p - 1]
            spans.append(BulletSpan(start=k, end=p - 1, content=content))
            i = p
        else:
            # Unbalanced; bail
            break
    return spans


# ============================================================
# 🌐 AIHumanize Client
# ============================================================

def _resolve_mode_id(mode: str) -> str:
    m = (mode or "").lower().strip()
    if m in _AIH_MODE_ID:
        return _AIH_MODE_ID[m]
    # accept tone synonyms
    if m in {"formal", "academic", "quality"}:
        return _AIH_MODE_ID.get("quality", "0")
    if m in {"balanced", "confident", "balance"}:
        return _AIH_MODE_ID.get("balance", "1")
    if m in {"conversational", "enhanced"}:
        return _AIH_MODE_ID.get("enhanced", "2")
    return _AIH_MODE_ID.get(_HUMANIZE_MODE_DEFAULT, "1")

def _header_variants(key: str) -> List[Dict[str, str]]:
    base = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
    }
    key = (key or "").strip()
    return [
        dict(base, **{"Authorization": key}),
        dict(base, **{"Authorization": f"Bearer {key}"}),
        dict(base, **{"X-API-KEY": key}),
    ]

async def _rewrite_bullet(
    client: httpx.AsyncClient,
    bullet_text: str,
    idx: int,
    mode_id: str,
    mail: str,
) -> str:
    """
    Call AIHumanize for a single bullet with retry + sanitize.
    Preserves metrics and percent signs by design (post-sanitizer escapes).
    """
    payload = {"model": mode_id, "mail": mail, "data": bullet_text}

    for attempt in range(RETRIES + 1):
        for headers in _header_variants(config.HUMANIZE_API_KEY):
            try:
                r = await client.post(AIHUMANIZE_REWRITE_URL, headers=headers, json=payload)
                # try JSON parse; if invalid JSON, surface HTTP error
                try:
                    data = r.json()
                except json.JSONDecodeError:
                    r.raise_for_status()
                    raise HTTPException(status_code=502, detail="Humanize returned invalid JSON.")

                if isinstance(data, dict) and int(data.get("code", r.status_code)) == 200 and data.get("data"):
                    candidate = clean_humanized_text(str(data["data"]).strip(), latex_safe=True)
                    if candidate:
                        # Resume bullets should be one line; avoid trailing period
                        candidate = candidate.replace("\n", " ").strip().rstrip(".")
                        log_event("aihumanize_bullet_ok", {"idx": idx, "len": len(candidate), "attempt": attempt})
                        return candidate
                    else:
                        # unsafe → revert to original
                        log_event("aihumanize_bullet_revert_unsafe", {"idx": idx, "attempt": attempt})
                        return bullet_text
                else:
                    # Unexpected shape or non-200 code; try next header/attempt
                    log_event("aihumanize_bad_response", {"idx": idx, "attempt": attempt, "resp": data})
            except Exception as e:
                log_event("aihumanize_bullet_error", {"idx": idx, "attempt": attempt, "error": str(e)})
        # exponential backoff between attempts
        await asyncio.sleep(0.5 * (2 ** attempt))

    log_event("aihumanize_bullet_fallback", {"idx": idx})
    return bullet_text


# ============================================================
# 🧩 Local (optional) tone-only stub — tagless and minimal
# ============================================================

def _local_tone_only(text: str) -> str:
    """
    Minimal, safe, tagless cleanup used only if SUPERHUMAN_LOCAL_ENABLED is True
    and remote Humanize is unavailable. Keeps numbers/metrics intact.
    """
    t = text or ""
    # normalize whitespace
    t = re.sub(r"[ \t]+", " ", t).strip()
    # tiny clarity nips (do not touch numbers, symbols)
    t = re.sub(r"\bu\b", "you", t, flags=re.IGNORECASE)
    t = re.sub(r"\bim\b", "I am", t, flags=re.IGNORECASE)
    # single line, no trailing period for bullets
    t = t.replace("\n", " ").rstrip(".")
    return clean_humanized_text(t, latex_safe=True)


# ============================================================
# 🧠 Public Core: Humanize all \resumeItem bullets
# ============================================================

async def humanize_resume_items(
    tex_content: str,
    mode: str = None,
    email: Optional[str] = None,
) -> Tuple[str, int, int]:
    """
    Humanize all \resumeItem{...} bullets concurrently.

    Returns:
        (new_tex, total_found, total_rewritten)
    """
    # Honor "Humanize always on" default; fail fast if disabled
    if not _HUMANIZE_DEFAULT_ON:
        raise RuntimeError("Humanize is disabled by configuration (HUMANIZE_DEFAULT_ON=false).")

    spans = _find_resume_items(tex_content or "")
    total_found = len(spans)
    if total_found == 0:
        log_event("aihumanize_no_bullets", {})
        return tex_content, 0, 0

    # Credentials
    has_creds = bool(config.HUMANIZE_API_KEY and config.HUMANIZE_MAIL)
    use_local = (not has_creds) and _LOCAL_ENABLED

    # Resolve mode and mail
    mode_id = _resolve_mode_id(mode or _HUMANIZE_MODE_DEFAULT)
    mail = (email or config.HUMANIZE_MAIL or "").strip()

    limits = httpx.Limits(max_keepalive_connections=MAX_CONCURRENT, max_connections=MAX_CONCURRENT)
    timeout = httpx.Timeout(TIMEOUT_SEC)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    rewritten_texts: List[str] = []

    if has_creds:
        async with httpx.AsyncClient(limits=limits, timeout=timeout, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}) as client:
            async def _task(idx: int, content: str) -> str:
                async with sem:
                    c = content.strip()
                    if not c:
                        return content
                    return await _rewrite_bullet(client, c, idx, mode_id, mail)

            rewritten_texts = await asyncio.gather(
                *[_task(i + 1, b.content) for i, b in enumerate(spans)], return_exceptions=False
            )
    elif use_local:
        # Local tagless cleanup per bullet
        rewritten_texts = [_local_tone_only(b.content) for b in spans]
    else:
        raise RuntimeError("HUMANIZE_API_KEY/HUMANIZE_MAIL missing and local fallback disabled.")

    # Rebuild the LaTeX safely by slicing with recorded spans
    out_parts: List[str] = []
    last = 0
    total_rewritten = 0
    for (span, new_txt) in zip(spans, rewritten_texts):
        out_parts.append(tex_content[last:span.start])
        safe_new = (new_txt or "").strip().rstrip(".")
        if safe_new != span.content.strip():
            total_rewritten += 1
        out_parts.append(safe_new)
        last = span.end
    out_parts.append(tex_content[last:])

    new_tex = "".join(out_parts)

    # Final safety: strip accidental preamble fragments and normalize whitespace
    for pat in _BAD_PREAMBLE_PATTERNS:
        new_tex = re.sub(pat, "", new_tex)
    new_tex = re.sub(r"\n{3,}", "\n\n", new_tex).strip()

    log_event("aihumanize_complete", {"found": total_found, "rewritten": total_rewritten, "mode": mode or _HUMANIZE_MODE_DEFAULT})
    return new_tex, total_found, total_rewritten


# ============================================================
# 🌐 FastAPI endpoints (optional, convenient for frontend)
# ============================================================

class BulletsReq(BaseModel):
    tex_content: str = Field(..., description="LaTeX content containing \\resumeItem{...} bullets.")
    mode: Optional[str] = Field(None, description="quality | balance | enhanced | synonyms accepted")
    email: Optional[str] = Field(None, description="Account email for AIHumanize (optional override).")

@router.post("/bullets")
async def api_humanize_bullets(req: BulletsReq):
    """
    Rewrites only \\resumeItem{...} bullets inside the provided LaTeX string.
    Returns sanitized LaTeX. Requires HUMANIZE_API_KEY/HUMANIZE_MAIL or enabled local fallback.
    """
    try:
        new_tex, found, rewritten = await humanize_resume_items(req.tex_content, mode=req.mode, email=req.email)
        return {
            "ok": True,
            "tex_content": new_tex,
            "found": found,
            "rewritten": rewritten,
            "mode": (req.mode or _HUMANIZE_MODE_DEFAULT),
        }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AIHumanize processing failed: {e}")


# ============================================================
# 🧪 Local CLI test
# ============================================================

if __name__ == "__main__":
    async def _run():
        sample_tex = r"""
        \resumeItem{worked on python scripts for data processing}
        \resumeItem{helped team with docker deployments}
        \resumeItem{deployed 3 APIs with 99\% uptime}
        """
        try:
            out, found, rewritten = await humanize_resume_items(sample_tex, mode="balance")
            print("\n=== Found:", found, "Rewritten:", rewritten, "===\n")
            print(out)
        except Exception as e:
            print("Local test error:", e)

    asyncio.run(_run())
