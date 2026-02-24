"""
============================================================
 HIREX v1.0.0 — superhuman.py
 ------------------------------------------------------------
 SuperHuman Humanizer API
  • Rewrites text for clarity, flow, and tone
  • Preserves factual integrity and metrics
  • Supports resume, coverletter, paragraph, sentence modes
  • LaTeX-safe output for integration with HIREX optimizer
  • Powered by AI Humanize API (https://aihumanize.io)
  • Local fallback available (tagless) if explicitly enabled

 Author: Sri Akash Kadali
============================================================
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import List, Union, Dict, Any, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.core import config
from backend.core.utils import log_event
from backend.core.security import secure_tex_input


# ============================================================
# 🔧 Setup
# ============================================================

router = APIRouter(prefix="/api/superhuman", tags=["superhuman"])

# Vendor endpoints used by HIREX.
HUMANIZE_REWRITE_URL = "https://aihumanize.io/api/v1/rewrite"
HUMANIZE_BALANCE_URL = "https://aihumanize.io/api/v1/surplus"

# Credentials (see core/config.py)
HUMANIZE_API_KEY: str = getattr(config, "HUMANIZE_API_KEY", "") or ""
HUMANIZE_MAIL: str = getattr(config, "HUMANIZE_MAIL", "") or ""

# Behavior toggles and limits
_MAX_ITEMS = 25
_TIMEOUT_S = 120
_LOCAL_ENABLED = bool(getattr(config, "SUPERHUMAN_LOCAL_ENABLED", False))  # default off per new requirements
_CONCURRENCY = 6  # limit concurrent outbound calls

# Humanize always-on defaults (from config; can be overridden via .env)
_HUMANIZE_DEFAULT_ON = getattr(config, "HUMANIZE_DEFAULT_ON", True)
_HUMANIZE_MODE_DEFAULT = str(getattr(config, "HUMANIZE_MODE_DEFAULT", "balance")).lower()
_AIH_MODE_ID = getattr(config, "AIHUMANIZE_MODE_ID", {"quality": "0", "balance": "1", "enhanced": "2"})


# ============================================================
# 🧠 Request Model
# ============================================================

from pydantic import BaseModel, Field
from typing import List, Union, Optional

class RewriteRequest(BaseModel):
    text: Union[str, List[str]] = Field(..., description="Text or list of texts to rewrite.")
    mode: str = Field(default="resume")
    tone: str = Field(default="balanced")
    latex_safe: bool = Field(default=True)
    constraints: Dict[str, Any] = Field(
        default_factory=lambda: {"no_fabrication": True, "keep_metrics": True},
        description="Behavior guards such as no_fabrication, keep_metrics",
    )
    max_len: int = Field(1600, description="Max input chars per item (truncate beyond).")


# ============================================================
# 🧩 Tone/Mode → AIHumanize model code
#   0: quality   1: balance   2: enhanced
# ============================================================

def _resolve_model_code(tone: str) -> int:
    t = (tone or "").lower().strip()
    if t in {"default", ""}:
        t = _HUMANIZE_MODE_DEFAULT
    if t in {"formal", "academic", "quality"}:
        return 0
    if t in {"balanced", "confident", "balance"}:
        return 1
    if t in {"conversational", "enhanced"}:
        return 2
    # Fallback to configured default mapping
    try:
        return int(_AIH_MODE_ID.get(_HUMANIZE_MODE_DEFAULT, "1"))
    except Exception:
        return 1


# ============================================================
# 🧼 Post-processing
# ============================================================

_FALLBACK_TAG_RE = re.compile(r"^\[LOCAL-FALLBACK:[^\]]+\]\s*", re.IGNORECASE)

def _clean_text(text: str, latex_safe: bool, mode: str) -> str:
    """Normalize whitespace, strip any fallback labels, and optionally LaTeX-sanitize."""
    t = (text or "").replace("\r", "")
    t = _FALLBACK_TAG_RE.sub("", t)  # never leak [LOCAL-FALLBACK:*]
    t = re.sub(r"[ \t\f\v]+", " ", t).strip()

    if mode == "resume":
        # Resume bullets should be single-line and punchy; trim trailing period
        t = t.replace("\n", " ").rstrip(" .")

    if latex_safe:
        t = secure_tex_input(t)

    return t


# ============================================================
# 🧩 Local fallback (offline mode) — TAGLESS by design
# ============================================================

def _local_rewrite_stub(text: str, tone: str, mode: str) -> str:
    """
    Offline fallback when no API connectivity or creds are invalid.
    Keep light-touch edits and NEVER prefix any label (tagless).
    """
    t = (text or "")
    # basic normalization + a couple of clarity nips
    t = re.sub(r"\bu\b", "you", t, flags=re.IGNORECASE)
    t = re.sub(r"\br\b", "are", t, flags=re.IGNORECASE)
    t = re.sub(r"\bim\b", "I am", t, flags=re.IGNORECASE)
    if (tone or "").lower().strip() in {"formal", "academic"}:
        t = (t.replace("I'm", "I am")
               .replace("don't", "do not")
               .replace("can't", "cannot"))
    return t.strip()


# ============================================================
# 🔐 Creds / Headers helpers
# ============================================================

def _require_api_creds() -> None:
    """
    Ensure Humanize is enabled by default and credentials are present.
    If fallback is enabled explicitly, we don't raise here (but still prefer API).
    """
    if not _HUMANIZE_DEFAULT_ON:
        raise HTTPException(status_code=503, detail="Humanize disabled by configuration.")
    if not HUMANIZE_API_KEY or not HUMANIZE_MAIL:
        if _LOCAL_ENABLED:
            log_event("⚠️ superhuman_local_fallback", {"reason": "missing_credentials"})
            return
        raise HTTPException(
            status_code=503,
            detail="Humanize API credentials missing. "
                   "Set HUMANIZE_API_KEY and HUMANIZE_MAIL in your .env.",
        )


def _header_variants(key: str) -> List[Dict[str, str]]:
    """
    Build header variants to maximize compatibility and reduce CF challenges.
    """
    key = (key or "").strip()
    base = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
    }
    return [
        dict(base, **{"Authorization": key}),
        dict(base, **{"Authorization": f"Bearer {key}"}),
        dict(base, **{"X-API-KEY": key}),
    ]


def _payload_with_aliases(model_code: int, mail: str, text: str) -> Dict[str, Any]:
    """
    Include common field aliases used by different Humanize integrations.
    Canonical fields:
      - model: 0|1|2
      - mail: registered email
      - data: text to rewrite
    Aliases:
      - email: same as mail
      - text: same as data
      - token: some gateways expect token in body (rare)
    """
    return {
        "model": model_code,
        "mail": mail,
        "email": mail,
        "data": text,
        "text": text,
        "token": HUMANIZE_API_KEY,  # harmless if ignored
    }


# ============================================================
# ⚙️ Core Rewrite
# ============================================================

async def _call_humanize(text: str, tone: str, mode: str) -> str:
    """
    Low-level caller that tries multiple header styles automatically.
    Raises HTTPException on hard failures.
    """
    model_code = _resolve_model_code(tone)
    payload = _payload_with_aliases(model_code, HUMANIZE_MAIL, text)

    last_err: Optional[str] = None
    start_time = time.time()

    async with httpx.AsyncClient(timeout=_TIMEOUT_S, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}) as client:
        for hdr in _header_variants(HUMANIZE_API_KEY):
            try:
                r = await client.post(HUMANIZE_REWRITE_URL, headers=hdr, json=payload)
                # Try to parse JSON; if not JSON, surface an HTTP error
                try:
                    data = r.json()
                except json.JSONDecodeError:
                    r.raise_for_status()
                    raise HTTPException(status_code=502, detail="Humanize API returned invalid JSON.")

                code = int(data.get("code", r.status_code))
                msg = str(data.get("msg") or "")
                if code != 200:
                    # Common: 1003 Invalid API Key
                    last_err = f"Humanize error ({code}): {msg or 'unknown'}; hdr={list(hdr.keys())[0]}"
                    # try next header variant
                    continue

                rewritten_raw = data.get("data") or ""
                if not isinstance(rewritten_raw, str):
                    rewritten_raw = str(rewritten_raw)

                latency = round(time.time() - start_time, 2)
                log_event(
                    "superhuman_rewrite_http_success",
                    {"mode": mode, "tone": tone, "chars": len(text), "latency_s": latency, "hdr": list(hdr.keys())[0]},
                )
                return rewritten_raw

            except Exception as e:
                last_err = f"{type(e).__name__}: {e}; hdr={list(hdr.keys())[0]}"
                # try next header variant

    # If we got here, all header variants failed
    raise HTTPException(status_code=502, detail=last_err or "Humanize call failed.")


async def rewrite_single(
    text: str,
    mode: str,
    tone: str,
    constraints: dict,  # reserved for future guarding/filtering
    latex_safe: bool,
    max_len: int,
) -> str:
    """Rewrite a single block of text using the AIHumanize API or (optional) fallback."""
    t_in = (text or "").strip()[: max_len]
    if not t_in:
        return ""

    # Prefer API if creds present; otherwise use optional fallback
    if HUMANIZE_API_KEY and HUMANIZE_MAIL:
        try:
            rewritten_raw = await _call_humanize(t_in, tone, mode)
            return _clean_text(rewritten_raw, latex_safe, mode)
        except HTTPException as e:
            log_event("⚠️ superhuman_rewrite_error", {"error": str(e)})
            if _LOCAL_ENABLED:
                log_event("⚙️ using_local_rewrite_fallback", {"text_len": len(t_in)})
                return _clean_text(_local_rewrite_stub(t_in, tone, mode), latex_safe, mode)
            raise
    else:
        if _LOCAL_ENABLED:
            log_event("⚙️ using_local_rewrite_fallback", {"reason": "missing_credentials", "text_len": len(t_in)})
            return _clean_text(_local_rewrite_stub(t_in, tone, mode), latex_safe, mode)
        raise HTTPException(status_code=503, detail="Humanize credentials missing and local fallback disabled.")


# ============================================================
# 🚀 Main Endpoint
# ============================================================

@router.post("/rewrite")
async def rewrite_text(req: RewriteRequest):
    """
    SuperHuman rewrite engine — transforms one or more text inputs via AIHumanize API.
    Fails fast if Humanize is disabled or credentials are missing (unless explicit local fallback is enabled).
    """
    _require_api_creds()

    if not req.text:
        raise HTTPException(status_code=400, detail="No text provided.")

    items = req.text if isinstance(req.text, list) else [req.text]
    if len(items) > _MAX_ITEMS:
        raise HTTPException(status_code=413, detail=f"Too many items (max {_MAX_ITEMS}).")

    sem = asyncio.Semaphore(_CONCURRENCY)

    async def _bounded_rewrite(t: str):
        async with sem:
            return await rewrite_single(t, req.mode, req.tone, req.constraints, req.latex_safe, req.max_len)

    results = await asyncio.gather(*[_bounded_rewrite(t) for t in items], return_exceptions=True)

    # Propagate first error if any
    for r in results:
        if isinstance(r, Exception):
            raise r

    log_event(
        "superhuman_batch_complete",
        {
            "count": len(items),
            "mode": req.mode,
            "tone": req.tone,
            "latex_safe": req.latex_safe,
            "creds_present": bool(HUMANIZE_API_KEY and HUMANIZE_MAIL),
            "local_enabled": _LOCAL_ENABLED,
        },
    )

    return {"rewritten": results if isinstance(req.text, list) else results[0]}


# ============================================================
# 💰 Balance Check Endpoint
# ============================================================

@router.get("/balance")
async def check_balance():
    """Return remaining words balance from AIHumanize API (or local indicator)."""
    if not (HUMANIZE_API_KEY and HUMANIZE_MAIL):
        if _LOCAL_ENABLED:
            return {"remaining_words": "∞ (local mode)"}
        raise HTTPException(status_code=503, detail="Humanize API credentials missing.")

    payload = {"mail": HUMANIZE_MAIL, "email": HUMANIZE_MAIL, "token": HUMANIZE_API_KEY}

    async with httpx.AsyncClient(timeout=_TIMEOUT_S, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}) as client:
        last_err: Optional[str] = None
        for hdr in _header_variants(HUMANIZE_API_KEY):
            try:
                r = await client.post(HUMANIZE_BALANCE_URL, headers=hdr, json=payload)
                try:
                    data = r.json()
                except json.JSONDecodeError:
                    r.raise_for_status()
                    raise HTTPException(status_code=502, detail="Humanize balance returned invalid JSON.")

                code = int(data.get("code", r.status_code))
                if code != 200:
                    last_err = f"Humanize balance error ({code}): {data.get('msg') or 'unknown'}; hdr={list(hdr.keys())[0]}"
                    continue

                remaining = data.get("data") or 0
                try:
                    remaining_int = int(remaining)
                except Exception:
                    remaining_int = remaining
                return {"remaining_words": remaining_int}
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}; hdr={list(hdr.keys())[0]}"

    log_event("⚠️ superhuman_balance_error", {"error": last_err or "unknown"})
    if _LOCAL_ENABLED:
        return {"remaining_words": "unknown (local fallback)"}
    raise HTTPException(status_code=502, detail=last_err or "Balance check failed.")


# ============================================================
# 🔎 Health / Debug
# ============================================================

@router.get("/health")
async def health():
    """Simple health check for the superhuman service."""
    return {
        "ok": True,
        "default_on": _HUMANIZE_DEFAULT_ON,
        "local_fallback": _LOCAL_ENABLED,
        "has_api_key": bool(HUMANIZE_API_KEY),
        "has_mail": bool(HUMANIZE_MAIL),
        "endpoint_rewrite": HUMANIZE_REWRITE_URL,
        "endpoint_balance": HUMANIZE_BALANCE_URL,
        "mode_default": _HUMANIZE_MODE_DEFAULT,
    }
