# ============================================================
#  ASTRA v2.1.2 — Utility & Diagnostics API (FINAL)
#  ------------------------------------------------------------
#  Endpoints:
#   • Health-ish ping / version / safe config subset
#   • Logging (frontend analytics)
#   • Text helpers (escape/unescape)
#   • Base64 encode/decode utilities
#   • Safe filename & slug helpers
#   • Recent Contexts (deduped per Company__Role)
#   • History + status dashboard support
#  Author: Sri Akash Kadali
# ============================================================

from __future__ import annotations

import base64
import json
import re
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Form, HTTPException, Query

from backend.core import config
from backend.core.utils import log_event, safe_filename
from backend.core.security import secure_tex_input

router = APIRouter(prefix="/api/utils", tags=["utils"])

CONTEXT_DIR: Path = config.get_contexts_dir()
LOG_PATH: Path = Path(config.LOG_PATH)  # use configured path only

# ============================================================
# ⚙️ 1) PING / VERSION / CONFIG
# ============================================================
@router.get("/ping")
async def ping():
    """Lightweight liveness probe (distinct from /health)."""
    return {
        "status": "ok",
        "service": "ASTRA Core API",
        "time": datetime.utcnow().isoformat() + "Z",
        "platform": platform.system(),
        "python": platform.python_version(),
    }


@router.get("/version")
async def get_version():
    """Return the current ASTRA version and model defaults."""
    return {
        "version": config.APP_VERSION,
        "default_model": getattr(config, "DEFAULT_MODEL", "gpt-4o-mini"),
        "talk_summary_model": getattr(config, "TALK_SUMMARY_MODEL", "gpt-4o-mini"),
        "talk_answer_model": getattr(
            config, "TALK_ANSWER_MODEL", getattr(config, "DEFAULT_MODEL", "gpt-4o-mini")
        ),
        "superhuman_local": getattr(config, "SUPERHUMAN_LOCAL_ENABLED", True),
        "build_time": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/config")
async def get_config():
    """Expose a safe subset of configuration variables for frontend diagnostics."""
    safe_keys = [
        "APP_VERSION",
        "DEFAULT_MODEL",
        "TALK_SUMMARY_MODEL",
        "TALK_ANSWER_MODEL",
        "SUPERHUMAN_LOCAL_ENABLED",
        "BASE_COVERLETTER_PATH",
        "MASTERMINDS_PATH",
        "LOG_PATH",
        "HISTORY_PATH",
        "API_BASE_URL",
    ]
    safe_data: Dict[str, Any] = {}
    for k in safe_keys:
        v = getattr(config, k, None)
        safe_data[k] = str(v) if isinstance(v, Path) else v
    return {"config": safe_data}

# ============================================================
# 🧾 2) FRONTEND LOGGING & ANALYTICS
# ============================================================
@router.post("/log")
async def log_frontend_event(
    msg: str = Form(...),
    page: str = Form("unknown"),
    version: str = Form("unknown"),
    origin: str = Form("client"),
    level: str = Form("info"),
):
    """Receives debug or analytic events from the frontend (UI telemetry)."""
    meta = {
        "msg": msg,
        "page": page,
        "version": version,
        "origin": origin,
        "level": level,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    log_event("frontend_log", meta)
    return {"logged": True, "time": meta["timestamp"]}

# ============================================================
# 🧩 3) TEXT UTILITIES
# ============================================================
@router.post("/escape")
async def escape_latex(text: str = Form(...)):
    """Return LaTeX-safe escaped string."""
    try:
        escaped = secure_tex_input(text)
        return {"escaped": escaped}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Escape failed: {e}") from e


@router.post("/unescape")
async def unescape_latex(text: str = Form(...)):
    """Reverse minimal LaTeX escapes for readability."""
    try:
        unescaped = (
            text.replace(r"\#", "#")
            .replace(r"\%", "%")
            .replace(r"\$", "$")
            .replace(r"\&", "&")
            .replace(r"\_", "_")
            .replace(r"\{", "{")
            .replace(r"\}", "}")
        )
        return {"unescaped": unescaped}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unescape failed: {e}") from e

# ============================================================
# 📦 4) ENCODING / DECODING HELPERS
# ============================================================
@router.post("/b64encode")
async def b64encode_data(raw: str = Form(...)):
    """Base64 encode a plain string."""
    try:
        encoded = base64.b64encode(raw.encode("utf-8")).decode("utf-8")
        return {"base64": encoded, "len": len(encoded)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Encode failed: {e}") from e


@router.post("/b64decode")
async def b64decode_data(encoded: str = Form(...)):
    """Base64 decode a string."""
    try:
        decoded = base64.b64decode(encoded.encode("utf-8")).decode("utf-8", errors="ignore")
        return {"decoded": decoded, "len": len(decoded)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Decode failed: {e}") from e

# ============================================================
# 🗂️ 5) FILENAME + SANITIZATION HELPERS
# ============================================================
@router.post("/safe_filename")
async def make_safe_filename(name: str = Form(...)):
    """Return a filesystem-safe version of the given filename."""
    safe = safe_filename(name)
    return {"input": name, "safe_name": safe}


@router.post("/slugify")
async def slugify_string(name: str = Form(...)):
    """Return a lowercase slugified string safe for URLs or filenames."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return {"slug": slug}

# ============================================================
# 🧭 6) RECENT CONTEXTS (deduped per Company__Role)
# ============================================================
def _updated_at(d: Dict[str, Any], default_ts: float) -> float:
    try:
        ts_raw = d.get("updated_at") or d.get("saved_at") or ""
        if isinstance(ts_raw, str):
            ts = ts_raw.rstrip("Z")
            if ts:
                return datetime.fromisoformat(ts).timestamp()
    except Exception:
        pass
    return default_ts


def _coerce_key(d: Dict[str, Any], p: Optional[Path]) -> str:
    if d.get("key"):
        return str(d["key"]).strip()
    c, r = (d.get("company") or "").strip(), (d.get("role") or "").strip()
    if c and r:
        return f"{safe_filename(c)}__{safe_filename(r)}"
    return p.stem if p else ""


def _compact_meta(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    return {
        "key": key,
        "title": d.get("title_for_memory") or d.get("title") or f"{d.get('company','')} — {d.get('role','')}",
        "company": d.get("company"),
        "role": d.get("role"),
        "updated_at": d.get("updated_at") or d.get("saved_at"),
        "has_optimized": bool(((d.get("optimized") or {}).get("tex")) or d.get("resume_tex")),
        "has_humanized": bool(((d.get("humanized") or {}).get("tex")) or d.get("humanized_tex")),
        "has_cover_letter": bool((d.get("cover_letter") or {}).get("tex")),
    }


@router.get("/recent")
async def recent_contexts(
    limit: int = Query(50, ge=1, le=500),
    dedupe: bool = Query(True, description="Collapse multiple files to the latest per (Company__Role)"),
):
    """
    List recent JD/resume contexts saved by the app.
    Mirrors context_store's behavior so frontend pages (Talk, Dashboard)
    can render a clean, de-duplicated '📜 JD + Resume History'.
    """
    if not CONTEXT_DIR.exists():
        return {"items": []}

    entries: List[Tuple[str, Path, Dict[str, Any], float]] = []
    for p in CONTEXT_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        key = _coerce_key(data, p)
        ts = _updated_at(data, p.stat().st_mtime)
        entries.append((key, p, data, ts))

    if dedupe:
        # choose the single newest per key
        latest_by_key: Dict[str, Tuple[Path, Dict[str, Any], float]] = {}
        for key, p, d, ts in entries:
            cur = latest_by_key.get(key)
            if (cur is None) or (ts > cur[2]):
                latest_by_key[key] = (p, d, ts)
        rows = sorted(latest_by_key.items(), key=lambda kv: kv[1][2], reverse=True)[:limit]
        items = [_compact_meta(d, _coerce_key(d, p)) for (_k, (p, d, _)) in rows]
    else:
        rows2 = sorted(entries, key=lambda t: t[3], reverse=True)[:limit]
        items = [_compact_meta(d, _coerce_key(d, p)) for (_k, p, d, _ts) in rows2]

    return {"items": items}

# ============================================================
# 🧭 7) HISTORY / LOG RETRIEVAL
# ============================================================
def _read_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
        out: List[Dict[str, Any]] = []
        for line in reversed(lines):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
        return out
    except Exception:
        return []


@router.get("/history")
async def get_history(limit: int = Query(100, ge=1, le=1000)):
    """Return the most recent event logs for diagnostics or dashboard."""
    events = _read_jsonl(LOG_PATH, limit)
    return {"count": len(events), "events": events}

# ============================================================
# 🧠 8) SYSTEM STATUS SUMMARY (Mini Dashboard)
# ============================================================
@router.get("/status")
async def get_status():
    """
    Lightweight system snapshot used by the dashboard sidebar.
    Provides event totals, last log timestamp, and environment details.
    """
    total, last_event = 0, None

    if LOG_PATH.exists():
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
                total = len(lines)
                if lines:
                    try:
                        last_event = json.loads(lines[-1])
                    except Exception:
                        last_event = None
        except Exception:
            last_event = None

    return {
        "status": "ok",
        "total_events": total,
        "last_event": last_event,
        "app_version": config.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "platform": platform.system(),
    }

# ============================================================
# 🧪 9) SELF-TEST: ENCODE-DECODE ROUNDTRIP
# ============================================================
@router.post("/selftest")
async def self_test(text: str = Form(...)):
    """Perform a simple base64 encode-decode validation."""
    try:
        encoded = base64.b64encode(text.encode("utf-8")).decode("utf-8")
        decoded = base64.b64decode(encoded.encode("utf-8")).decode("utf-8")
        return {
            "input": text,
            "encoded": encoded[:50] + ("..." if len(encoded) > 50 else ""),
            "decoded_match": decoded == text,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Self-test failed: {e}") from e
