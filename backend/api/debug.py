"""
============================================================
 HIREX v2.1.0 — api/debug.py
 ------------------------------------------------------------
 Lightweight diagnostic endpoint for frontend → backend logs.

  • Accepts any POSTed JSON payload (dict or list) or raw text
  • Prints to console in readable, truncated format
  • Persists structured event via log_event()
  • Auto-tags origin, page, level, and timestamps
  • Never crashes on malformed or non-JSON payloads

 Author: Sri Akash Kadali
============================================================
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from backend.core.utils import log_event


# IMPORTANT:
#  - NO `/api` here
#  - `/api` is owned by main.py
router = APIRouter(prefix="/api/debug", tags=["debug"])


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _truncate(obj: Any, limit: int = 800) -> str:
    """Truncate a JSON-serialized preview to avoid spammy console logs."""
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = str(obj)
    return (s[:limit] + "…") if len(s) > limit else s


@router.get("/ping")
async def debug_ping():
    """Simple liveness check for the debug router."""
    return {"ok": True, "router": "debug", "time": _now_iso()}


# ============================================================
# 🧠 Frontend → Backend Debug / Analytics Logger
# ============================================================
@router.post("/log")
async def debug_log(request: Request):
    """
    Receives arbitrary frontend debug or analytics payloads
    and logs them both to console and persistent JSONL.
    """

    # 1️⃣ Parse body (JSON or raw text)
    payload: Dict[str, Any]
    try:
        body = await request.json()
        if isinstance(body, dict):
            payload = body
        else:
            payload = {"data": body, "non_dict_json": True}
    except Exception:
        raw = (await request.body()).decode("utf-8", "ignore")
        payload = {"raw": raw, "format_error": True}

    # 2️⃣ Inject metadata
    headers = request.headers
    client_ip = request.client.host if request.client else "unknown"

    payload.setdefault("received_at", _now_iso())
    payload.setdefault("origin", client_ip)
    payload.setdefault("page", payload.get("page", "unknown"))
    payload.setdefault("level", payload.get("level", "debug"))
    payload.setdefault("user_agent", headers.get("user-agent", ""))
    payload.setdefault("referer", headers.get("referer", ""))
    payload.setdefault("timestamp", payload.get("received_at"))

    # 3️⃣ Console output (safe + truncated)
    msg = payload.get("msg", "(no message)")
    page = payload.get("page", "?")
    print(f"[FE DEBUG] ({page}) {msg}")
    print("  └─", _truncate({k: v for k, v in payload.items() if k != "raw"}))

    # 4️⃣ Persist event
    try:
        log_event("frontend_debug", payload)
    except Exception as e:
        print(f"[WARN] Failed to persist debug event: {e}")

    # 5️⃣ Response
    return JSONResponse(
        {
            "ok": True,
            "logged": True,
            "timestamp": payload["received_at"],
        }
    )