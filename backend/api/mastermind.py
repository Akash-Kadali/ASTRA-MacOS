"""
============================================================
 HIREX v2.1.0 — mastermind.py
 ------------------------------------------------------------
 MasterMind AI Assistant API
 • Context-aware reasoning and Q&A engine
 • Supports multi-turn sessions (filesystem store)
 • Integrates cleanly with SuperHuman/Talk modules
 • Persona, tone, and model controls

 Author: Sri Akash Kadali
============================================================
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from backend.core import config
from backend.core.utils import log_event
from backend.core.security import secure_tex_input

router = APIRouter(prefix="/api/mastermind", tags=["mastermind"])
openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

# ---------------------------------------------
# Defaults
# ---------------------------------------------
DEFAULT_MODEL = getattr(config, "MASTERMINDS_MODEL", getattr(config, "DEFAULT_MODEL", "gpt-4o-mini"))
STORE_DIR: Path = Path(getattr(config, "MASTERMINDS_PATH", (Path("data") / "mastermind_sessions")))
STORE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 🗂 Filesystem session store (fallback-safe)
# ============================================================

@dataclass
class Session:
    id: str
    meta: Dict[str, Any]
    messages: List[Dict[str, str]]
    created_at: str
    updated_at: str

def _session_path(session_id: str) -> Path:
    return STORE_DIR / f"{session_id}.json"

def _new_session_id() -> str:
    return f"mm_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def start_session(meta: Dict[str, Any]) -> Dict[str, Any]:
    sid = _new_session_id()
    now = datetime.utcnow().isoformat()
    sess = Session(id=sid, meta=meta or {}, messages=[], created_at=now, updated_at=now)
    _session_path(sid).write_text(json.dumps(asdict(sess), ensure_ascii=False, indent=2), encoding="utf-8")
    return asdict(sess)

def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    p = _session_path(session_id)
    return _read_json(p)

def append_message(session_id: str, msg: Dict[str, str]) -> Dict[str, Any]:
    p = _session_path(session_id)
    data = _read_json(p) or {"id": session_id, "meta": {}, "messages": [], "created_at": datetime.utcnow().isoformat()}
    data.setdefault("messages", []).append(msg)
    data["updated_at"] = datetime.utcnow().isoformat()
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data

def list_sessions() -> List[Dict[str, Any]]:
    files = sorted(STORE_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    out: List[Dict[str, Any]] = []
    for f in files:
        data = _read_json(f)
        if not data:
            continue
        out.append({
            "id": data.get("id"),
            "meta": data.get("meta", {}),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "message_count": len(data.get("messages", [])),
        })
    return out

# ============================================================
# 🧠 Utility: truncate long history by chars
# ============================================================
def _trim_messages(msgs: List[Dict[str, str]], max_chars: int = 9000) -> List[Dict[str, str]]:
    """
    Keep only the latest messages up to max_chars (on content),
    preserving order, to control context size.
    """
    joined = ""
    kept: List[Dict[str, str]] = []
    for m in reversed(msgs):
        text = str(m.get("content", "") or "")
        if len(joined) + len(text) > max_chars:
            break
        kept.insert(0, m)
        joined += text
    return kept

def _resp_text(resp) -> str:
    """Robust extraction for OpenAI Responses API output."""
    try:
        t = getattr(resp, "output_text", None)
        if t:
            return str(t).strip()
    except Exception:
        pass
    try:
        return str(resp.output[0].content[0].text).strip()
    except Exception:
        return ""


# ============================================================
# 🚀 Endpoint: Start a new MasterMind session
# ============================================================
@router.post("/start")
async def start_session_api(
    persona: str = Form("General"),
    model: str = Form(DEFAULT_MODEL),
    purpose: str = Form("interactive reasoning"),
):
    if not config.OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY missing in environment.")
    meta = {"persona": persona, "model": model, "purpose": purpose}
    data = start_session(meta)
    log_event("mastermind_session_started", {"persona": persona, "model": model})
    return JSONResponse({"session": data})


# ============================================================
# 🧩 Endpoint: Continue conversation / respond
# ============================================================
@router.post("/chat")
async def mastermind_chat(
    session_id: str = Form(...),
    prompt: str = Form(...),
    tone: str = Form("balanced"),
    model: str = Form(DEFAULT_MODEL),
    persona: str = Form("General"),
    temperature: float = Form(0.6),
    max_ctx_chars: int = Form(9000),
):
    if not config.OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY missing in environment.")

    # 1️⃣ Load or initialize session
    data = load_session(session_id)
    if not data:
        data = start_session({"persona": persona, "model": model})

    history: List[Dict[str, str]] = _trim_messages(data.get("messages", []), max_chars=int(max_ctx_chars))

    # 2️⃣ Build AI prompt context
    system_prompt = (
        f"You are MasterMind — an intelligent, concise reasoning assistant inside HIREX. "
        f"You adopt the persona of '{persona}' and respond in a {tone} tone. "
        "Answer clearly and practically, focusing on career, resumes, job search, or technical reasoning. "
        "Keep answers compact and precise. Markdown allowed. Avoid repetition."
    )

    messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": prompt}]

    # 3️⃣ Store user message immediately
    append_message(session_id, {"role": "user", "content": prompt})

    # 4️⃣ Query the model
    try:
        resp = await openai_client.responses.create(
            model=model,
            input=messages,
            temperature=float(temperature),
            max_output_tokens=800,
        )
        reply_text = _resp_text(resp) or "No response."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM request failed: {e}")

    # 5️⃣ Sanitize for LaTeX safety
    safe_reply = secure_tex_input(reply_text)

    # 6️⃣ Append assistant reply
    append_message(session_id, {"role": "assistant", "content": safe_reply})

    # 7️⃣ Log event
    log_event(
        "mastermind_chat",
        {
            "session_id": session_id,
            "persona": persona,
            "tone": tone,
            "model": model,
            "chars": len(reply_text),
        },
    )

    return JSONResponse(
        {
            "reply": safe_reply,
            "persona": persona,
            "tone": tone,
            "model": model,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


# ============================================================
# 📜 Endpoint: Retrieve session history
# ============================================================
@router.get("/history")
async def get_session_history(session_id: str = Query(..., description="MasterMind session id")):
    """Fetch conversation messages for a given session."""
    data = load_session(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": data}


# ============================================================
# 🗂️ Endpoint: List all sessions
# ============================================================
@router.get("/sessions")
async def list_sessions_api():
    """List all MasterMind sessions with metadata."""
    return {"sessions": list_sessions()}


# ============================================================
# 🧹 Endpoint: Reset / delete a session
# ============================================================
@router.delete("/session")
async def delete_session_api(session_id: str = Query(..., description="MasterMind session id")):
    """Delete a specific session JSON file."""
    p = _session_path(session_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        p.unlink()
        log_event("mastermind_session_deleted", {"id": session_id})
        return {"deleted": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")
