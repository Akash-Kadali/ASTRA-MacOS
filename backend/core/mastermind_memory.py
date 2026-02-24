# ============================================================
#  HIREX v1.0.0 — backend/data/mastermind_sessions.py
#  ------------------------------------------------------------
#  Lightweight filesystem store for MasterMind chat sessions.
#  Compatible with /api/mastermind endpoints.
#  • Robust path resolution via backend.core.config
#  • UTF-8 safe read/write with graceful fallbacks
#  • Returns both legacy (created) and modern (created_at/updated_at) fields
# ============================================================

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Prefer configured directory; fall back to repo-friendly defaults
try:
    from backend.core import config
    _DEFAULT_DIR = Path(
        getattr(config, "MASTERMINDS_PATH", getattr(config, "MASTERMINDS_DIR", Path("backend") / "data" / "mastermind_sessions"))
    )
except Exception:
    _DEFAULT_DIR = Path("backend/data/mastermind_sessions")

DATA_DIR: Path = _DEFAULT_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _session_path(session_id: str) -> Path:
    return DATA_DIR / f"{session_id}.json"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _new_session_id() -> str:
    return f"mm_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"


def start_session(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new session file with provided metadata.
    Returns full session dict.
    """
    sid = _new_session_id()
    now = _now_iso()
    data: Dict[str, Any] = {
        "id": sid,
        "created": now,            # legacy
        "created_at": now,         # modern
        "updated_at": now,
        "meta": meta or {},
        "messages": [],
    }
    _write_json(_session_path(sid), data)
    return data


def load_session(session_id: str) -> Dict[str, Any]:
    """
    Load a session by id. Returns {} if not found or unreadable.
    """
    p = _session_path(session_id)
    data = _read_json(p)
    return data or {}


def append_message(session_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Append a single message dict to a session (creates file if missing).
    Returns the updated session dict.
    """
    p = _session_path(session_id)
    data = _read_json(p) or {
        "id": session_id,
        "created": _now_iso(),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "meta": {},
        "messages": [],
    }
    data.setdefault("messages", []).append(message)
    data["updated_at"] = _now_iso()
    _write_json(p, data)
    return data


def list_sessions() -> List[Dict[str, Any]]:
    """
    Return a list of session metadata for dashboards.
    Sorted by updated_at (desc), falling back to created/created_at.
    """
    items: List[Dict[str, Any]] = []
    for file in DATA_DIR.glob("*.json"):
        try:
            data = _read_json(file)
            if not data:
                continue
            created = data.get("created_at") or data.get("created")
            updated = data.get("updated_at") or created
            meta = data.get("meta", {}) or {}
            items.append(
                {
                    "id": data.get("id"),
                    "created": created,
                    "created_at": created,
                    "updated_at": updated,
                    "persona": meta.get("persona", "General"),
                    "message_count": len(data.get("messages", []) or []),
                }
            )
        except Exception:
            continue

    def _key(d: Dict[str, Any]) -> str:
        return (d.get("updated_at") or d.get("created_at") or "")

    return sorted(items, key=_key, reverse=True)
