"""Transactional operational state for ASTRA's agent and recovery workflows."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.core import config

DB_PATH = Path(config.DATA_DIR) / "astra.sqlite3"
SCHEMA_VERSION = 2
_LOCK = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def initialize() -> None:
    with _LOCK, _connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                step TEXT NOT NULL,
                url TEXT NOT NULL DEFAULT '',
                resume_path TEXT NOT NULL DEFAULT '',
                completed_fields TEXT NOT NULL DEFAULT '[]',
                unanswered_fields TEXT NOT NULL DEFAULT '[]',
                answers TEXT NOT NULL DEFAULT '{}',
                evidence_path TEXT NOT NULL DEFAULT '',
                message TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                job_id TEXT NOT NULL DEFAULT '',
                message TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS json_snapshots (
                name TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                source_mtime_ns INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS records (
                kind TEXT NOT NULL,
                record_id TEXT NOT NULL,
                company TEXT NOT NULL DEFAULT '',
                role TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(kind, record_id)
            );
            CREATE INDEX IF NOT EXISTS idx_activity_created ON activity(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_activity_job ON activity(job_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_records_company_role ON records(kind, company, role);
            CREATE INDEX IF NOT EXISTS idx_records_status ON records(kind, status, updated_at DESC);
            """
        )
        current = db.execute("SELECT MAX(version) AS version FROM schema_meta").fetchone()["version"]
        if not current:
            db.execute("INSERT INTO schema_meta(version, applied_at) VALUES (?, ?)", (SCHEMA_VERSION, _now()))
        elif current < SCHEMA_VERSION:
            db.execute("INSERT INTO schema_meta(version, applied_at) VALUES (?, ?)", (SCHEMA_VERSION, _now()))


def sync_records(kind: str, records: list[dict[str, Any]]) -> int:
    """Transactionally replace one indexed entity collection."""
    initialize()
    rows = []
    for item in records:
        record_id = str(item.get("id") or item.get("external_id") or "").strip()
        if not record_id:
            continue
        rows.append((kind, record_id, str(item.get("company") or ""),
                     str(item.get("role") or item.get("title") or ""),
                     str(item.get("lifecycle_stage") or item.get("status") or ""),
                     str(item.get("updated_at") or _now()), json.dumps(item, ensure_ascii=False)))
    with _LOCK, _connect() as db:
        db.execute("DELETE FROM records WHERE kind=?", (kind,))
        db.executemany(
            "INSERT INTO records(kind,record_id,company,role,status,updated_at,payload) VALUES(?,?,?,?,?,?,?)",
            rows,
        )
    return len(rows)


def record_counts() -> dict[str, int]:
    initialize()
    with _connect() as db:
        return {str(row["kind"]): int(row["count"]) for row in db.execute(
            "SELECT kind, COUNT(*) AS count FROM records GROUP BY kind").fetchall()}


def save_checkpoint(job_id: str, **values: Any) -> dict[str, Any]:
    initialize()
    existing = get_checkpoint(job_id) or {}
    row = {
        "job_id": job_id,
        "status": values.get("status", existing.get("status", "in_progress")),
        "step": values.get("step", existing.get("step", "started")),
        "url": values.get("url", existing.get("url", "")),
        "resume_path": values.get("resume_path", existing.get("resume_path", "")),
        "completed_fields": values.get("completed_fields", existing.get("completed_fields", [])),
        "unanswered_fields": values.get("unanswered_fields", existing.get("unanswered_fields", [])),
        "answers": values.get("answers", existing.get("answers", {})),
        "evidence_path": values.get("evidence_path", existing.get("evidence_path", "")),
        "message": values.get("message", existing.get("message", "")),
        "updated_at": _now(),
    }
    with _LOCK, _connect() as db:
        db.execute(
            """INSERT INTO checkpoints(job_id,status,step,url,resume_path,completed_fields,
                   unanswered_fields,answers,evidence_path,message,updated_at)
               VALUES(:job_id,:status,:step,:url,:resume_path,:completed_fields,:unanswered_fields,
                   :answers,:evidence_path,:message,:updated_at)
               ON CONFLICT(job_id) DO UPDATE SET status=excluded.status, step=excluded.step,
                   url=excluded.url, resume_path=excluded.resume_path,
                   completed_fields=excluded.completed_fields, unanswered_fields=excluded.unanswered_fields,
                   answers=excluded.answers, evidence_path=excluded.evidence_path,
                   message=excluded.message, updated_at=excluded.updated_at""",
            {**row,
             "completed_fields": json.dumps(row["completed_fields"], ensure_ascii=False),
             "unanswered_fields": json.dumps(row["unanswered_fields"], ensure_ascii=False),
             "answers": json.dumps(row["answers"], ensure_ascii=False)},
        )
    return row


def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    for key in ("completed_fields", "unanswered_fields", "answers", "details"):
        if key in result:
            try:
                result[key] = json.loads(result[key])
            except Exception:
                result[key] = [] if key.endswith("fields") else {}
    return result


def get_checkpoint(job_id: str) -> dict[str, Any] | None:
    initialize()
    with _connect() as db:
        return _decode(db.execute("SELECT * FROM checkpoints WHERE job_id=?", (job_id,)).fetchone())


def checkpoints(limit: int = 250) -> list[dict[str, Any]]:
    initialize()
    with _connect() as db:
        return [_decode(row) for row in db.execute(
            "SELECT * FROM checkpoints ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()]


def add_activity(kind: str, status: str, message: str, job_id: str = "", **details: Any) -> None:
    initialize()
    with _LOCK, _connect() as db:
        db.execute(
            "INSERT INTO activity(kind,status,job_id,message,details,created_at) VALUES(?,?,?,?,?,?)",
            (kind, status, job_id, message, json.dumps(details, ensure_ascii=False), _now()),
        )


def activities(limit: int = 250) -> list[dict[str, Any]]:
    initialize()
    with _connect() as db:
        return [_decode(row) for row in db.execute(
            "SELECT * FROM activity ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()]


def mirror_json(name: str, path: Path) -> bool:
    """Transactionally mirror a legacy JSON store while APIs migrate to SQLite."""
    initialize()
    if not path.is_file():
        return False
    try:
        payload = path.read_text(encoding="utf-8")
        json.loads(payload)
        stamp = path.stat().st_mtime_ns
    except (OSError, ValueError):
        return False
    with _LOCK, _connect() as db:
        db.execute(
            """INSERT INTO json_snapshots(name,payload,source_mtime_ns,updated_at) VALUES(?,?,?,?)
               ON CONFLICT(name) DO UPDATE SET payload=excluded.payload,
                 source_mtime_ns=excluded.source_mtime_ns, updated_at=excluded.updated_at""",
            (name, payload, stamp, _now()),
        )
    return True


def status() -> dict[str, Any]:
    initialize()
    with _connect() as db:
        return {
            "path": str(DB_PATH),
            "schema_version": db.execute("SELECT MAX(version) FROM schema_meta").fetchone()[0],
            "checkpoints": db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0],
            "activity_events": db.execute("SELECT COUNT(*) FROM activity").fetchone()[0],
            "snapshots": db.execute("SELECT COUNT(*) FROM json_snapshots").fetchone()[0],
            "records": record_counts(),
        }


initialize()
