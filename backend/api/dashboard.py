# ============================================================
#  HIREX v1.0.0 — Dashboard Analytics & History Endpoint
#  ------------------------------------------------------------
#  Provides:
#   • Aggregated event summaries (counts, not clones)
#   • Tone/mode analytics
#   • Weekly trend data (Mon..Sun)
#   • Recent history listing (deduped per Company__Role)
#   • Event type registry
#   • Robust log reading & safe JSONL parsing
#   • Total resumes count for dashboard stats
#  Author: Sri Akash Kadali
# ============================================================

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from backend.core import config

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

# ============================================================
# 📁 Paths (read from config only; directories auto-created)
# ============================================================
LOG_PATH = Path(config.LOG_PATH)
HISTORY_PATH = Path(config.HISTORY_PATH)

# Output directories for counting files
OPTIMIZED_DIR = Path(config.OPTIMIZED_DIR)
HUMANIZED_DIR = Path(config.HUMANIZED_DIR)
COVER_LETTERS_DIR = Path(config.COVER_LETTERS_DIR)
CONTEXTS_DIR = Path(config.CONTEXTS_DIR)

for p in (LOG_PATH.parent, HISTORY_PATH.parent):
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# 🧩 Helpers: Time / JSONL / Normalization / Dedupe
# ============================================================
def _now_iso() -> str:
    """UTC now in ISO-8601 with trailing Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_jsonl(path: Path, limit: int = 500) -> List[Dict[str, Any]]:
    """Safely read the last N lines of a JSONL file (newest first)."""
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
        records: List[Dict[str, Any]] = []
        # Reverse so newest first
        for line in reversed(lines):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                continue
        return records
    except Exception:
        return []


def _iso(ts: Optional[str]) -> str:
    """Coerce to ISO timestamp string (safe fallback to now)."""
    if not ts:
        return _now_iso()
    try:
        _ = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return ts
    except Exception:
        return _now_iso()


def _event_name(e: Dict[str, Any]) -> str:
    """Normalize event/type name."""
    return (e.get("event") or e.get("type") or "unknown").lower()


def _company_role_from_meta(e: Dict[str, Any]) -> Tuple[str, str]:
    """Extract (company, role) from common locations."""
    m = e.get("meta") or {}
    company = (m.get("company") or e.get("company") or "").strip()
    role = (m.get("role") or e.get("role") or "").strip()
    return company, role


def _ts_value(e: Dict[str, Any]) -> float:
    """Timestamp (epoch seconds) for ordering/dedupe."""
    ts_raw = e.get("timestamp") or e.get("time") or ""
    try:
        return datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _dedupe_company_role(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collapse multiple records for the same (company, role) combo,
    keeping the most recent one. This prevents dashboard tables
    from exploding with clones of similar actions for the same job.
    """
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in records:
        company, role = _company_role_from_meta(r)
        if not (company and role):
            # keep entries with no company/role (diagnostics) — single latest
            key = ("", "")
            prev = best.get(key)
            if prev is None or _ts_value(r) > _ts_value(prev):
                best[key] = r
            continue

        k = (company, role)
        prev = best.get(k)
        if prev is None or _ts_value(r) > _ts_value(prev):
            best[k] = r

    # Keep deterministic newest-first order
    out = sorted(best.values(), key=_ts_value, reverse=True)
    return out


def _count_files_in_dir(directory: Path, extensions: List[str] = None) -> int:
    """Count files in a directory, optionally filtering by extension."""
    if not directory.exists():
        return 0
    try:
        if extensions:
            return sum(1 for f in directory.iterdir() if f.is_file() and f.suffix.lower() in extensions)
        return sum(1 for f in directory.iterdir() if f.is_file())
    except Exception:
        return 0


def _count_context_files() -> int:
    """Count the number of context JSON files (represents unique job applications)."""
    if not CONTEXTS_DIR.exists():
        return 0
    try:
        return sum(1 for f in CONTEXTS_DIR.glob("*.json") if f.is_file())
    except Exception:
        return 0


# ============================================================
# 📊 Aggregations
# ============================================================
def summarize_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate analytic aggregates for dashboard visualizations."""
    summary: Dict[str, Any] = {
        "total_events": len(events),
        "optimize_runs": 0,
        "coverletters": 0,
        "superhuman_calls": 0,
        "talk_queries": 0,
        "mastermind_chats": 0,
        "tones": Counter(),
        "modes": Counter(),
        "avg_resume_length": 0.0,
        "distinct_company_roles": 0,
        # NEW: File-based counts for accurate totals
        "total_resumes": 0,
        "total_optimized": 0,
        "total_humanized": 0,
        "total_coverletters_files": 0,
        "total_contexts": 0,
    }

    # Distinct (company, role) counter for high-level dedup metric
    distinct_pairs = set()

    total_len = 0
    len_count = 0
    for e in events:
        evt = _event_name(e)
        meta = e.get("meta", {}) or {}

        if "optimize" in evt:
            summary["optimize_runs"] += 1
        if "coverletter" in evt:
            summary["coverletters"] += 1
        if "superhuman" in evt or "humanize" in evt:
            summary["superhuman_calls"] += 1
        if "talk" in evt:
            summary["talk_queries"] += 1
        if "mastermind" in evt:
            summary["mastermind_chats"] += 1

        tone = str(meta.get("tone", "balanced")).lower()
        mode = str(meta.get("mode", "general")).lower()
        if tone:
            summary["tones"][tone] += 1
        if mode:
            summary["modes"][mode] += 1

        try:
            rl = int(meta.get("resume_len") or 0)
            if rl > 0:
                total_len += rl
                len_count += 1
        except Exception:
            pass

        c, r = _company_role_from_meta(e)
        if c or r:
            distinct_pairs.add((c, r))

    # Average only over entries that actually reported resume_len
    denom = max(len_count, 1)
    summary["avg_resume_length"] = round(total_len / denom, 2)
    summary["distinct_company_roles"] = len(distinct_pairs)

    # Count actual files in output directories for accurate totals
    summary["total_optimized"] = _count_files_in_dir(OPTIMIZED_DIR, [".pdf"])
    summary["total_humanized"] = _count_files_in_dir(HUMANIZED_DIR, [".pdf"])
    summary["total_coverletters_files"] = _count_files_in_dir(COVER_LETTERS_DIR, [".pdf"])
    summary["total_contexts"] = _count_context_files()
    
    # Total resumes = optimized + humanized (unique by counting optimized as base)
    # Or use contexts as the authoritative count of job applications
    summary["total_resumes"] = max(
        summary["total_optimized"],
        summary["total_humanized"],
        summary["total_contexts"],
        summary["distinct_company_roles"]
    )

    # Convert counters to plain dicts for JSON
    summary["tones"] = dict(summary["tones"])
    summary["modes"] = dict(summary["modes"])
    return summary


def summarize_history(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract recent high-level activity (for dashboard table)."""
    out: List[Dict[str, Any]] = []
    for h in records:
        meta = h.get("meta", {}) or {}
        ts = _iso(h.get("timestamp") or h.get("time"))
        evt = _event_name(h)
        out.append(
            {
                "timestamp": ts,
                "event": evt,
                "company": meta.get("company", ""),
                "role": meta.get("role", ""),
                "tone": meta.get("tone", "balanced"),
                "score": meta.get("fit_score", None),
                "length": meta.get("resume_len", None),
                "source": h.get("origin", "system"),
            }
        )
    return out


def weekly_trend(records: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """
    Build Mon..Sun trend counts per category.
    """
    buckets = {
        "optimizations": [0] * 7,
        "coverletters": [0] * 7,
        "superhuman": [0] * 7,
        "mastermind": [0] * 7,
        "talk": [0] * 7,
    }

    def _dow(ts: str) -> int:
        try:
            d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            # Monday=0
            return d.weekday()
        except Exception:
            return 0

    for r in records:
        evt = _event_name(r)
        ts = _iso(r.get("timestamp") or r.get("time"))
        i = _dow(ts)
        if "optimize" in evt:
            buckets["optimizations"][i] += 1
        elif "coverletter" in evt:
            buckets["coverletters"][i] += 1
        elif "superhuman" in evt or "humanize" in evt:
            buckets["superhuman"][i] += 1
        elif "mastermind" in evt:
            buckets["mastermind"][i] += 1
        elif "talk" in evt:
            buckets["talk"][i] += 1

    return buckets


# ============================================================
# 🚀 Root: Combined payload (summary + trend + history)
# ============================================================
@router.get("")
@router.get("/")  # compatibility
async def dashboard_root(
    limit: int = Query(300, ge=1, le=2000),
    dedupe: bool = Query(True, description="Collapse multiple actions per (Company,Role) to the newest one"),
):
    """
    Main dashboard endpoint returning all data needed for the frontend.
    Returns summary stats, weekly trends, and recent history.
    """
    events = _read_jsonl(LOG_PATH, limit)
    history = _read_jsonl(HISTORY_PATH, limit)
    records = history or events

    if dedupe:
        records = _dedupe_company_role(records)

    # Even if no records, still return file-based counts
    summary = summarize_events(records) if records else _get_file_based_summary()
    
    return {
        "summary": summary,
        "trend": weekly_trend(records) if records else _empty_trend(),
        "history": summarize_history(records)[:100] if records else [],
        "updated": _now_iso(),
        # Additional fields for frontend compatibility
        "total_resumes": summary.get("total_resumes", 0),
        "optimize_runs": summary.get("optimize_runs", 0),
        "coverletters": summary.get("coverletters", 0),
        "superhuman_calls": summary.get("superhuman_calls", 0),
        "talk_queries": summary.get("talk_queries", 0),
        "mastermind_chats": summary.get("mastermind_chats", 0),
    }


def _get_file_based_summary() -> Dict[str, Any]:
    """Get summary based purely on file counts when no events exist."""
    total_optimized = _count_files_in_dir(OPTIMIZED_DIR, [".pdf"])
    total_humanized = _count_files_in_dir(HUMANIZED_DIR, [".pdf"])
    total_coverletters = _count_files_in_dir(COVER_LETTERS_DIR, [".pdf"])
    total_contexts = _count_context_files()
    
    return {
        "total_events": 0,
        "optimize_runs": total_optimized,
        "coverletters": total_coverletters,
        "superhuman_calls": total_humanized,
        "talk_queries": 0,
        "mastermind_chats": 0,
        "tones": {},
        "modes": {},
        "avg_resume_length": 0.0,
        "distinct_company_roles": total_contexts,
        "total_resumes": max(total_optimized, total_humanized, total_contexts),
        "total_optimized": total_optimized,
        "total_humanized": total_humanized,
        "total_coverletters_files": total_coverletters,
        "total_contexts": total_contexts,
    }


def _empty_trend() -> Dict[str, List[int]]:
    """Return empty trend data structure."""
    return {
        "optimizations": [0] * 7,
        "coverletters": [0] * 7,
        "superhuman": [0] * 7,
        "mastermind": [0] * 7,
        "talk": [0] * 7,
    }


# ============================================================
# 🚀 Endpoint: /summary
# ============================================================
@router.get("/summary")
async def get_summary(
    limit: int = Query(300, ge=1, le=2000),
    dedupe: bool = Query(True),
):
    """
    Aggregated dashboard summary used for top metrics and charts.
    Combines analytics from events.jsonl and history.jsonl.
    Also includes file-based counts for accurate totals.
    """
    events = _read_jsonl(LOG_PATH, limit)
    history = _read_jsonl(HISTORY_PATH, limit)

    # Prefer history when present, but combine both for complete picture
    records = history or events
    
    if not events and not history:
        # Return file-based summary even when no events logged
        summary = _get_file_based_summary()
        return JSONResponse({
            "message": "No event logs available. Showing file-based counts.",
            "summary": summary,
            "recent": [],
            "updated": _now_iso(),
            # Top-level fields for easy frontend access
            "total_resumes": summary.get("total_resumes", 0),
            "optimize_runs": summary.get("optimize_runs", 0),
            "coverletters": summary.get("coverletters", 0),
            "superhuman_calls": summary.get("superhuman_calls", 0),
            "talk_queries": summary.get("talk_queries", 0),
            "mastermind_chats": summary.get("mastermind_chats", 0),
        })

    if dedupe:
        records = _dedupe_company_role(records)

    summary = summarize_events(records)
    hist_data = summarize_history(records)

    return {
        "summary": summary,
        "recent": hist_data[:100],
        "updated": _now_iso(),
        # Top-level fields for easy frontend access
        "total_resumes": summary.get("total_resumes", 0),
        "optimize_runs": summary.get("optimize_runs", 0),
        "coverletters": summary.get("coverletters", 0),
        "superhuman_calls": summary.get("superhuman_calls", 0),
        "talk_queries": summary.get("talk_queries", 0),
        "mastermind_chats": summary.get("mastermind_chats", 0),
    }


# ============================================================
# 🚀 Endpoint: /trend
# ============================================================
@router.get("/trend")
async def get_trend(
    limit: int = Query(300, ge=1, le=2000),
    dedupe: bool = Query(True),
):
    """Weekly Mon..Sun trend counts by category."""
    history = _read_jsonl(HISTORY_PATH, limit) or _read_jsonl(LOG_PATH, limit)
    if dedupe:
        history = _dedupe_company_role(history)
    return {"trend": weekly_trend(history), "updated": _now_iso()}


# ============================================================
# 🚀 Endpoint: /recent
# ============================================================
@router.get("/recent")
async def get_recent(
    limit: int = Query(100, ge=1, le=1000),
    dedupe: bool = Query(True),
):
    """Returns a chronological list of recent user-visible actions."""
    history = _read_jsonl(HISTORY_PATH, limit) or _read_jsonl(LOG_PATH, limit)
    if dedupe:
        history = _dedupe_company_role(history)
    return {"events": summarize_history(history), "updated": _now_iso()}


# ============================================================
# 🚀 Endpoint: /types
# ============================================================
@router.get("/types")
async def list_event_types():
    """Returns a deduplicated list of event types for frontend filters."""
    # Combine both sources for a more complete registry
    events = _read_jsonl(LOG_PATH, 1000) + _read_jsonl(HISTORY_PATH, 1000)
    types = sorted(
        {
            (e.get("event") or e.get("type") or "").lower()
            for e in events
            if (e.get("event") or e.get("type"))
        }
    )
    return {"types": types, "updated": _now_iso()}


# ============================================================
# 🧠 Endpoint: /metrics
# ============================================================
@router.get("/metrics")
async def metrics_summary(
    limit: int = Query(500, ge=1, le=3000),
    dedupe: bool = Query(True),
):
    """
    Returns lightweight numeric insights (for quick dashboard cards).
    This is the primary endpoint for dashboard stat cards.
    """
    events = _read_jsonl(LOG_PATH, limit)
    
    # Always include file-based counts for accuracy
    total_optimized = _count_files_in_dir(OPTIMIZED_DIR, [".pdf"])
    total_humanized = _count_files_in_dir(HUMANIZED_DIR, [".pdf"])
    total_coverletters_files = _count_files_in_dir(COVER_LETTERS_DIR, [".pdf"])
    total_contexts = _count_context_files()
    
    if not events:
        return {
            "optimize": total_optimized,
            "coverletters": total_coverletters_files,
            "superhuman": total_humanized,
            "talk": 0,
            "mastermind": 0,
            "distinct_company_roles": total_contexts,
            "total_resumes": max(total_optimized, total_humanized, total_contexts),
            "total_optimized": total_optimized,
            "total_humanized": total_humanized,
            "total_coverletters_files": total_coverletters_files,
            "total_contexts": total_contexts,
            "updated": _now_iso(),
        }

    records = _dedupe_company_role(events) if dedupe else events
    summary = summarize_events(records)
    
    return {
        "optimize": max(summary["optimize_runs"], total_optimized),
        "coverletters": max(summary["coverletters"], total_coverletters_files),
        "superhuman": max(summary["superhuman_calls"], total_humanized),
        "talk": summary["talk_queries"],
        "mastermind": summary["mastermind_chats"],
        "distinct_company_roles": max(summary["distinct_company_roles"], total_contexts),
        "total_resumes": summary["total_resumes"],
        "total_optimized": total_optimized,
        "total_humanized": total_humanized,
        "total_coverletters_files": total_coverletters_files,
        "total_contexts": total_contexts,
        "updated": _now_iso(),
    }


# ============================================================
# 🧾 Endpoint: /raw
# ============================================================
@router.get("/raw")
async def raw_dump(limit: int = Query(100, ge=1, le=2000)):
    """
    Developer-only diagnostic endpoint: returns raw JSON lines.
    Use for backend debugging or analytics export.
    """
    events = _read_jsonl(LOG_PATH, limit)
    return {"count": len(events), "events": events, "updated": _now_iso()}


# ============================================================
# 📊 Endpoint: /stats (NEW - simplified stats for dashboard cards)
# ============================================================
@router.get("/stats")
async def get_stats():
    """
    Simplified stats endpoint specifically for dashboard stat cards.
    Returns only the key metrics needed for the main dashboard view.
    """
    # File-based counts (most accurate)
    total_optimized = _count_files_in_dir(OPTIMIZED_DIR, [".pdf"])
    total_humanized = _count_files_in_dir(HUMANIZED_DIR, [".pdf"])
    total_coverletters = _count_files_in_dir(COVER_LETTERS_DIR, [".pdf"])
    total_contexts = _count_context_files()
    
    # Event-based counts
    events = _read_jsonl(LOG_PATH, 500)
    talk_count = sum(1 for e in events if "talk" in _event_name(e))
    mastermind_count = sum(1 for e in events if "mastermind" in _event_name(e))
    
    return {
        "total_resumes": max(total_optimized, total_humanized, total_contexts),
        "total_optimized": total_optimized,
        "total_humanized": total_humanized,
        "total_coverletters": total_coverletters,
        "total_contexts": total_contexts,
        "talk_queries": talk_count,
        "mastermind_chats": mastermind_count,
        "updated": _now_iso(),
    }


# ============================================================
# 🔍 Endpoint: /health (NEW - health check)
# ============================================================
@router.get("/health")
@router.get("/ping")
async def health_check():
    """Health check endpoint for the dashboard service."""
    return {
        "ok": True,
        "service": "dashboard",
        "version": "v1.0.0",
        "updated": _now_iso(),
        "paths": {
            "log_exists": LOG_PATH.exists(),
            "history_exists": HISTORY_PATH.exists(),
            "optimized_dir_exists": OPTIMIZED_DIR.exists(),
            "humanized_dir_exists": HUMANIZED_DIR.exists(),
            "coverletters_dir_exists": COVER_LETTERS_DIR.exists(),
            "contexts_dir_exists": CONTEXTS_DIR.exists(),
        }
    }


# ============================================================
# 📁 Endpoint: /files (NEW - list generated files)
# ============================================================
@router.get("/files")
async def list_generated_files(
    file_type: str = Query("all", description="Filter by type: optimized, humanized, coverletters, contexts, all"),
    limit: int = Query(50, ge=1, le=200),
):
    """
    List generated files for the dashboard file browser.
    Returns file metadata for display in the UI.
    """
    files = []
    
    def _get_file_info(path: Path, category: str) -> Dict[str, Any]:
        try:
            stat = path.stat()
            return {
                "name": path.name,
                "category": category,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "path": str(path),
            }
        except Exception:
            return None
    
    if file_type in ("all", "optimized") and OPTIMIZED_DIR.exists():
        for f in sorted(OPTIMIZED_DIR.glob("*.pdf"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
            info = _get_file_info(f, "optimized")
            if info:
                files.append(info)
    
    if file_type in ("all", "humanized") and HUMANIZED_DIR.exists():
        for f in sorted(HUMANIZED_DIR.glob("*.pdf"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
            info = _get_file_info(f, "humanized")
            if info:
                files.append(info)
    
    if file_type in ("all", "coverletters") and COVER_LETTERS_DIR.exists():
        for f in sorted(COVER_LETTERS_DIR.glob("*.pdf"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
            info = _get_file_info(f, "coverletter")
            if info:
                files.append(info)
    
    if file_type in ("all", "contexts") and CONTEXTS_DIR.exists():
        for f in sorted(CONTEXTS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
            info = _get_file_info(f, "context")
            if info:
                files.append(info)
    
    # Sort by modified date (newest first)
    files.sort(key=lambda x: x.get("modified", ""), reverse=True)
    
    return {
        "files": files[:limit],
        "total": len(files),
        "updated": _now_iso(),
    }