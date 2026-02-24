# ============================================================
#  HIREX v1.0.0 — context_store.py (FINAL, stable-key + dedupe)
#  Store & retrieve JD + (optimized/humanized) resume + cover-letter.
#  File-per-(Company,Role) using key:  {safe_company}__{safe_role}.json
#  "title_for_memory" keeps a timestamped title for UI memory only.
#  Back-compat:
#   • /api/context/list returns items with both key and id (id===key)
#   • /api/context/get accepts key=..., id_or_title=..., latest=true
#   • /api/context/get flattens nested fields for older UIs
# ============================================================

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from fastapi import APIRouter, HTTPException, Form, Query

from backend.core import config
from backend.core.utils import safe_filename, log_event, ensure_dir

router = APIRouter(prefix="/api/context", tags=["context"])

# ---- Resolve contexts directory with robust fallback ----
def _default_contexts_dir() -> Path:
    # Conservative default inside backend data; ensures writeable path exists
    here = Path(__file__).resolve().parent
    root = here.parents[2] if len(here.parents) >= 3 else here
    return (root / "backend" / "data" / "contexts")

_get_dir = getattr(config, "get_contexts_dir", None)
try:
    CONTEXT_DIR: Path = Path(_get_dir()) if callable(_get_dir) else _default_contexts_dir()
except Exception:
    CONTEXT_DIR: Path = _default_contexts_dir()

ensure_dir(CONTEXT_DIR)


# ---------------------- internal helpers ----------------------

def _nowstamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _iso_utc() -> str:
    # ISO 8601 with 'Z' suffix for UTC
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _make_key(company: str, role: str) -> str:
    return f"{safe_filename(company)}__{safe_filename(role)}"


def _path_for_key(key: str) -> Path:
    return CONTEXT_DIR / f"{safe_filename(key)}.json"


def _read(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _coerce_key(d: Dict[str, Any], path: Optional[Path] = None) -> str:
    # Prefer explicit key in file; else derive from company/role; else from filename (legacy)
    if "key" in d and str(d["key"]).strip():
        return str(d["key"]).strip()
    company = str(d.get("company") or "").strip()
    role = str(d.get("role") or "").strip()
    if company and role:
        return _make_key(company, role)
    if path is not None:
        return path.stem
    return ""


def _updated_at(d: Dict[str, Any], default_ts: float = 0.0) -> float:
    try:
        ts = d.get("updated_at") or d.get("saved_at") or ""
        if isinstance(ts, str) and ts:
            # accept '...Z' or bare ISO
            iso = ts.rstrip("Z")
            return datetime.fromisoformat(iso).timestamp()
    except Exception:
        pass
    return default_ts


def _compact_meta(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return a compact row for list(), compatible with old UIs."""
    title = d.get("title_for_memory") or d.get("title") or f"{d.get('company','')} — {d.get('role','')}"
    has_opt = bool(((d.get("optimized") or {}).get("tex")) or d.get("resume_tex"))
    has_hum = bool(((d.get("humanized") or {}).get("tex")) or d.get("humanized_tex"))
    cl = d.get("cover_letter") or {}
    has_cl = bool(cl.get("tex") or cl.get("pdf_b64") or d.get("cover_letter_tex") or d.get("cover_letter_pdf_b64"))
    return {
        "key": key,
        "id": key,                     # back-compat for frontends expecting .id
        "title": title,
        "company": d.get("company"),
        "role": d.get("role"),
        "updated_at": d.get("updated_at") or d.get("saved_at"),
        "has_optimized": has_opt,
        "has_humanized": has_hum,
        "has_cover_letter": has_cl,
        "model": d.get("model"),
        "fit_score": d.get("fit_score"),
    }


def _load_all_contexts() -> List[Tuple[str, Path, Dict[str, Any]]]:
    """Return list of (key, path, data)."""
    items: List[Tuple[str, Path, Dict[str, Any]]] = []
    for p in CONTEXT_DIR.glob("*.json"):
        data = _read(p)
        key = _coerce_key(data, p) or p.stem
        items.append((key, p, data))
    return items


def _merge_update(existing: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge PATCH into existing:
      - Only overwrite nested blobs if new fields are non-empty.
      - Maintain 'optimized', 'humanized', 'cover_letter' subobjects.
    """
    out = dict(existing or {})
    # shallow updates
    for k in ["key", "company", "role", "jd_text", "model", "fit_score"]:
        v = patch.get(k, None)
        if v is not None and (not isinstance(v, str) or v.strip() != ""):
            out[k] = v

    # maintain timestamped memory title
    if "title_for_memory" in patch and patch["title_for_memory"]:
        out["title_for_memory"] = patch["title_for_memory"]

    # legacy flat fields -> normalize into subobjects
    if "resume_tex" in patch or "pdf_base64" in patch:
        opt = dict(out.get("optimized") or {})
        if str(patch.get("resume_tex", "")).strip():
            opt["tex"] = patch["resume_tex"]
        if str(patch.get("pdf_base64", "")).strip():
            opt["pdf_b64"] = patch["pdf_base64"]
        out["optimized"] = opt

    if "humanized_tex" in patch or "pdf_base64_humanized" in patch:
        hum = dict(out.get("humanized") or {})
        if str(patch.get("humanized_tex", "")).strip():
            hum["tex"] = patch["humanized_tex"]
        if str(patch.get("pdf_base64_humanized", "")).strip():
            hum["pdf_b64"] = patch["pdf_base64_humanized"]
        out["humanized"] = hum

    # modern subobjects (prefer these if provided)
    if "optimized" in patch and isinstance(patch["optimized"], dict):
        base = dict(out.get("optimized") or {})
        for k, v in patch["optimized"].items():
            if v is not None and (not isinstance(v, str) or v.strip() != ""):
                base[k] = v
        out["optimized"] = base

    if "humanized" in patch and isinstance(patch["humanized"], dict):
        base = dict(out.get("humanized") or {})
        for k, v in patch["humanized"].items():
            if v is not None and (not isinstance(v, str) or v.strip() != ""):
                base[k] = v
        out["humanized"] = base

    if "cover_letter" in patch and isinstance(patch["cover_letter"], dict):
        base = dict(out.get("cover_letter") or {})
        for k, v in patch["cover_letter"].items():
            if v is not None and (not isinstance(v, str) or v.strip() != ""):
                base[k] = v
        out["cover_letter"] = base

    out["updated_at"] = _iso_utc()
    return out


def _flatten_for_frontend(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provide flat aliases expected by older front-ends:
      resume_tex, pdf_base64, humanized_tex, pdf_base64_humanized,
      cover_letter_tex, cover_letter_pdf_b64
    """
    out = dict(d)
    opt = d.get("optimized") or {}
    hum = d.get("humanized") or {}
    cl  = d.get("cover_letter") or {}

    # Resume (optimized)
    out["resume_tex"] = out.get("resume_tex") or opt.get("tex") or ""
    out["pdf_base64"] = out.get("pdf_base64") or opt.get("pdf_b64") or ""

    # Humanized
    out["humanized_tex"] = out.get("humanized_tex") or hum.get("tex") or ""
    out["pdf_base64_humanized"] = out.get("pdf_base64_humanized") or hum.get("pdf_b64") or ""

    # Cover letter
    out["cover_letter_tex"] = out.get("cover_letter_tex") or cl.get("tex") or ""
    out["cover_letter_pdf_b64"] = out.get("cover_letter_pdf_b64") or cl.get("pdf_b64") or ""

    # Convenience — mirror some names used by UIs
    out["company_name"] = out.get("company")
    out["role_name"] = out.get("role")
    out["title"] = out.get("title_for_memory") or out.get("title") or f"{out.get('company','')} — {out.get('role','')}"

    return out


# -------------------------- routes ----------------------------

@router.post("/save")
async def save_context(
    company: str = Form(...),
    role: str = Form(...),
    jd_text: str = Form(""),

    # Legacy flat payloads from older UI (will be normalized)
    resume_tex: str = Form(""),
    humanized_tex: str = Form(""),
    pdf_base64: str = Form(""),
    pdf_base64_humanized: str = Form(""),

    # Optional modern nested payloads (optimized/humanized/cover_letter)
    optimized: str = Form("", description="JSON string of optimized block (tex, pdf_b64, pdf_path)"),
    humanized: str = Form("", description="JSON string of humanized block (tex, pdf_b64, pdf_path, enabled)"),
    cover_letter: str = Form("", description="JSON string of cover_letter block (tex, pdf_b64, pdf_path, tone, length)"),

    model: str = Form(""),
    fit_score: str = Form(""),
):
    """
    Persist context under a STABLE key per (Company, Role).
    Overwrites previous file with the same key (dedup by design).
    Keeps a 'title_for_memory' that includes a timestamp for UI history labels.
    """
    key = _make_key(company, role)
    path = _path_for_key(key)

    # Parse optional JSON blocks if provided
    def _parse_json_str(s: str) -> Dict[str, Any]:
        try:
            return json.loads(s) if s and s.strip().startswith("{") else {}
        except Exception:
            return {}

    patch: Dict[str, Any] = {
        "key": key,
        "company": company,
        "role": role,
        "jd_text": jd_text,
        "model": model or getattr(config, "DEFAULT_MODEL", "gpt-4o-mini"),
        "fit_score": fit_score,
        "title_for_memory": f"{safe_filename(company)}_{safe_filename(role)}_{_nowstamp()}",
    }

    # legacy flat fields
    if resume_tex or pdf_base64:
        patch.update({"resume_tex": resume_tex, "pdf_base64": pdf_base64})
    if humanized_tex or pdf_base64_humanized:
        patch.update({"humanized_tex": humanized_tex or resume_tex, "pdf_base64_humanized": pdf_base64_humanized})

    # modern nested overrides (preferred)
    opt_obj = _parse_json_str(optimized)
    hum_obj = _parse_json_str(humanized)
    cl_obj = _parse_json_str(cover_letter)
    if opt_obj: patch["optimized"] = opt_obj
    if hum_obj: patch["humanized"] = hum_obj
    if cl_obj: patch["cover_letter"] = cl_obj

    existing = _read(path) if path.exists() else {}
    merged = _merge_update(existing, patch)
    _write(path, merged)

    log_event("context_saved", {"key": key, "company": company, "role": role, "path": str(path)})
    return {"ok": True, "key": key, "path": str(path), "updated_at": merged["updated_at"]}


@router.get("/list")
async def list_contexts(
    limit: int = Query(50, ge=1, le=500),
    dedupe: bool = Query(True),
):
    """
    List recent contexts (newest first). If legacy timestamped files exist,
    `dedupe=True` collapses multiple files to the single latest per key.
    """
    entries = _load_all_contexts()

    # Build (key -> newest (path,data))
    # If dedupe=False we pseudo-namespace keys to keep all files distinct.
    by_key: Dict[str, Tuple[Path, Dict[str, Any], float]] = {}
    for key, p, d in entries:
        ts = _updated_at(d, p.stat().st_mtime)
        idx_key = key if dedupe else f"{key}::{p.name}"
        cur = by_key.get(idx_key)
        if (cur is None) or (ts > cur[2]):
            by_key[idx_key] = (p, d, ts)

    # Sort newest first and compact
    rows = sorted(by_key.items(), key=lambda kv: kv[1][2], reverse=True)[:limit]
    items = [_compact_meta(d, _coerce_key(d, p)) for (_k, (p, d, _)) in rows]
    return {"items": items}


@router.get("/get")
async def get_context(
    key: str = Query("", description="Stable key: {company}__{role}"),
    id_or_title: str = Query("", description="(Back-compat) Either a stable key or a title_for_memory string."),
    latest: bool = Query(False),
):
    """
    Fetch full saved context by:
      • latest=true (newest item overall), OR
      • key=stable_key, OR
      • id_or_title=(stable_key or title_for_memory) — back-compat

    Response includes flattened fields for older UIs:
      resume_tex, pdf_base64, humanized_tex, pdf_base64_humanized,
      cover_letter_tex, cover_letter_pdf_b64
    """
    # Case 1: newest overall
    # Case 1: newest overall
    if latest or (not key.strip() and not id_or_title.strip()):
        newest_path: Optional[Path] = None
        newest_ts = -1.0
        for k, p, d in _load_all_contexts():
            ts = _updated_at(d, p.stat().st_mtime)
            if ts > newest_ts:
                newest_ts, newest_path = ts, p
        if not newest_path:
            raise HTTPException(status_code=404, detail="Context not found")
        return _flatten_for_frontend(_read(newest_path))

    # Case 2: explicit stable key
    if key.strip():
        path = _path_for_key(key)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Context not found")
        return _flatten_for_frontend(_read(path))

    # Case 3: back-compat lookup via id_or_title (try key match, then title_for_memory)
    wanted = id_or_title.strip()
    # Try as key first
    candidate = _path_for_key(wanted)
    if candidate.exists():
        return _flatten_for_frontend(_read(candidate))

    # Fallback: scan for title_for_memory match (exact, then loose)
    best: Optional[Dict[str, Any]] = None
    best_ts = -1.0
    for k, p, d in _load_all_contexts():
        title = (d.get("title_for_memory") or d.get("title") or "").strip()
        if title == wanted or safe_filename(title) == safe_filename(wanted):
            ts = _updated_at(d, p.stat().st_mtime)
            if ts > best_ts:
                best, best_ts = d, ts
    if best is not None:
        return _flatten_for_frontend(best)

    raise HTTPException(status_code=404, detail="Context not found")


@router.delete("/delete")
async def delete_context(
    key: str = Query(..., description="Stable key: {company}__{role}")
):
    """
    Delete the single context file for the given (Company, Role) key.
    """
    path = _path_for_key(key)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Context not found")
    path.unlink()
    log_event("context_deleted", {"key": key})
    return {"deleted": True, "key": key}
