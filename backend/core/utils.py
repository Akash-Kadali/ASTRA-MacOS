"""
ASTRA • core/utils.py (v2.1.2)
Common utility functions shared across backend modules.
For this version: No LaTeX escaping or text cleaning — passes LaTeX as-is.
Author: Sri Akash Kadali
"""

from __future__ import annotations

import re
import html
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ============================================================
# 📁 Logging path (prefer config; safe local fallback if import fails)
# ============================================================
try:
    from backend.core import config as _cfg  # type: ignore
    LOG_PATH = Path(getattr(_cfg, "LOG_PATH"))
except Exception:
    # Dev/test fallback (project-local)
    LOG_PATH = Path("data/logs/events.jsonl")

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 🗂️ Filesystem Helpers
# ============================================================
def ensure_dir(p: Path | str) -> None:
    """Create directory (and parents) if it does not exist."""
    Path(p).mkdir(parents=True, exist_ok=True)


# ============================================================
# 🔐 HASHING UTILITIES
# ============================================================
def sha256_str(data: Optional[str]) -> str:
    """Generate a full SHA256 hash of a string."""
    if data is None:
        data = ""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def simple_hash(data: Optional[str], length: int = 8) -> str:
    """Generate a short deterministic hash (used for cache keys or content IDs)."""
    return sha256_str(data or "")[:max(1, int(length))]


# ============================================================
# 🏷️ Naming Helpers (no LaTeX escaping)
# ============================================================
def safe_filename(name: Optional[str]) -> str:
    """Convert a string into a safe, cross-platform filename."""
    if not name:
        return "file"
    # keep letters, digits, underscore, dot, dash; replace others with underscore
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    # Avoid leading/trailing dots or underscores; trim length
    name = name.strip("._") or "file"
    return name[:64]


def slug_part(s: Optional[str]) -> str:
    """
    A permissive slug for path/filename parts:
      - Replace non [A-Za-z0-9_.-] with underscores
      - Keep length generous (no hard trim here; callers can trim if needed)
    """
    s = s or ""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("._")
    return s or "unnamed"


def build_filenames(company: str, role: str) -> Dict[str, str]:
    """
    Standardized final artifact names:

      • Optimized resumes:
          "Sri_{Company}_{Role}.pdf"
      • Humanized resumes:
          "Sri_Kadali_{Company}_{Role}.pdf"
      • Cover letters:
          "Sri_{Company}_{Role}_Cover_Letter.pdf"
    """
    c = slug_part(company)
    r = slug_part(role)
    return {
        "optimized":    f"Sri_{c}_{r}.pdf",
        "humanized":    f"Sri_Kadali_{c}_{r}.pdf",
        "cover_letter": f"Sri_{c}_{r}_Cover_Letter.pdf",
    }


def build_output_paths(company: str, role: str) -> Dict[str, Path]:
    """
    Convenience helper that returns the *full* Paths for each final artifact,
    using directories from config (if available).

    Returns dict with keys: optimized, humanized, cover_letter
    """
    names = build_filenames(company, role)
    try:
        # Prefer config-provided directories (typically absolute/user-data-safe)
        opt_dir = Path(getattr(_cfg, "OPTIMIZED_DIR"))
        hum_dir = Path(getattr(_cfg, "HUMANIZED_DIR"))
        cov_dir = Path(getattr(_cfg, "COVER_LETTERS_DIR"))
    except Exception:
        # Safe fallbacks — project-local relative paths
        opt_dir = Path("data/Optimized")
        hum_dir = Path("data/Humanized")
        cov_dir = Path("data/Cover Letters")

    # Ensure the directories exist
    for d in (opt_dir, hum_dir, cov_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "optimized": (opt_dir / names["optimized"]).resolve(),
        "humanized": (hum_dir / names["humanized"]).resolve(),
        "cover_letter": (cov_dir / names["cover_letter"]).resolve(),
    }


# ============================================================
# 📜 TEXT HELPERS (NO LATEX ESCAPING)
# ============================================================
def tex_escape(text: Optional[str]) -> str:
    """
    Passthrough for LaTeX text (no escaping).
    Used when sending LaTeX to or receiving from OpenAI/Humanize.
    """
    return text or ""


def html_escape(text: Optional[str]) -> str:
    """HTML-escape text for safe display inside web UIs (not LaTeX)."""
    return html.escape(text or "")


def clean_text(text: Optional[str]) -> str:
    """
    Lightweight text cleaner (no normalization, no space compression).
    Keeps LaTeX intact.
    """
    if not text:
        return ""
    return str(text)


# ============================================================
# 🧠 LOGGING & DIAGNOSTIC HELPERS
# ============================================================
def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.utcnow().isoformat() + "Z"


def log_event(event: str, meta: Optional[Dict[str, Any]] = None) -> None:
    """
    Append a JSON line to the global event log and print to console.
    Used by all backend modules for analytics and dashboard.

    Accepts:
      • event: short event string
      • meta : optional dict payload (anything JSON-serializable; non-serializable values coerced to str)
    """
    record = {
        "timestamp": utc_now_iso(),
        "event": str(event),
        "meta": meta or {},
    }

    # Console log (truncate very large metas for readability)
    try:
        preview = json.dumps(record["meta"], ensure_ascii=False, default=str)
        if len(preview) > 800:
            preview = preview[:800] + "…"
        print(f"[{record['timestamp']}] {record['event']} :: {preview}")
    except Exception:
        print(f"[{record['timestamp']}] {record['event']} :: (unserializable meta)")

    # Persistent log (append JSONL)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        print(f"[ASTRA] ⚠️ Failed to write event log: {e}")


def benchmark(name: str):
    """
    Context manager for timing code blocks.

    Example:
        with benchmark("Optimize Resume"):
            run_some_code()
    """
    import time

    class _Timer:
        def __enter__(self):
            self._start = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration_ms = (time.time() - self._start) * 1000.0
            log_event("⏱️ benchmark", {"name": name, "duration_ms": round(duration_ms, 1)})

    return _Timer()


# ============================================================
# 🧪 Local Test (generic for any companies/roles)
# ============================================================
if __name__ == "__main__":
    import sys

    # 1) Quick LaTeX + hash sanity check
    sample = r"""
    \documentclass{article}
    \begin{document}
    Hello \textbf{World!} $E = mc^2$
    \end{document}
    """
    print("Original LaTeX (unchanged):")
    print(sample)
    print("SHA256:", sha256_str(sample))
    print("Short Hash:", simple_hash(sample))
    print("Safe File:", safe_filename("My Resume (final).tex"))

    # 2) Filename + output path checks (accept multiple pairs from stdin or argv)
    pairs: list[tuple[str, str]] = []

    # Read JSONL from stdin: each line like {"company":"...", "role":"..."}
    if sys.stdin and not sys.stdin.isatty():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pairs.append((str(obj["company"]), str(obj["role"])))
            except Exception:
                # Ignore malformed lines
                pass

    # Or accept args as JSON or "Company:Role"
    if not pairs and len(sys.argv) > 1:
        for a in sys.argv[1:]:
            a = a.strip()
            if not a:
                continue
            try:
                obj = json.loads(a)
                pairs.append((str(obj["company"]), str(obj["role"])))
                continue
            except Exception:
                pass
            if ":" in a:
                c, r = a.split(":", 1)
                pairs.append((c.strip(), r.strip()))

    # Fallback instructional example if nothing was provided
    if not pairs:
        print("\nUsage:")
        print("  • cat jobs.jsonl | python -m backend.core.utils")
        print('  • python -m backend.core.utils "Company:Role" ...')
        print('  • python -m backend.core.utils \'{"company":"X","role":"Y"}\' ...')
        pairs = [("ExampleCo", "ExampleRole")]

    # Verify the filename scheme + full paths for each pair
    for company, role in pairs:
        names = build_filenames(company, role)
        paths = build_output_paths(company, role)
        print(f"\n[{company} • {role}]")
        print(" Names :", names)
        print(" Paths :", {k: str(v) for k, v in paths.items()})

    # 3) Micro-benchmark
    with benchmark("Hash Generation"):
        for _ in range(10000):
            sha256_str(sample)
