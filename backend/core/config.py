"""
============================================================
 HIREX • core/config.py
 ------------------------------------------------------------
 Global configuration for backend constants, environment
 variables, and directory paths.

 Version : 2.1.2
 Author  : Sri Akash Kadali
============================================================
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv


# ============================================================
# 🌍 Environment Setup
# ============================================================

_env_loaded = (
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
    or load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
    or load_dotenv()
)


def _clean_env(val: str | None, default: str = "") -> str:
    v = (val if val is not None else default)
    return str(v).strip().strip('"').strip("'")


def _getenv_clean(name: str, default: str = "") -> str:
    return _clean_env(os.getenv(name), default)


# ============================================================
# 📁 Directory Structure (portable / any machine)
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
if not (BASE_DIR / "backend").exists():
    candidate = Path(__file__).resolve().parents[1]
    BASE_DIR = candidate if (candidate / "backend").exists() else Path.cwd()

BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"

DATA_DIR = BACKEND_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
TEMP_LATEX_DIR = CACHE_DIR / "latex_builds"
TEMPLATE_DIR = BACKEND_DIR / "templates"

# Kept for backwards compatibility; not used for PDFs anymore
OUTPUT_DIR = DATA_DIR / "output"
SAMPLES_DIR = DATA_DIR / "samples"
LOGS_DIR = DATA_DIR / "logs"
HISTORY_DIR = DATA_DIR / "history"
MASTERMINDS_DIR = DATA_DIR / "mastermind_sessions"
CONTEXTS_DIR = DATA_DIR / "contexts"

# --- Legacy sample dirs (kept to avoid breaking any imports) ---
SAMPLES_JOB_RESUMES_DIR = SAMPLES_DIR / "Job Resumes"
SAMPLES_COVER_LETTERS_DIR = SAMPLES_DIR / "Cover Letters"
SAMPLES_JOB_RESUME_HUMANIZED_DIR = SAMPLES_DIR / "Job Resume Humanized"

# --- NEW flat libraries (single folder per artifact type) ---
# Default to project-local backend/data/*; allow absolute overrides via .env
def _resolve_env_path(var_name: str, default_path: Path) -> Path:
    raw = _getenv_clean(var_name, "")
    if not raw:
        return default_path
    p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        p = BASE_DIR / p
    return p

OPTIMIZED_DIR = _resolve_env_path("OPTIMIZED_DIR", DATA_DIR / "Optimized")
HUMANIZED_DIR = _resolve_env_path("HUMANIZED_DIR", DATA_DIR / "Humanized")
COVER_LETTERS_DIR = _resolve_env_path("COVER_LETTERS_DIR", DATA_DIR / "Cover Letters")

for d in (
    DATA_DIR,
    CACHE_DIR,
    TEMP_LATEX_DIR,
    TEMPLATE_DIR,
    OUTPUT_DIR,
    SAMPLES_DIR,
    LOGS_DIR,
    HISTORY_DIR,
    MASTERMINDS_DIR,
    CONTEXTS_DIR,
    SAMPLES_JOB_RESUMES_DIR,
    SAMPLES_COVER_LETTERS_DIR,
    SAMPLES_JOB_RESUME_HUMANIZED_DIR,
    OPTIMIZED_DIR,
    HUMANIZED_DIR,
    COVER_LETTERS_DIR,
):
    d.mkdir(parents=True, exist_ok=True)

(LOGS_DIR / "events.jsonl").touch(exist_ok=True)


# ============================================================
# ⚙️ Core Settings
# ============================================================

APP_NAME = "HIREX"
APP_VERSION = "2.1.2"
DEBUG_MODE = _getenv_clean("DEBUG", "true").lower() == "true"

MAX_UPLOAD_MB = int(_getenv_clean("MAX_UPLOAD_MB", "5"))
ALLOWED_EXTENSIONS = {".tex", ".txt"}

DEFAULT_MODEL = _getenv_clean("DEFAULT_MODEL", "gpt-4o-mini")
API_BASE_URL = _getenv_clean("API_BASE_URL", "http://127.0.0.1:8000")

CANDIDATE_NAME = _getenv_clean("CANDIDATE_NAME", "Sri Akash Kadali")
APPLICANT_EMAIL = _getenv_clean("APPLICANT_EMAIL", "kadali18@umd.edu")
APPLICANT_PHONE = _getenv_clean("APPLICANT_PHONE", "+1 240-726-9356")
APPLICANT_CITYSTATE = _getenv_clean("APPLICANT_CITYSTATE", "College Park, MD")

# ============================================================
# 🔐 Security & Secrets
# ============================================================

SECRET_KEY = _getenv_clean("HIREX_SECRET", "hirex-dev-secret")
JWT_ALGORITHM = "HS256"


# ============================================================
# 🤖 API Keys (OpenAI + Humanize)
# ============================================================

OPENAI_API_KEY = _getenv_clean("OPENAI_API_KEY", "")

HUMANIZE_API_KEY = _clean_env(
    os.getenv("HUMANIZE_API_KEY")
    or os.getenv("HUMANIZE_TOKEN")
    or os.getenv("AI_HUMANIZE_KEY")
    or os.getenv("AIHUMANIZE_KEY")
    or ""
)

HUMANIZE_MAIL = _clean_env(
    os.getenv("HUMANIZE_MAIL")
    or os.getenv("AI_HUMANIZE_MAIL")
    or os.getenv("HUMANIZE_API_MAIL")
    or os.getenv("AIHUMANIZE_MAIL")
    or "kadali18@terpmail.umd.edu"
)

# ✅ Default to using Humanize everywhere (can be overridden in .env)
HUMANIZE_DEFAULT_ON = _getenv_clean("HUMANIZE_DEFAULT_ON", "true").lower() in {"1", "true", "yes", "on"}
# Default Humanize mode for the service (0=quality, 1=balance, 2=enhanced)
HUMANIZE_MODE_DEFAULT = _getenv_clean("HUMANIZE_MODE_DEFAULT", "balance").lower()
AIHUMANIZE_MODE_ID = {"quality": "0", "balance": "1", "enhanced": "2"}

HUMANIZE_CONFIG_OK = bool(HUMANIZE_API_KEY and HUMANIZE_MAIL)

if DEBUG_MODE:
    if not OPENAI_API_KEY:
        print("[HIREX] ⚠️ OPENAI_API_KEY not found in environment.")
    if not HUMANIZE_API_KEY:
        print("[HIREX] ⚠️ HUMANIZE_API_KEY not found. Humanize will fail unless fallback is enabled.")
    if not HUMANIZE_MAIL:
        print("[HIREX] ⚠️ HUMANIZE_MAIL not set. It must match your AIHumanize registered email.")
    print(f"[HIREX] ℹ️ HUMANIZE_DEFAULT_ON={HUMANIZE_DEFAULT_ON}, HUMANIZE_MODE_DEFAULT={HUMANIZE_MODE_DEFAULT}")


# ============================================================
# 🧷 Helpers
# ============================================================

def _ensure_file(path: Path, content: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        if DEBUG_MODE:
            print(f"[HIREX] 📄 Created default: {path}")


def _slugify(s: str) -> str:
    import re, unicodedata
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s.-]", "", s)
    s = re.sub(r"\s+", "_", s.strip())
    return s or "unnamed"


def _candidate_prefix() -> str:
    return _slugify(CANDIDATE_NAME) or "Candidate"


# ============================================================
# 🧠 Feature Module Paths & Defaults
# ============================================================

BASE_COVERLETTER_PATH = _resolve_env_path("BASE_COVERLETTER_PATH", SAMPLES_DIR / "base_coverletter.tex")
BASE_RESUME_PATH = _resolve_env_path("BASE_RESUME_PATH", SAMPLES_DIR / "base_resume.tex")

LOG_PATH = LOGS_DIR / "events.jsonl"
HISTORY_PATH = HISTORY_DIR / "history.jsonl"

MASTERMINDS_PATH = MASTERMINDS_DIR
MASTERMINDS_MODEL = _getenv_clean("MASTERMINDS_MODEL", DEFAULT_MODEL)

# ❗ Keep fallback OFF so we don't leak [LOCAL-FALLBACK:*]. Fail hard if Humanize is down.
SUPERHUMAN_LOCAL_ENABLED = _getenv_clean("SUPERHUMAN_LOCAL_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
SUPERHUMAN_MODEL = _getenv_clean("SUPERHUMAN_MODEL", DEFAULT_MODEL)
COVERLETTER_MODEL = _getenv_clean("COVERLETTER_MODEL", DEFAULT_MODEL)
TALK_SUMMARY_MODEL = _getenv_clean("TALK_SUMMARY_MODEL", "gpt-4o-mini")
TALK_ANSWER_MODEL = _getenv_clean("TALK_ANSWER_MODEL", DEFAULT_MODEL)


def is_humanize_enabled() -> bool:
    """
    Humanize is considered enabled when:
      • HUMANIZE_DEFAULT_ON is true (default),
      • we have valid key+mail,
      • and we are NOT using local fallback.
    """
    return HUMANIZE_DEFAULT_ON and HUMANIZE_CONFIG_OK and not SUPERHUMAN_LOCAL_ENABLED


def get_contexts_dir() -> Path:
    """Return the canonical contexts directory used by /api/context and friends."""
    return CONTEXTS_DIR


# ============================================================
# ✨ Portable Default Templates
# ============================================================

_DEFAULT_RESUME_TEX = r"""% HIREX default base_resume.tex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{enumitem}
\pagenumbering{gobble}
\begin{document}
\begin{center}
{\LARGE Your Name}\\
\vspace{2pt}
your.email@example.com \quad | \quad (123) 456-7890 \quad | \url{https://example.com}
\end{center}
\vspace{8pt}
\section*{Summary}
Results-oriented professional with experience in software engineering and AI.
\section*{Experience}
\textbf{Company} \hfill City, ST \\
\emph{Role} \hfill 2023--Present
\begin{itemize}[leftmargin=*]
    \item Bullet 1 describing impact.
    \item Bullet 2 describing impact.
\end{itemize}
\section*{Education}
\textbf{University}, Degree, Year
\end{document}
"""

_DEFAULT_COVERLETTER_TEX = r"""% HIREX default base_coverletter.tex (portable)
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\pagenumbering{gobble}
\begin{document}
\noindent Date: \today

\vspace{10pt}
Hiring Manager \\
Company \\
City, State

\vspace{10pt}
Dear Hiring Manager,

I am excited to apply for the role. My background in software and AI aligns with your needs.

%-----------BODY-START-----------
% (Body content will be injected here by HIREX)
%-----------BODY-END-------------

\vspace{12pt}
Sincerely, \\
Your Name
\end{document}
"""

_ensure_file(BASE_RESUME_PATH, _DEFAULT_RESUME_TEX)
_ensure_file(BASE_COVERLETTER_PATH, _DEFAULT_COVERLETTER_TEX)

DEFAULT_BASE_RESUME = BASE_RESUME_PATH

os.environ.setdefault("BASE_RESUME_PATH", str(BASE_RESUME_PATH))
os.environ.setdefault("BASE_COVERLETTER_PATH", str(BASE_COVERLETTER_PATH))


# ============================================================
# 💰 Model Catalog & Pricing
# ============================================================

OPENAI_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-chat-latest",
    "gpt-5-thinking",
    "gpt-5-thinking-mini",
    "gpt-5-thinking-nano",
    "gpt-5-pro",
    "gpt-4o",
    "gpt-4o-mini",
    "o3",
    "o3-mini",
]

MODEL_ALIASES = {
    "GPT-5 (Auto)": "gpt-5",
    "GPT-5 Fast / Instant": "gpt-5-chat-latest",
    "GPT-5 Thinking": "gpt-5-thinking",
    "GPT-5 Pro": "gpt-5-pro",
    "GPT-5 Mini": "gpt-5-mini",
    "GPT-5 Nano": "gpt-5-nano",
    "GPT-4o": "gpt-4o",
}

OPENAI_MODEL_PRICING = {
    "gpt-5": {"input": 1.25, "output": 10.00, "cached_input": 0.125},
    "gpt-5-mini": {"input": 0.25, "output": 2.00, "cached_input": 0.025},
    "gpt-5-nano": {"input": 0.05, "output": 0.40, "cached_input": 0.005},
    "gpt-5-pro": {"input": 15.00, "output": 120.00},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "o3": {"input": 1.10, "output": 4.40},
    "o3-mini": {"input": 0.60, "output": 2.50},
}

AIHUMANIZE_PLANS = {
    "basic": {"price_month": 6, "words_per_request": 500},
    "starter": {"price_month": 15, "words_per_request": 500},
    "pro": {"price_month": 25, "words_per_request": 1500},
    "premium": {"price_month": 40, "words_per_request": 3000},
}

AVAILABLE_MODELS = {
    "openai": OPENAI_MODELS,
    "aihumanize": ["quality", "balance", "enhanced", "private"],
}

MODEL_PRICING = {
    "openai": OPENAI_MODEL_PRICING,
    "aihumanize": {
        "modes": ["quality", "balance", "enhanced", "private"],
        "plans": AIHUMANIZE_PLANS,
        "unit": "subscription",
    },
}


# ============================================================
# 🧩 Output Path Utilities (NEW flat-folder scheme)
# ============================================================

def _slug(s: str) -> str:
    return _slugify(s)


def build_filenames(company: str, role: str) -> dict[str, str]:
    """
    Standardized final artifact names.

    • Optimized resumes:
        "Sri_{Company}_{Role}.pdf"
    • Humanized resumes:
        "Sri_Kadali_{Company}_{Role}.pdf"
    • Cover letters:
        "Sri_{Company}_{Role}_Cover_Letter.pdf"
    """
    c, r = _slug(company), _slug(role)
    return {
        "optimized":    f"Sri_{c}_{r}.pdf",
        "humanized":    f"Sri_Kadali_{c}_{r}.pdf",
        "cover_letter": f"Sri_{c}_{r}_CoverLetter.pdf",
    }


# --- Final output locations (no per-job subfolders) ---

def get_optimized_pdf_path(company: str, role: str) -> Path:
    names = build_filenames(company, role)
    return (OPTIMIZED_DIR / names["optimized"]).resolve()


def get_humanized_pdf_path(company: str, role: str) -> Path:
    names = build_filenames(company, role)
    return (HUMANIZED_DIR / names["humanized"]).resolve()


def get_coverletter_pdf_path(company: str, role: str) -> Path:
    names = build_filenames(company, role)
    return (COVER_LETTERS_DIR / names["cover_letter"]).resolve()


# --- Backwards-compat sample helpers now map to the same flat scheme ---

def get_sample_resume_pdf_path(company: str, role: str) -> Path:
    return get_optimized_pdf_path(company, role)


def get_sample_humanized_pdf_path(company: str, role: str) -> Path:
    return get_humanized_pdf_path(company, role)


def get_sample_coverletter_pdf_path(company: str, role: str) -> Path:
    return get_coverletter_pdf_path(company, role)


# --- Deprecated: per-job folder builder (left as no-op alias to avoid import errors) ---

def get_job_run_dir(company: str, role: str) -> Path:
    """
    Deprecated: Per-job directories are no longer used.
    Returns OPTIMIZED_DIR for compatibility if any caller still references this.
    """
    return OPTIMIZED_DIR


# ============================================================
# 📊 Diagnostics
# ============================================================

if __name__ == "__main__":
    print("=========== HIREX CONFIG ===========")
    print(f"APP_NAME              : {APP_NAME}")
    print(f"VERSION               : {APP_VERSION}")
    print(f"BASE_DIR              : {BASE_DIR}")
    print(f"OPENAI_API_KEY_LEN    : {len(OPENAI_API_KEY) if OPENAI_API_KEY else 0}")
    print(f"HUMANIZE_API_KEY_LEN  : {len(HUMANIZE_API_KEY) if HUMANIZE_API_KEY else 0}")
    print(f"HUMANIZE_MAIL         : {HUMANIZE_MAIL or 'missing'}")
    print(f"SUPERHUMAN_LOCAL      : {SUPERHUMAN_LOCAL_ENABLED}")
    print(f"HUMANIZE_DEFAULT_ON   : {HUMANIZE_DEFAULT_ON}")
    print(f"HUMANIZE_MODE_DEFAULT : {HUMANIZE_MODE_DEFAULT} (id={AIHUMANIZE_MODE_ID.get(HUMANIZE_MODE_DEFAULT,'?')})")
    print(f"HUMANIZE_ENABLED      : {is_humanize_enabled()}")
    print(f"DEFAULT_MODEL         : {DEFAULT_MODEL}")
    print(f"OPTIMIZED_DIR         : {OPTIMIZED_DIR}")
    print(f"HUMANIZED_DIR         : {HUMANIZED_DIR}")
    print(f"COVER_LETTERS_DIR     : {COVER_LETTERS_DIR}")