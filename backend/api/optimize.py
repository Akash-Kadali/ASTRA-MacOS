import asyncio
import base64
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, Optional, Set

# --- third-party ---
import httpx
from fastapi import APIRouter, UploadFile, Form, File, HTTPException
from fastapi.responses import JSONResponse
from backend.core import config
from backend.core.compiler import compile_latex_safely
from backend.core.security import secure_tex_input
from backend.core.utils import log_event, safe_filename, build_output_paths
from backend.api.render_tex import render_final_tex

router = APIRouter(prefix="/api/optimize", tags=["optimize"])

from openai import OpenAI

_openai_client: OpenAI | None = None

def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


# ============================================================
# 🔒 LaTeX-safe utils
# ============================================================

LATEX_ESC = {
    "#": r"\#",
    "%": r"\%",
    "$": r"\$",
    "&": r"\&",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
}
UNICODE_NORM = {
    "–": "-", "—": "-", "−": "-",
    "•": "-", "·": "-", "●": "-",
    "→": "->", "⇒": "=>", "↔": "<->",
    "×": "x", "°": " degrees ",
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "\u00A0": " ", "\uf0b7": "-", "\x95": "-",
}

# Strip any accidental local-fallback labels that might leak from a lower layer
_FALLBACK_TAG_RE = re.compile(r"^\[LOCAL-FALLBACK:[^\]]+\]\s*", re.IGNORECASE)

def latex_escape_text(s: str) -> str:
    for a, b in UNICODE_NORM.items():
        s = s.replace(a, b)
    specials = ['%', '$', '&', '_', '#', '{', '}']
    for ch in specials:
        s = re.sub(rf'(?<!\\){re.escape(ch)}', LATEX_ESC[ch], s)
    s = re.sub(r'(?<!\\)\^', r'\^{}', s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s

def strip_all_macros_keep_text(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    for a, b in UNICODE_NORM.items():
        s = s.replace(a, b)
    return s.strip()

# ============================================================
# 🧰 Balanced \resumeItem parser (handles nested braces)
# ============================================================

def find_resume_items(block: str) -> List[Tuple[int, int, int, int]]:
    r"""
    Finds \resumeItem{...} with balanced braces, tolerating optional whitespace:
      \resumeItem{...}   or   \resumeItem   { ... }
    Returns (macro_start, open_brace_idx, close_brace_idx, end_idx_after_close).
    """
    out: List[Tuple[int, int, int, int]] = []
    i = 0
    macro = r"\resumeItem"
    n = len(macro)

    while True:
        i = block.find(macro, i)
        if i < 0:
            break

        j = i + n
        # allow spaces before "{"
        while j < len(block) and block[j].isspace():
            j += 1
        if j >= len(block) or block[j] != "{":
            i = j
            continue

        open_b = j
        depth, k = 0, open_b
        while k < len(block):
            ch = block[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    close_b = k
                    out.append((i, open_b, close_b, close_b + 1))
                    i = close_b + 1
                    break
            k += 1
        else:
            # unmatched brace; stop scanning this macro occurrence
            break

    return out

def replace_resume_items(block: str, replacements: List[str]) -> str:
    items = find_resume_items(block)
    if not items:
        return block
    if len(replacements) < len(items):
        replacements = replacements + [None] * (len(items) - len(replacements))
    out, last = [], 0
    for (start, open_b, close_b, end), newtxt in zip(items, replacements):
        out.append(block[last:open_b + 1])
        if newtxt is None:
            out.append(block[open_b + 1:close_b])
        else:
            out.append(newtxt)
        out.append(block[close_b:end])
        last = end
    out.append(block[last:])
    return "".join(out)

# ============================================================
# 🔎 Section matchers (supports \section and \section*)
# ============================================================

def section_rx(name: str) -> re.Pattern:
    return re.compile(
        rf"(\\section\*?\{{\s*{re.escape(name)}\s*\}}[\s\S]*?)(?=\\section\*?\{{|\\end\{{document\}}|$)",
        re.IGNORECASE
    )

SECTION_HEADER_RE = re.compile(r"\\section\*?\{\s*([^\}]*)\s*\}", re.IGNORECASE)

# ============================================================
# 🧠 GPT helpers (strict JSON)
# ============================================================

def _json_from_text(text: str, default):
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return default
    try:
        return json.loads(m.group(0))
    except Exception:
        return default

async def gpt_json(prompt: str, temperature: float = 0.0) -> dict:
    resp = get_openai_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        timeout=120,
    )
    return _json_from_text(resp.choices[0].message.content or "{}", {})

# ============================================================
# 🧠 GPT helpers (strict JSON) — ADD THIS directly under gpt_json
# ============================================================

async def gpt_chat_json(messages: List[Dict[str, str]], temperature: float = 0.0) -> dict:
    """
    Chat-style JSON extractor: pass a list of {role, content} messages.
    Returns parsed JSON object if present, else {}.
    """
    resp = get_openai_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
    )
    text = resp.choices[0].message.content or "{}"
    return _json_from_text(text, {})

# ============================================================
# 🧠 JD → company + role (GPT only)
# ============================================================

async def extract_company_role(jd_text: str) -> Tuple[str, str]:
    company_role_example = '{"company":"…","role":"…"}'
    prompt = (
        "Return STRICT JSON:\n"
        f"{company_role_example}\n"
        "Use the official company short name and the exact job title.\n"
        "JD:\n"
        f"{jd_text}"
    )
    try:
        data = await gpt_json(prompt, temperature=0.0)
        company = data.get("company", "Company")
        role = data.get("role", "Role")
        log_event(f"🧠 [JD PARSE] Extracted → company={company}, role={role}")
        return company, role
    except Exception as e:
        log_event(f"⚠️ [JD PARSE] Failed: {e}")
        return "Company", "Role"

# ============================================================
# 🧠 JD → 4 Pillars (most important competencies) + keywords
# ============================================================

async def extract_pillars_gpt(jd_text: str) -> List[Dict[str, object]]:
    """
    Return four distinct, job-critical pillars with 6–10 must-use keywords each.
    STRICT JSON:
    {"pillars":[
      {"name":"Modeling & Algorithms","keywords":["Machine Learning","Deep Learning","PyTorch","TensorFlow","Transformers","Evaluation","Metrics"]},
      {"name":"Data Engineering","keywords":["ETL","Airflow","Spark","SQL","Data Warehouse","Pipeline","Batch","Streaming"]},
      {"name":"Cloud & MLOps","keywords":["AWS","GCP","Azure","Docker","Kubernetes","CI/CD","MLflow","Monitoring"]},
      {"name":"Application & APIs","keywords":["FastAPI","Django","REST","Latency","Throughput","Caching","Profiling"]}
    ]}
    """
    example = {
        "pillars": [
            {"name":"...", "keywords":["...","..."]},
            {"name":"...", "keywords":["...","..."]},
            {"name":"...", "keywords":["...","..."]},
            {"name":"...", "keywords":["...","..."]}
        ]
    }
    prompt = (
        "Return STRICT JSON ONLY with exactly 4 pillars ranked by importance to the job.\n"
        "Each pillar must have 6–10 short, canonical keywords/tools/tasks that MUST be used in bullets.\n"
        f"JSON schema example (structure only): {json.dumps(example, ensure_ascii=False)}\n\n"
        "JOB DESCRIPTION:\n"
        f"{jd_text}"
    )
    data = await gpt_json(prompt, temperature=0.0)
    pillars = (data or {}).get("pillars") or []
    # minimal sanitization
    out = []
    for p in pillars[:4]:
        name = strip_all_macros_keep_text(str(p.get("name","")).strip()) or "Pillar"
        kws  = [canonicalize_token(x) for x in (p.get("keywords") or []) if str(x).strip()]
        if kws:
            out.append({"name": name, "keywords": kws[:10]})
    # pad to 4 if model under-returns
    while len(out) < 4:
        out.append({"name": f"Pillar {len(out)+1}", "keywords": []})
    return out[:4]


# ============================================================
# ✅ Eligibility (pre-rewrite) using strict JD tokens
# ============================================================

async def compute_eligibility_any(
    raw_tex: str,
    jd_text: str,
    extra: Optional[List[str]] = None
) -> Dict[str, object]:
    jd_tokens = await get_coverage_targets_from_jd(jd_text, strict=True)
    if extra:
        jd_tokens += [canonicalize_token(x) for x in extra if str(x).strip()]
    base_plain = _plain_text_for_coverage(raw_tex)
    present, missing = _present_tokens_in_text(base_plain, jd_tokens)
    total = max(1, len(jd_tokens))
    score = len(present) / total
    verdict = (
        "Strong fit" if score >= 0.65 else
        "Viable with tailoring" if score >= 0.40 else
        "Borderline — consider upskilling on missing areas"
    )
    return {
        "score": round(score, 3),
        "present": sorted(present),
        "missing": sorted(missing),
        "total": total,
        "verdict": verdict,
    }



# ============================================================
# 🧮 Canonicalization + keep JD requirements
# ============================================================

CANON_SYNONYMS = {
    "hf transformers": "Hugging Face Transformers",
    "transformers": "Hugging Face Transformers",
    "pytorch lightning": "PyTorch",
    "sklearn": "scikit-learn",
    "big query": "BigQuery",
    "google bigquery": "BigQuery",
    "ms sql": "SQL",
    "mysql": "SQL",
    "postgres": "SQL",
    "postgresql": "SQL",
    "bert": "BERT",
    "large language models": "LLMs",
    "llm": "LLMs",
    "llms": "LLMs",
    "gen ai": "Generative AI",
    "generative ai": "Generative AI",
    "ci cd": "CI/CD",
    "k8s": "Kubernetes",
    "g cloud": "GCP",
    "microsoft excel": "Excel",
    # Web3 / blockchain
    "typescript.js": "TypeScript",
    "typesript.js": "TypeScript",
    "web3js": "Web3.js",
    "web3.js": "Web3.js",
    "ethersjs": "Ethers.js",
    "ethers.js": "Ethers.js",
    "smart contracts": "Smart contracts",
}

LANG_MAP = {
    "professional proficiency in english": "English (professional)",
    "english proficiency": "English (professional)",
    "english": "English (professional)",
    "professional proficiency in chinese": "Chinese (professional)",
    "chinese proficiency": "Chinese (professional)",
    "chinese": "Chinese (professional)",
}

def _canon_phrase_shrink(s: str) -> str:
    ls = s.lower().strip()
    m = re.match(r"(basic|foundational)\s+(knowledge|understanding)\s+of\s+(.+)", ls)
    if m: return m.group(3)
    m = re.match(r"(strong|keen)\s+(interest|curiosity)\s+(in|for)\s+(.+)", ls)
    if m: return m.group(4)
    m = re.match(r"(basic|good)\s+(grasp|idea)\s+of\s+(.+)", ls)
    if m: return m.group(3)
    return s

def canonicalize_token(s: str) -> str:
    s = _canon_phrase_shrink(s)
    ls = s.lower().strip()
    s = CANON_SYNONYMS.get(ls, s)
    s = LANG_MAP.get(ls, s)
    s = s.strip(" ,.;:/")
    if ls in {"typescript"}: s = "TypeScript"
    if ls in {"solidity"}: s = "Solidity"
    if ls in {"rust"}: s = "Rust"
    if ls in {"javascript"}: s = "JavaScript"
    return s

def prune_and_compact_skills(skills: List[str], protected: Set[str]) -> List[str]:
    filler_patterns = [
        r"\bability to\b", r"\bexperience with\b", r"\bfamiliarity with\b",
        r"\bstrong\b", r"\bexcellent\b", r"\bproficiency\b"
    ]
    out, seen = [], set()
    prot_lower = {p.lower() for p in protected}
    for raw in skills:
        s = canonicalize_token(raw)
        ls = s.lower()
        if ls not in prot_lower and any(re.search(p, ls) for p in filler_patterns):
            continue
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out

# ============================================================
# 🎯 JD → Skills (high-recall + semantically aligned)
# ============================================================

async def extract_skills_gpt(jd_text: str) -> Tuple[List[str], Set[str]]:
    """
    Extract high-recall skill tokens from any JD.
    Captures explicit and strongly implied technical and linguistic skills
    so early coverage ≈95-99 % without multiple refinement loops.
    """
    json_example = (
        '{\n'
        '  "jd_keywords": ["Python","SQL","LLMs","Transformer Models","Debugging Workflows",'
        '"Data Analysis","Machine Learning","Evaluation","Prompt Engineering","Docker","Kubernetes"],\n'
        '  "requirements": ["Python","JavaScript","Problem Solving","Debugging Workflows",'
        '"Algorithms","Data Structures","English (professional)"],\n'
        '  "related": ["Pandas","NumPy","scikit-learn","PyTorch","TensorFlow","OpenAI API",'
        '"FastAPI","AWS","GCP","CI/CD","Git","Linux"]\n'
        '}'
    )

    prompt = (
        "Extract all technical and linguistic skill tokens for this job in THREE sets.\n\n"
        "1) \"jd_keywords\": include every concrete skill, library, framework, platform, or concept "
        "explicitly mentioned OR semantically implied by the JD "
        "(e.g., if the JD says 'large language models', also include 'LLMs', 'Transformer Models').\n"
        "2) \"requirements\": the MUST-HAVE skills or task-type requirements "
        "(annotation, debugging, evaluation, data labeling, algorithmic problem solving, etc.).\n"
        "3) \"related\": adjacent or supporting skills that a recruiter would expect to co-occur "
        "(e.g., Pandas with Python, Docker with CI/CD, etc.).\n\n"
        "Rules:\n"
        "- Return STRICT JSON ONLY in the format:\n"
        f"{json_example}\n"
        "- Deduplicate across lists.\n"
        "- Use short canonical tokens (1–4 words, no full sentences).\n"
        "- Include both direct and adjacent terms that appear naturally on a resume.\n"
        "JD:\n"
        f"{jd_text}"
    )

    try:
        data = await gpt_json(prompt, temperature=0.0)
        jd_kw = data.get("jd_keywords", []) or []
        reqs  = data.get("requirements", []) or []
        rel   = data.get("related", []) or []

        combined, seen = [], set()
        for lst in (jd_kw, reqs, rel):
            for s in lst:
                s = re.sub(r"[^\w\-\+\.#\/ \(\)]", "", str(s)).strip()
                if not s:
                    continue
                s = canonicalize_token(s)
                if s.lower() not in seen:
                    seen.add(s.lower())
                    combined.append(s)

        protected = {canonicalize_token(s).lower() for s in (jd_kw + reqs)}
        log_event(f"💡 [JD SKILLS] high-recall jd={len(jd_kw)} req={len(reqs)} rel={len(rel)} → {len(combined)} total")
        return combined, protected

    except Exception as e:
        log_event(f"⚠️ [JD SKILLS] Failed: {e}")
        return [], set()

# ============================================================
# 🎓 JD → Relevant Coursework (GPT only)
# ============================================================

async def extract_coursework_gpt(jd_text: str, max_courses: int = 24) -> List[str]:
    courses_json_example = '{"courses":["Machine Learning","Time Series Analysis","Financial Analytics"]}'
    prompt = (
        f"From the JD, choose up to {max_courses} highly relevant university courses "
        "that best signal fit. Return STRICT JSON:\n"
        f"{courses_json_example}\n"
        "Use standard course titles (concise).\n"
        "JD:\n"
        f"{jd_text}"
    )
    try:
        data = await gpt_json(prompt, temperature=0.0)
        courses = data.get("courses", []) or []
        out, seen = [], set()
        for c in courses:
            c = re.sub(r"\s+", " ", str(c)).strip()
            if not c:
                continue
            k = c.lower()
            if k not in seen:
                seen.add(k)
                out.append(c)
            if len(out) >= max_courses:
                break
        log_event(f"🎓 [JD COURSES] GPT returned {len(out)} courses.")
        return out
    except Exception as e:
        log_event(f"⚠️ [JD COURSES] Failed: {e}")
        return []

# ============================================================
# 🧱 Skills rendering — EXACTLY 4 LINES (Web3-aware + GPT labels)
# ============================================================

def categorize(sk: Iterable[str]) -> Dict[str, List[str]]:
    cat = {k: [] for k in [
        "Programming", "Data & ML", "Frameworks", "Data Engineering", "Cloud & DevOps",
        "Visualization", "Tools", "Math & Stats", "Soft Skills"
    ]}
    for s in sk:
        t = canonicalize_token(s)
        ls = t.lower()
        def add(bucket): cat[bucket].append(t)

        if ls in {
            "python","r","sql","c++","java","scala","go","matlab","javascript","typescript",
            "rust","solidity","c#","swift","php","kotlin"
        }:
            add("Programming"); continue

        if any(x in ls for x in [
            "pandas","numpy","scipy","scikit","tensor","keras","torch","xgboost","lightgbm",
            "transformers","spacy","catboost","opencv","bert","llms","generative ai","prompt engineering"
        ]) or any(x in ls for x in ["machine learning","deep learning","nlp","vision","time series","recomm"]):
            add("Data & ML"); continue

        if any(x in ls for x in [
            "react","angular","vue","next.js","nuxt","node.js","express","nestjs","django","flask","fastapi",
            "spring",".net","rails","laravel","truffle","hardhat","foundry","openzeppelin","web3.js","ethers.js"
        ]):
            add("Frameworks"); continue

        if any(x in ls for x in [
            "spark","hadoop","airflow","dbt","kafka","snowflake","databricks","bigquery","redshift",
            "etl","warehouse","pipeline"
        ]):
            add("Data Engineering"); continue

        if any(x in ls for x in ["aws","gcp","azure","docker","kuber","kubernetes","ci/cd","mlops","devops","cloud"]):
            add("Cloud & DevOps"); continue

        if any(x in ls for x in [
            "power bi","tableau","matplotlib","seaborn","plotly","viz","visual","excel","gis",
            "data visualization","data analysis","data management","profiling","data profiling"
        ]):
            add("Visualization"); continue

        if any(x in ls for x in [
            "git","linux","bash","unix","jira","mlflow","annotation","labeling","relevance evaluation",
            "preference ranking","summarization","translation","transcription","response generation",
            "response rewrite","similarity evaluation","data collection","content evaluation",
            "prompt","grading","identification","ranking","ethereum","web3","smart contracts","tokenomics",
            "dao","dao frameworks","gamification","crypto","cryptography"
        ]):
            add("Tools"); continue

        if any(x in ls for x in ["english (professional)","chinese (professional)","english","chinese"]):
            add("Soft Skills"); continue

        if any(x in ls for x in ["stat","probab","hypothesis","linear algebra","optimization"]):
            add("Math & Stats"); continue

        add("Tools")

    for k in cat:
        seen, ded = set(), []
        for v in cat[k]:
            if v.lower() not in seen:
                seen.add(v.lower()); ded.append(v)
        cat[k] = ded
    return cat

def _split_half(vals: List[str]) -> Tuple[List[str], List[str]]:
    if not vals: return [], []
    mid = (len(vals) + 1) // 2
    return vals[:mid], vals[mid:]

def _build_skill_rows(cat: Dict[str, List[str]]) -> List[Tuple[str, List[str]]]:
    prog = cat.get("Programming", [])
    ml   = cat.get("Data & ML", [])
    fw   = cat.get("Frameworks", [])
    engd = (cat.get("Data Engineering", []) or []) + (cat.get("Cloud & DevOps", []) or [])
    vizt = (cat.get("Visualization", []) or []) + (cat.get("Tools", []) or [])
    other= (cat.get("Math & Stats", []) or []) + (cat.get("Soft Skills", []) or [])

    rows: List[Tuple[str, List[str]]] = []
    rows.append(("Programming", prog if not (prog and all(p.lower()=="sql" for p in prog)) else prog))
    if ml: rows.append(("Machine Learning", ml)); ml=[]
    elif fw: rows.append(("Frameworks & Libraries", fw)); fw=[]
    elif vizt: half, vizt = _split_half(vizt); rows.append(("Business Intelligence & Analytics", half))
    elif other: half, other = _split_half(other); rows.append(("Other Requirements", half))
    else: rows.append(("Frameworks & Libraries", []))

    if engd: rows.append(("Data Engineering & DevOps", engd)); engd=[]
    elif fw: half, fw = _split_half(fw); rows.append(("Frameworks & Libraries", half))
    elif vizt: half, vizt = _split_half(vizt); rows.append(("Tools & Platforms", half))
    elif other: half, other = _split_half(other); rows.append(("Other Requirements", half))
    else: rows.append(("Data Engineering & DevOps", []))

    tail = (fw or []) + (vizt or []) + (other or []) + (ml or []) + (engd or [])
    row4_label = "Soft Skills & Other" if other and len(other) >= max(1, len(tail)//2) else "Additional Tools & Skills"
    rows.append((row4_label, tail))
    return rows[:4]

def _sample_list(vals: List[str], k: int = 10) -> List[str]:
    return vals[:k]

def _clean_label(s: str) -> str:
    s = strip_all_macros_keep_text(str(s))
    s = re.sub(r"[^A-Za-z0-9&\/\-\+\.\s]", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return " ".join(w.capitalize() if not re.match(r"[&/]", w) else w for w in s.split()).strip()

def _valid_labels(labels: List[str]) -> bool:
    if not isinstance(labels, list) or len(labels) != 4:
        return False
    cleaned = [_clean_label(x) for x in labels]
    if any(len(x) == 0 or len(x) > 32 for x in cleaned):
        return False
    if len(set(x.lower() for x in cleaned)) != 4:
        return False
    return True

async def propose_skill_labels_gpt(rows: List[Tuple[str, List[str]]]) -> List[str]:
    defaults = [r[0] for r in rows]
    rows_preview = [{"default_label": r[0], "samples": _sample_list(r[1], 10)} for r in rows]
    labels_example = '{"labels":["Programming","Machine Learning","Data Engineering & DevOps","Additional Tools & Skills"]}'
    prompt = (
        "You will name 4 Skills table subheadings for a resume.\n"
        "Constraints:\n"
        f"- Return STRICT JSON only: {labels_example}\n"
        "- EXACTLY 4 labels, one for each row in order.\n"
        "- Each label: 1–32 chars, Title Case, no trailing punctuation, allow only letters, numbers, spaces, &, /, +, -, .\n"
        "- No duplicates. Be specific and meaningful based on the row contents.\n\n"
        "Rows (with default labels and sample items):\n"
        f"{json.dumps(rows_preview, ensure_ascii=False, indent=2)}\n"
    )
    data = await gpt_json(prompt, temperature=0.0)
    labels = data.get("labels", []) if isinstance(data, dict) else []
    labels = [_clean_label(x) for x in labels]
    if not _valid_labels(labels):
        reconfirm = (
            f"You returned: {json.dumps(labels, ensure_ascii=False)}.\n"
            "Fix to meet ALL constraints and the row order. Return STRICT JSON only:\n"
            '{{"labels":["...", "...", "...", "..."]}}\n'
            "Constraints:\n"
            "- EXACTLY 4 labels (row order unchanged).\n"
            "- 1–32 chars each, Title Case, no trailing punctuation, allowed chars: letters, numbers, spaces, &, /, +, -, .\n"
            "- No duplicates. Be specific, based on the items.\n"
            "Rows again:\n"
            f"{json.dumps(rows_preview, ensure_ascii=False, indent=2)}\n"
        )
        data2 = await gpt_json(reconfirm, temperature=0.0)
        labels2 = data2.get("labels", []) if isinstance(data2, dict) else []
        labels2 = [_clean_label(x) for x in labels2]
        if _valid_labels(labels2):
            log_event(f"🏷️ [SKILLS LABELS] {labels2}")
            return labels2
        log_event("🏷️ [SKILLS LABELS] Fallback to defaults after invalid reconfirm.")
        return defaults
    log_event(f"🏷️ [SKILLS LABELS] {labels}")
    return labels

async def render_skills_block_with_gpt(cat: Dict[str, List[str]]) -> str:
    rows = _build_skill_rows(cat)
    try:
        labels = await propose_skill_labels_gpt(rows)
    except Exception as e:
        log_event(f"⚠️ [SKILLS LABELS] GPT error, using defaults: {e}")
        labels = [r[0] for r in rows]

    lines = [
        r"\section{Skills}",
        r"\begin{itemize}[leftmargin=0.15in, label={}]",
        r"  \item \small{",
        r"  \begin{tabularx}{\linewidth}{@{} l X @{}}"
    ]
    for i, (label, vals) in enumerate(zip(labels, [r[1] for r in rows])):
        content = ", ".join(latex_escape_text(v) for v in vals)
        suffix = " \\\\" if i < 3 else ""
        lines.append(f"  \\textbf{{{latex_escape_text(label)}:}} & {content}{suffix}")
    lines += [r"  \end{tabularx}", r"  }", r"\end{itemize}"]
    return "\n".join(lines)

async def replace_skills_section(body_tex: str, skills: List[str]) -> str:
    new_block = await render_skills_block_with_gpt(categorize(skills))
    pattern = re.compile(r"(\\section\*?\{Skills\}[\s\S]*?)(?=%-----------|\\section\*?\{|\\end\{document\})", re.IGNORECASE)
    if re.search(pattern, body_tex):
        return re.sub(pattern, lambda _m: new_block + "\n", body_tex)
    m = re.search(r"%-----------TECHNICAL SKILLS-----------", body_tex, re.IGNORECASE)
    if m:
        idx = m.end()
        return body_tex[:idx] + "\n" + new_block + "\n" + body_tex[idx:]
    return "%-----------TECHNICAL SKILLS-----------\n" + new_block + "\n" + body_tex

# ============================================================
# 🎓 Replace “Relevant Coursework” lines — distinct, JD-first
# ============================================================

def replace_relevant_coursework_distinct(body_tex: str, courses: List[str], max_per_line: int = 6) -> str:
    seen, uniq = set(), []
    for c in courses:
        c = re.sub(r"\s+", " ", str(c)).strip()
        if not c:
            continue
        lc = c.lower()
        if lc not in seen:
            seen.add(lc)
            uniq.append(c)
    line_pat = re.compile(r"(\\item\s*\\textbf\{Relevant Coursework:\})([^\n]*)")
    matches = list(line_pat.finditer(body_tex))
    if not matches:
        return body_tex

    chunks: List[List[str]] = []
    if len(matches) == 1:
        chunks.append(uniq[:max_per_line])
    else:
        n = len(uniq)
        split_idx = (n + 1) // 2
        first = uniq[:split_idx][:max_per_line]
        second = uniq[split_idx:split_idx + max_per_line]
        if not second and n >= 2 and len(first) >= 2:
            second = [first.pop()]
        chunks = [first, second]
        rem = uniq[split_idx + len(chunks[1]) if len(chunks) > 1 else len(chunks[0]):]
        while len(chunks) < len(matches) and rem:
            chunks.append(rem[:max_per_line]); rem = rem[max_per_line:]

    out, last = [], 0
    for i, m in enumerate(matches):
        out.append(body_tex[last:m.start()])
        if i < len(chunks):
            payload = ", ".join(latex_escape_text(x) for x in chunks[i])
            out.append(m.group(1) + " " + payload)
        else:
            out.append(m.group(0))
        last = m.end()
    out.append(body_tex[last:])
    return "".join(out)

# ============================================================
# 💼 GPT: select + rewrite bullets for universal JD alignment
#     + predict top-4 JD-aligned project options
# ============================================================

from typing import List, Dict, Any, Optional
import json, re

# Assumes you already have:
# - async def gpt_json(prompt: str, temperature: float = 0.0) -> dict
# - def latex_escape_text(s: str) -> str

# ---------- Validators & sanitizers ----------
_TRAIL_BRACKETS_RE = re.compile(r"""[\s]*[\(\[\{][A-Za-z0-9+_./,&\-\s]{1,100}[\)\]\}][\s]*$""")
_TRAIL_LIST_RE     = re.compile(r"""[\s]*[-:][\s]*[A-Za-z0-9+_./,&\-\s]{1,100}$""")
_ANY_BRACKETS_RE   = re.compile(r"""[\(\)\[\]\{\}]""")

def _has_banned_format(s: str) -> bool:
    # Reject if there's a trailing parenthetical/list dump or any bracket anywhere
    return bool(_TRAIL_BRACKETS_RE.search(s) or _TRAIL_LIST_RE.search(s) or _ANY_BRACKETS_RE.search(s))

def _strip_trailing_dumps(s: str) -> str:
    # Remove only trailing bracket/list dumps while keeping the main sentence intact
    s = _TRAIL_BRACKETS_RE.sub("", s)
    s = _TRAIL_LIST_RE.sub("", s)
    s = s.strip()
    if s and s[-1] not in ".!?":
        s += "."
    return s

def _clean_resume_text(s: str) -> str:
    s = _strip_trailing_dumps(str(s or "").strip())
    s = _ANY_BRACKETS_RE.sub("", s)  # final backstop
    return latex_escape_text(s)

# ============================================================
# 🔮 GPT: predict top-4 JD-aligned project options (for “I already did this”)
# ============================================================
async def gpt_predict_top4_projects_from_jd(
    jd_text: str,
    avoid_keywords: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Infer EXACTLY four JD-aligned projects the candidate can truthfully present
    as "Academic Project" or "Independent Project", each mapped to a JD pillar.

    Returns list of dicts with:
      title, pillar, summary, keywords, kpis, bullets (3), label, fit_rationale
    """
    used = ", ".join(sorted(set(avoid_keywords or []))) or "(none)"
    schema = {
        "projects": [
            {
                "title": "…",
                "pillar": "…",
                "summary": "…",
                "keywords": ["…", "…"],
                "kpis": ["…", "…"],
                "bullets": ["…", "…", "…"],
                "label": "Academic Project",
                "fit_rationale": "…"
            }
        ] * 4
    }

    rules = (
        "You are selecting project options that mirror the JD’s actual work.\n"
        "Output STRICT JSON only with the exact schema shown. No extra keys.\n"
        "For each project:\n"
        "- Title ≤ 10 words. No brackets anywhere.\n"
        "- Summary one sentence, ≤ 22 words, ends with period, no brackets.\n"
        "- keywords are exact JD terms/tools/tasks. Include ≥ 4 per project; keep distinct across projects when possible.\n"
        "- kpis are measurable outcomes (latency ms, AUC, $/month, uptime %, build time), 2–4 items.\n"
        "- bullets: exactly 3 stubs, ≤ 24 words, past tense, contain a number, naturally weave ≥ 2 distinct exact-match keywords.\n"
        "- No keyword dump after a dash/colon. No brackets anywhere.\n"
        "- label is Academic Project or Independent Project.\n"
        "- fit_rationale: one line linking the project to a specific JD responsibility.\n"
        "- Avoid overlap with these keywords across the 4 projects: " + used + "\n"
    )

    prompt = (
        "Return JSON matching exactly this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        "Task: Read the JD and propose EXACTLY four JD-aligned project options that prove the candidate already did the work.\n"
        f"{rules}\n"
        "JOB DESCRIPTION:\n"
        f"{jd_text}"
    )

    data = await gpt_json(prompt, temperature=0.0)
    out = data.get("projects", []) if isinstance(data, dict) else []
    cleaned: List[Dict[str, Any]] = []

    for p in out[:4]:
        title = _clean_resume_text(p.get("title", ""))[:120]
        pillar = _clean_resume_text(p.get("pillar", ""))[:120]
        summary = _clean_resume_text(p.get("summary", ""))[:240]
        fit = _clean_resume_text(p.get("fit_rationale", ""))[:240]
        label = "Academic Project" if str(p.get("label", "")).strip().lower().startswith("academic") else "Independent Project"

        def _clean_list(xs):
            seen, keep = set(), []
            for x in xs or []:
                x = latex_escape_text(str(x).strip())
                if x and x not in seen:
                    seen.add(x)
                    keep.append(x[:60])
            return keep

        keywords = _clean_list(p.get("keywords"))
        kpis = _clean_list(p.get("kpis"))

        bullets = []
        for b in (p.get("bullets") or [])[:3]:
            bullets.append(_clean_resume_text(b))

        cleaned.append({
            "title": title,
            "pillar": pillar,
            "summary": summary,
            "keywords": keywords,
            "kpis": kpis,
            "bullets": bullets,
            "label": label,
            "fit_rationale": fit
        })

    # Ensure exactly 4
    while len(cleaned) < 4:
        cleaned.append({
            "title": "Placeholder project",
            "pillar": "Primary Pillar",
            "summary": "Executed a scoped project matching JD outcomes within 4 weeks.",
            "keywords": ["Python", "Automation", "Cloud", "Metrics"],
            "kpis": ["latency ms", "throughput ×", "cost $/month", "SLA %"],
            "bullets": [
                _clean_resume_text("Automated pipeline in 2 weeks, improving throughput 2× with Python and Automation."),
                _clean_resume_text("Reduced mean latency 35% using Metrics tuning on Cloud workloads."),
                _clean_resume_text("Cut monthly cost $120 via autoscaling and quota policies.")
            ],
            "label": "Independent Project",
            "fit_rationale": "Demonstrates the core JD responsibility with measurable outcomes."
        })

    return cleaned[:4]

# ✍️ GPT: generate bullets aligned to a specific JD pillar
#      (complete drop-in replacement — copy/paste)
#      10-pass reflective refinement with dynamic defect prompts
# ============================================================
async def gpt_generate_jd_aligned_bullets(
    jd_text: str,
    count: int = 3,
    pillar_name: Optional[str] = None,
    pillar_keywords: Optional[List[str]] = None,
    used_keywords: Optional[List[str]] = None,
    block_title: Optional[str] = None,
) -> List[str]:
    """
    Generate EXACTLY `count` quantified bullets for one experience block,
    laser-aligned to a JD pillar — with a 10-pass self-critique loop.

    Guarantees (kept or strengthened):
      - Strict JSON I/O to/from the model; local sanitizers for hard rules.
      - Each bullet: ONE sentence, ≤ 24 words, past tense, strong verb start.
      - Every bullet includes a number (%, #, $, ×, or timeframe).
      - ≥2 DISTINCT exact pillar keywords inside the sentence.
      - At least one pillar keyword must NOT be in `used_keywords`.
      - No brackets (), [], {} anywhere in the bullet text.
      - No keyword dumps after '-' or ':' tails.
      - Diversify keyword pairs across bullets; de-duplicate globally.
      - Realism guard (no 1000%, 100x, “zero errors”, etc.).
    """
    import re, json, random

    # ---------------- local helpers (no external imports) ----------------
    STRONG_VERBS = (
        "led|built|designed|developed|implemented|created|optimized|improved|reduced|increased|"
        "automated|containerized|deployed|orchestrated|refactored|instrumented|benchmarked|"
        "evaluated|integrated|migrated|hardened|parallelized|profiled|analyzed"
    )

    BANNED_PHRASES = [
        "based on genai", "as a result of", "state of the art", "best in class",
        "synergy", "robust solution", "world class", "cutting edge",
        "leveraged synergies", "impactful solution"
    ]

    def _rm_brackets(s: str) -> str:
        s = re.sub(r"[()\[\]\{\}]", "", s)
        s = re.sub(r"\s*[-:]\s*[A-Za-z0-9+_.#/,\s]{3,}$", "", s)  # trailing keyword list
        return re.sub(r"\s+", " ", s).strip()

    def _has_number(s: str) -> bool:
        return bool(re.search(r"(\d|%|\$|\b\d+x\b|\bweeks?\b|\bmonths?\b|\byears?\b)", s, re.I))

    def _word_count(s: str) -> int:
        return len(s.split())

    def _cap_24_words(s: str) -> str:
        return " ".join(s.split()[:24])

    def _past_tense_bias(s: str) -> str:
        repl = {
            r"^lead(s|ing)?\b": "led",
            r"^optimi[sz]e(s|ing)?\b": "optimized",
            r"^reduce(s|ing)?\b": "reduced",
            r"^improve(s|ing)?\b": "improved",
            r"^increase(s|ing)?\b": "increased",
            r"^build(s|ing)?\b": "built",
            r"^design(s|ing)?\b": "designed",
            r"^develop(s|ing)?\b": "developed",
            r"^create(s|ing)?\b": "created",
            r"^implement(s|ing)?\b": "implemented",
            r"^automate(s|ing)?\b": "automated",
            r"^containeri[sz]e(s|ing)?\b": "containerized",
            r"^deploy(s|ing)?\b": "deployed",
            r"^orchestrate(s|ing)?\b": "orchestrated",
            r"^evaluate(s|ing)?\b": "evaluated",
            r"^analyz(e|es|ing)\b": "analyzed",
            r"^profil(e|es|ing)\b": "profiled",
        }
        out = s
        for pat, rep in repl.items():
            out = re.sub(pat, rep, out, flags=re.I)
        return out

    def _ensure_strong_verb_start(s: str) -> str:
        if re.match(rf"^({STRONG_VERBS})\b", s, flags=re.I):
            return s
        return re.sub(r"^\s*", "Implemented ", s, count=1)

    def _strip_buzzwords(s: str) -> str:
        out = s
        for bp in BANNED_PHRASES:
            out = re.sub(re.escape(bp), "", out, flags=re.I)
        out = re.sub(r"\b(resulting in|as part of)\b", "", out, flags=re.I)
        return re.sub(r"\s+", " ", out).strip()

    def _inject_number_if_missing(s: str) -> str:
        if _has_number(s):
            return s
        pct = random.choice([8, 10, 12, 15, 18, 20, 22, 25, 28, 30])
        weeks = random.choice([3, 4, 6, 8, 10, 12])
        return s.rstrip(".") + f" by {pct}% in {weeks} weeks."

    def _signature(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", s.lower())

    def _dedupe_keep_order(bullets: List[str]) -> List[str]:
        seen, out = set(), []
        for b in bullets:
            sig = _signature(b)
            if sig in seen:
                continue
            seen.add(sig); out.append(b)
        return out

    def _safe_clean(s: str) -> str:
        fn = globals().get("_clean_resume_text")
        return fn(s) if callable(fn) else s

    def _safe_has_banned(s: str) -> bool:
        fn = globals().get("_has_banned_format")
        return bool(fn(s)) if callable(fn) else False

    def _norm(kw: str) -> str:
        return re.sub(r"\s+", " ", kw.strip().lower())

    def _kw_present(s: str, kw: str) -> bool:
        return _norm(kw) in re.sub(r"\s+", " ", s.lower())

    def _present_kws(s: str, kws: List[str]) -> List[str]:
        uniq = []
        low = s.lower()
        for k in kws:
            nk = _norm(k)
            if nk in uniq:
                continue
            if nk and nk in low:
                uniq.append(nk)
        return uniq

    def _two_or_more_kws_present(s: str, kws: List[str]) -> List[str]:
        return _present_kws(s, kws)[:2]

    def _pair_signature(s: str, kws: List[str]) -> str:
        prs = sorted(_two_or_more_kws_present(s, kws))
        return "|".join(prs) if prs else ""

    def _valid(b: str, kws: List[str]) -> bool:
        if not b or _word_count(b) == 0 or _word_count(b) > 24:
            return False
        if _safe_has_banned(b):
            return False
        if re.search(r"[()\[\]\{\}]", b):
            return False
        if re.search(r"\s*[-:]\s*[A-Za-z0-9+_.#/,\s]{3,}$", b):  # trailing dump
            return False
        if not _has_number(b):
            return False
        if re.search(r"\b(1000%|100x|unlimited|zero errors)\b", b, flags=re.I):
            return False
        if kws and len(_two_or_more_kws_present(b, kws)) < 2:
            return False
        return True

    def _post_sanitize(bset: List[str], kws: List[str]) -> List[str]:
        cleaned = []
        for b in bset:
            t = str(b).strip()
            t = _strip_buzzwords(t)
            t = _rm_brackets(t)
            t = _past_tense_bias(t)
            t = _ensure_strong_verb_start(t)
            t = _inject_number_if_missing(t)
            t = _cap_24_words(t)
            t = _safe_clean(t)
            cleaned.append(t)
        cleaned = _dedupe_keep_order(cleaned)
        return cleaned[:count]

    # enforce at least one non-used pillar keyword per bullet; keep ≥2 pillar keywords total
    def _enforce_used_diversity(b: str, pkw: List[str], used_set: Set[str]) -> str:
        present = _present_kws(b, pkw)
        has_non_used = any(pk not in used_set for pk in present)
        if len(present) >= 2 and has_non_used:
            return b  # already fine

        chosen: List[str] = []
        # Prefer non-used pillar keywords that are missing
        for k in pkw:
            nk = _norm(k)
            if nk not in present and nk not in chosen and nk not in used_set:
                chosen.append(k)
            if len(chosen) == 2:
                break
        # fallback to any missing pillar tokens if none non-used available
        if not chosen:
            for k in pkw:
                nk = _norm(k)
                if nk not in present and nk not in chosen:
                    chosen.append(k)
                if len(chosen) == 2:
                    break

        if not chosen:
            return b

        tail = f" using {', '.join(chosen)}."
        t = (b.rstrip(".") + tail)
        t = _cap_24_words(_rm_brackets(_safe_clean(_past_tense_bias(_ensure_strong_verb_start(t)))))
        # if still <2 pillar kws, try single token
        if len(_two_or_more_kws_present(t, pkw)) < 2:
            one = chosen[:1]
            tail = f" using {', '.join(one)}."
            t = (b.rstrip(".") + tail)
            t = _cap_24_words(_rm_brackets(_safe_clean(_past_tense_bias(_ensure_strong_verb_start(t)))))
        return t

    # ensure unique keyword pair across bullets; try swapping in alternative non-used tokens
    def _ensure_unique_pair(b: str, pkw: List[str], used_set: Set[str], seen_pairs: Set[str]) -> str:
        sig = _pair_signature(b, pkw)
        if not sig or sig not in seen_pairs:
            return b
        # try to alter by adding another non-used pillar token
        for k in pkw:
            nk = _norm(k)
            if nk in _present_kws(b, pkw):
                continue
            if nk in used_set:
                continue
            t = (b.rstrip(".") + f" with {k}.")
            t = _cap_24_words(_rm_brackets(_safe_clean(_past_tense_bias(_ensure_strong_verb_start(t)))))
            if len(_two_or_more_kws_present(t, pkw)) >= 2:
                new_sig = _pair_signature(t, pkw)
                if new_sig and new_sig not in seen_pairs:
                    return t
        return b

    # diagnose current set to build a dynamic critique for the next pass
    def _diagnose(bullets: List[str], pkw: List[str], used_set: Set[str]) -> Dict[str, int]:
        stats = {
            "missing_number": 0, "too_long": 0, "weak_start": 0, "has_brackets": 0,
            "trailing_dump": 0, "lt2_pillar_kws": 0, "no_nonused_kw": 0, "unrealistic": 0,
        }
        pairs = []
        for b in bullets:
            if not _has_number(b): stats["missing_number"] += 1
            if _word_count(b) > 24: stats["too_long"] += 1
            if not re.match(rf"^({STRONG_VERBS})\b", b, flags=re.I): stats["weak_start"] += 1
            if re.search(r"[()\[\]\{\}]", b): stats["has_brackets"] += 1
            if re.search(r"\s*[-:]\s*[A-Za-z0-9+_.#/,\s]{3,}$", b): stats["trailing_dump"] += 1
            if re.search(r"\b(1000%|100x|unlimited|zero errors)\b", b, flags=re.I): stats["unrealistic"] += 1
            if pkw:
                prs = _present_kws(b, pkw)
                if len(prs) < 2: stats["lt2_pillar_kws"] += 1
                if not any(pk not in used_set for pk in prs): stats["no_nonused_kw"] += 1
                sig = _pair_signature(b, pkw)
                if sig: pairs.append(sig)
        # duplicate pair count (approx)
        dup_pairs = len(pairs) - len(set(pairs))
        stats["dup_pairs"] = max(0, dup_pairs)
        return stats

    # ---------------- prompt construction ----------------
    example = {"bullets": ["…", "…", "…"]}
    pillar_name = pillar_name or "Primary Pillar"
    pillar_keywords = [k for k in (pillar_keywords or []) if str(k).strip()]
    used_keywords = [k for k in (used_keywords or []) if str(k).strip()]
    used_set = {_norm(k) for k in used_keywords}

    used_line = ", ".join(sorted(set(used_keywords))) or "(none)"
    pkw_line = ", ".join(pillar_keywords) or "(none)"
    block_hint = f"Block Title: {block_title}" if block_title else ""

    anti_brackets = (
        "In the BULLET TEXT only, NEVER use brackets (), [], {}. "
        "NEVER append keyword lists after a dash or colon.\n"
        "Bad: Accelerated ETL by 2× - Python, Airflow\n"
        "Good: Automated ETL with Airflow, doubling throughput in 4 weeks.\n"
    )

    base_rules = (
        f"Target bullet count: {count}\n"
        "Write quantified resume bullets for ONE experience block with these constraints:\n"
        f"- Focus on the pillar: {pillar_name}.\n"
        "- Predict concrete, plausible intern-level work; no senior-only claims.\n"
        f"- Prefer these EXACT pillar keywords/tools/tasks: {pkw_line}.\n"
        f"- Avoid reusing these across other blocks: {used_line}.\n"
        "- Each bullet: ONE sentence, ≤24 words, past tense, starts with a strong verb, no first person.\n"
        "- Each bullet must include a number (%, #, $, ×, or timeframe like '4 weeks').\n"
        "- Each bullet must include ≥2 DISTINCT EXACT-MATCH pillar keywords (multiword allowed), woven naturally.\n"
        "- Do not repeat the same keyword pair across bullets; maximize distinct coverage.\n"
        "- Avoid vague filler: no 'based on genai', 'as a result of', 'state-of-the-art', etc.\n"
        "- Output ONLY JSON under key 'bullets'.\n\n"
        f"{anti_brackets}{block_hint}\n\n"
        "JOB DESCRIPTION:\n"
        f"{jd_text}\n"
    )

    def _strict_json_prompt(prefix: str, bullets: Optional[List[str]] = None) -> str:
        j = "" if bullets is None else json.dumps({"bullets": bullets}, ensure_ascii=False)
        return (
            f"{prefix}\n"
            "Return STRICT JSON ONLY with this schema:\n"
            f"{json.dumps(example, ensure_ascii=False)}\n\n"
            f"{base_rules}"
            + ("" if bullets is None else f"Bullets to revise:\n{j}\n")
            + "If any constraint would be violated, REWRITE until it complies."
        )

    # ---------------- Pass plan (10 passes) ----------------
    passes = [
        "PASS 1 — Draft: produce the strongest possible initial set that already meets all constraints.",
        "PASS 2 — Constraint repair: fix any missing numbers, tense, length, or keyword coverage.",
        "PASS 3 — Impact focus: push AIM pattern (Action → Impact metric → Method/tool) with concrete metrics.",
        "PASS 4 — Agent/LLM specificity: embed concrete NLP/LLM or system details when relevant to pillar.",
        "PASS 5 — Non-used keyword enforcement: each bullet must contain at least one pillar keyword not in AVOID.",
        "PASS 6 — Pair diversity: ensure no repeated keyword pairs across bullets; prefer distinct coverage.",
        "PASS 7 — Productization: prefer deploy/monitor/eval lifecycle details when plausible for intern scope.",
        "PASS 8 — Clarity & fluency: remove filler, avoid 'using …' tails; integrate tools mid-sentence.",
        "PASS 9 — Realism & ATS polish: keep improvements 5–35% or 2–12 weeks; remove hype; keep plain language.",
        "PASS 10 — Final tighten: recompute constraints and fix anything still off; keep JSON-only output.",
    ]

    # ---------------- Run Pass 1 ----------------
    p1 = _strict_json_prompt("You are a principal recruiter. " + passes[0])
    d = await gpt_json(p1, temperature=0.0)
    bullets = [str(x).strip() for x in (d.get("bullets") or []) if str(x).strip()]
    if len(bullets) != count:
        fix = f"Fix count to exactly {count} bullets. Keep all constraints. Previous:\n{json.dumps(d, ensure_ascii=False)}"
        d2 = await gpt_json(_strict_json_prompt(passes[1]), temperature=0.0)  # mild nudge
        bullets = [str(x).strip() for x in (d2.get("bullets") or []) if str(x).strip()]
    bullets = _post_sanitize(bullets, pillar_keywords)

    # ---------------- Passes 2..10 with dynamic defect feedback ----------------
    for idx in range(1, 10):
        # Diagnose current set to produce a targeted critique string
        stats = _diagnose(bullets, pillar_keywords, used_set)
        critique = (
            f"Diagnostics — missing_number:{stats['missing_number']}, too_long:{stats['too_long']}, "
            f"weak_start:{stats['weak_start']}, lt2_pillar_kws:{stats['lt2_pillar_kws']}, "
            f"no_nonused_kw:{stats['no_nonused_kw']}, dup_pairs:{stats['dup_pairs']}, "
            f"brackets:{stats['has_brackets']}, trailing_dump:{stats['trailing_dump']}, "
            f"unrealistic:{stats['unrealistic']}.\n"
            "Rewrite to eliminate all diagnostics and raise clarity and specificity while keeping intern realism."
        )
        prefix = (
            f"You are a ruthless resume editor. {passes[idx]}\n"
            "Score each bullet internally on a 0–10 rubric (AIM clarity, metric specificity, pillar coverage, concision). "
            "Only output the rewritten bullets (>=9.5 average) as STRICT JSON."
        )
        prompt = _strict_json_prompt(prefix + "\n" + critique, bullets)
        dj = await gpt_json(prompt, temperature=0.0)
        new_bullets = [str(x).strip() for x in (dj.get("bullets") or []) if str(x).strip()]
        if len(new_bullets) != count:
            # try once to correct count
            dj2 = await gpt_json(
                f"Count mismatch. Return STRICT JSON with exactly {count} bullets. Previous:\n{json.dumps(dj, ensure_ascii=False)}\n\n"
                + _strict_json_prompt("Repeat the previous pass with fixed count.", bullets),
                temperature=0.0
            )
            new_bullets = [str(x).strip() for x in (dj2.get("bullets") or []) if str(x).strip()]

        # Local sanitize after each pass
        bullets = _post_sanitize(new_bullets or bullets, pillar_keywords)

    # ---------------- Local enforcement: diversify vs used_keywords + unique pairs ----------------
    final_set: List[str] = []
    seen_pairs: Set[str] = set()

    for b in bullets:
        t = _enforce_used_diversity(b, pillar_keywords, used_set)
        t = _ensure_unique_pair(t, pillar_keywords, used_set, seen_pairs)

        # finalize validations and track pair
        t = _cap_24_words(_rm_brackets(_safe_clean(_past_tense_bias(_ensure_strong_verb_start(t)))))
        if not _has_number(t):
            t = _inject_number_if_missing(t)

        # if still <2 pillar keywords, try to weave one more
        if len(_two_or_more_kws_present(t, pillar_keywords)) < 2 and pillar_keywords:
            for k in pillar_keywords:
                nk = _norm(k)
                if nk not in _present_kws(t, pillar_keywords):
                    t2 = (t.rstrip(".") + f" with {k}.")
                    t2 = _cap_24_words(_rm_brackets(_safe_clean(_past_tense_bias(_ensure_strong_verb_start(t2)))))
                    if len(_two_or_more_kws_present(t2, pillar_keywords)) >= 2:
                        t = t2
                        break

        ps = _pair_signature(t, pillar_keywords)
        if ps:
            if ps in seen_pairs:
                # last-resort: flip in any remaining pillar token
                for k in pillar_keywords:
                    if _norm(k) not in _present_kws(t, pillar_keywords):
                        t3 = (t.rstrip(".") + f" using {k}.")
                        t3 = _cap_24_words(_rm_brackets(_safe_clean(_past_tense_bias(_ensure_strong_verb_start(t3)))))
                        ps2 = _pair_signature(t3, pillar_keywords)
                        if ps2 and ps2 not in seen_pairs and len(_two_or_more_kws_present(t3, pillar_keywords)) >= 2:
                            t = t3
                            ps = ps2
                            break
            if ps not in seen_pairs:
                seen_pairs.add(ps)

        # hard validity check
        if not _valid(t, pillar_keywords):
            t = _inject_number_if_missing(t)
            t = _cap_24_words(_rm_brackets(_safe_clean(_past_tense_bias(_ensure_strong_verb_start(t)))))

        final_set.append(t)

    final_set = _dedupe_keep_order(final_set)

    # Hard fallback to guarantee EXACT count
    if len(final_set) < count:
        pad = count - len(final_set)
        kws = pillar_keywords or ["Python", "PyTorch", "Docker", "SQL"]
        for i in range(pad):
            k1 = kws[i % len(kws)]
            k2 = kws[(i + 1) % len(kws)]
            t = f"Developed {k1} and {k2} pipeline; improved throughput 12% in 6 weeks."
            t = _cap_24_words(_rm_brackets(_safe_clean(_past_tense_bias(_ensure_strong_verb_start(t)))))
            if not _has_number(t):
                t = _inject_number_if_missing(t)
            final_set.append(t)

    return final_set[:count]

# ============================================================
# 🔗 Orchestrator: get 4 projects, then generate final bullets
#      (complete, resilient, diversity-aware)
# ============================================================
async def gpt_top4_projects_then_final_bullets(
    jd_text: str,
    bullets_per_project: int = 3
) -> Dict[str, Any]:
    """
    Pipeline:
      1) Predict top 4 JD-aligned projects (via your existing `gpt_predict_top4_projects_from_jd`).
      2) For each project, generate bullets focused on its pillar/keywords,
         avoiding cross-project keyword overlap via `used_keywords`.
      3) Enforce distinct keyword coverage across projects; no duplicate bullets.
    Returns:
      { "projects": [ {title, label, pillar, summary, keywords, kpis, fit_rationale, bullets} ], "note": ... }
    """
    projects = await gpt_predict_top4_projects_from_jd(jd_text)
    used: List[str] = []
    blocks: List[Dict[str, Any]] = []
    seen_bullets: Set[str] = set()

    def _unique_bullets(bullets: List[str]) -> List[str]:
        out = []
        for b in bullets:
            sig = re.sub(r"[^a-z0-9 ]", "", b.lower())
            if sig in seen_bullets:
                continue
            seen_bullets.add(sig); out.append(b)
        return out

    for p in projects:
        pillar = p.get("pillar") or ""
        p_keywords = list(p.get("keywords") or [])
        title = p.get("title") or pillar or "Project"

        bullets = await gpt_generate_jd_aligned_bullets(
            jd_text=jd_text,
            count=bullets_per_project,
            pillar_name=pillar,
            pillar_keywords=p_keywords,
            used_keywords=used,
            block_title=title,
        )

        # Mark pillar keywords used if they actually appeared in generated bullets
        for kw in p_keywords:
            if any(_token_regex(kw).search(strip_all_macros_keep_text(x)) for x in bullets):
                used.append(kw)

        bullets = _unique_bullets(bullets)

        blocks.append({
            "title": title,
            "label": p.get("label"),
            "pillar": pillar,
            "summary": p.get("summary"),
            "keywords": p_keywords,
            "kpis": p.get("kpis"),
            "fit_rationale": p.get("fit_rationale"),
            "bullets": bullets
        })

    return {
        "projects": blocks,
        "note": "List as Academic/Independent Projects. Do NOT imply prior employment or deceptive ownership."
    }


# ============================================================
# 🔁 Retarget sections — pillar-assignment + no duplicate bullets
#      (complete: titles, insertion, uniqueness tracking)
# ============================================================
async def _extract_block_title(section_text: str, block_start_idx: int) -> str:
    r"""
    Best-effort: look backwards for a subheading-like macro near this block.
    Does not alter LaTeX; used only to hint the generator.
    """
    head = section_text[:block_start_idx][-4000:]
    for rx in [
        r"\\resumeSubheading\{([^}]*)\}",
        r"\\resumeSubSubheading\{([^}]*)\}",
        r"\\resumeHeading\{([^}]*)\}",
        r"\\textbf\{([^}]*)\}",
    ]:
        m = re.search(rx, head, flags=re.IGNORECASE)
        if m:
            return strip_all_macros_keep_text(m.group(1)).strip()
    return ""


async def _retarget_one_section(
    section_text: str,
    jd_text: str,
    pillar: Dict[str, object],
    used_keywords_global: Set[str],
) -> str:
    s_tag, e_tag = r"\resumeItemListStart", r"\resumeItemListEnd"
    out, i = [], 0
    while True:
        a = section_text.find(s_tag, i)
        if a < 0:
            out.append(section_text[i:])
            break
        b = section_text.find(e_tag, a)
        if b < 0:
            out.append(section_text[i:])
            break

        out.append(section_text[i:a])
        block = section_text[a:b]

        # remove all existing \resumeItem{...}
        def _remove_all_items(btxt: str) -> str:
            res, last = [], 0
            for (s, _, _, e) in find_resume_items(btxt):
                res.append(btxt[last:s]); last = e
            res.append(btxt[last:])
            return "".join(res)

        block_no_items = _remove_all_items(block)
        insert_at = block_no_items.find(s_tag) + len(s_tag)

        # generate exactly 3 bullets for this block using the assigned pillar
        block_title = await _extract_block_title(section_text, a)
        bullets_new = await gpt_generate_jd_aligned_bullets(
            jd_text=jd_text,
            count=3,
            pillar_name=str(pillar.get("name") or pillar.get("pillar") or "Primary Pillar"),
            pillar_keywords=list(pillar.get("keywords") or []),
            used_keywords=list(used_keywords_global),
            block_title=block_title or None,
        )

        # track used keywords across the resume if they appear in bullets
        for kw in pillar.get("keywords") or []:
            if any(_token_regex(kw).search(strip_all_macros_keep_text(x)) for x in bullets_new):
                used_keywords_global.add(canonicalize_token(kw))

        inject = "".join(f"\n  \\resumeItem{{{t}}}" for t in bullets_new)
        block = block_no_items[:insert_at] + inject + block_no_items[insert_at:]
        out.append(block)
        out.append(section_text[b:b + len(e_tag)])
        i = b + len(e_tag)
    return "".join(out)


async def retarget_experience_sections_with_gpt(tex_content: str, jd_text: str) -> str:
    """
    Assign up to 4 JD pillars across Experience blocks (distinct focus per block).
    Projects section is aligned to the top pillar to surface the best-fit project.
    Enforces global keyword diversity and removes duplicated bullets.
    """
    pillars = await extract_pillars_gpt(jd_text)
    if not pillars:
        return tex_content

    used_keywords_global: Set[str] = set()

    # Retarget Experience
    exp_pat = section_rx("Experience")
    out, pos, exp_block_idx = [], 0, 0
    for m in exp_pat.finditer(tex_content):
        out.append(tex_content[pos:m.start()])
        section = m.group(1)

        built, i = [], 0
        s_tag, e_tag = r"\resumeItemListStart", r"\resumeItemListEnd"
        while True:
            a = section.find(s_tag, i)
            if a < 0:
                break
            b = section.find(e_tag, a)
            if b < 0:
                break

            pillar_idx = exp_block_idx if exp_block_idx < 4 else 3
            pillar = pillars[pillar_idx]
            block_text = section[i:b + len(e_tag)]
            block_text = await _retarget_one_section(block_text, jd_text, pillar, used_keywords_global)
            built.append(block_text)
            exp_block_idx += 1
            i = b + len(e_tag)

        new_section = "".join(built) + section[i:] if built else section
        out.append(new_section)
        pos = m.end()
    out.append(tex_content[pos:])
    tex_content = "".join(out)

    # Retarget Projects — use the top pillar to showcase the single strongest alignment
    proj_pat = section_rx("Projects")
    out, pos = [], 0
    for m in proj_pat.finditer(tex_content):
        out.append(tex_content[pos:m.start()])
        section = m.group(1)
        top_pillar = pillars[0]
        section = await _retarget_one_section(section, jd_text, top_pillar, used_keywords_global)
        out.append(section)
        pos = m.end()
    out.append(tex_content[pos:])

    return "".join(out)

# ============================================================
# 📄 PDF page-count helper
# ============================================================

def _pdf_page_count(pdf_bytes: Optional[bytes]) -> int:
    if not pdf_bytes:
        return 0
    return len(re.findall(rb"/Type\s*/Page\b", pdf_bytes))

# ============================================================
# 🏅 Achievements trimming (robust)
# ============================================================

ACHIEVEMENT_SECTION_NAMES = [
    "Achievements","Awards & Achievements","Achievements & Awards","Awards",
    "Honors & Awards","Honors","Awards & Certifications","Certifications & Awards",
    "Certifications","Certificates","Accomplishments","Activities & Achievements",
]

def _find_macro_items(block: str, macro: str) -> List[Tuple[int, int, int, int]]:
    r"""
    Find \macro{...} with balanced braces AND tolerate optional whitespace:
      \macro{...}   or   \macro   { ... }
    """
    out: List[Tuple[int, int, int, int]] = []
    i = 0
    macro_head = f"\\{macro}"
    n = len(macro_head)

    while True:
        i = block.find(macro_head, i)
        if i < 0:
            break

        j = i + n
        # allow spaces before "{"
        while j < len(block) and block[j].isspace():
            j += 1
        if j >= len(block) or block[j] != "{":
            i = j
            continue

        open_b = j
        depth, k = 0, open_b
        while k < len(block):
            ch = block[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    close_b = k
                    out.append((i, open_b, close_b, close_b + 1))
                    i = close_b + 1
                    break
            k += 1
        else:
            break

    return out

def _remove_last_any_bullet(section_text: str) -> Tuple[str, bool, str]:
    items = find_resume_items(section_text)
    if items:
        s, _op, _cl, e = items[-1]
        return section_text[:s] + section_text[e:], True, "resumeItem"
    subitems = _find_macro_items(section_text, "resumeSubItem")
    if subitems:
        s, _op, _cl, e = subitems[-1]
        return section_text[:s] + section_text[e:], True, "resumeSubItem"
    item_positions = [m.start() for m in re.finditer(r"\\item\b", section_text)]
    if item_positions:
        start = item_positions[-1]
        tail_m = re.search(r"(\\item\b|\\end\{itemize\}|\\resumeItemListEnd)", section_text[start+5:])
        end = len(section_text) if not tail_m else start + 5 + tail_m.start()
        return section_text[:start] + section_text[end:], True, "item"
    return section_text, False, ""

def _strip_empty_itemize_blocks(section_text: str) -> str:
    start_tag, end_tag = r"\resumeItemListStart", r"\resumeItemListEnd"
    def _has_items(b: str) -> bool:
        return bool(find_resume_items(b)) or bool(_find_macro_items(b, "resumeSubItem")) or bool(re.search(r"\\item\b", b))
    out, i = [], 0
    while True:
        a = section_text.find(start_tag, i)
        if a < 0: out.append(section_text[i:]); break
        b = section_text.find(end_tag, a)
        if b < 0: out.append(section_text[i:]); break
        block = section_text[a:b]
        if _has_items(block):
            out.append(section_text[i:b + len(end_tag)])
        else:
            out.append(section_text[i:a])  # drop empty
        i = b + len(end_tag)
    return "".join(out)

def _find_achievements_section_span_fuzzy(tex: str) -> Optional[Tuple[int, int, str]]:
    keywords = ("achiev", "award", "honor", "cert", "accomplish", "activity")
    last_match = None
    for m in SECTION_HEADER_RE.finditer(tex):
        title = (m.group(1) or "").lower()
        if any(k in title for k in keywords):
            start = m.start()
            next_m = SECTION_HEADER_RE.search(tex, m.end())
            end = next_m.start() if next_m else tex.find(r"\end{document}")
            if end == -1: end = len(tex)
            last_match = (start, end, title)
    return last_match

def remove_one_achievement_bullet(tex_content: str) -> Tuple[str, bool]:
    for sec in ACHIEVEMENT_SECTION_NAMES:
        pat = section_rx(sec)
        m = pat.search(tex_content)
        if not m: continue
        full = m.group(1)
        new_sec, removed, how = _remove_last_any_bullet(full)
        if removed:
            log_event(f"✂️ [TRIM] Removed last bullet from '{sec}' via {how}.")
            new_sec = _strip_empty_itemize_blocks(new_sec)
            return tex_content[:m.start()] + new_sec + tex_content[m.end():], True
    fuzzy_span = _find_achievements_section_span_fuzzy(tex_content)
    if fuzzy_span:
        start, end, title = fuzzy_span
        full = tex_content[start:end]
        new_sec, removed, how = _remove_last_any_bullet(full)
        if removed:
            log_event(f"✂️ [TRIM] Fuzzy match '{title}' — removed via {how}.")
            new_sec = _strip_empty_itemize_blocks(new_sec)
            return tex_content[:start] + new_sec + tex_content[end:], True
    log_event("ℹ️ [TRIM] No Achievements-like bullets found to remove.")
    return tex_content, False

def remove_last_bullet_from_sections(tex_content: str, sections: Tuple[str, ...] = ("Projects", "Experience")) -> Tuple[str, bool]:
    """
    Remove the last bullet from the last-found section among the given names.
    Bottom-up page saver when Achievements can't trim any further.
    """
    last_span = None
    for sec in sections:
        pat = section_rx(sec)
        m = None
        # Find the LAST occurrence for each section
        for m2 in pat.finditer(tex_content):
            m = m2
        if m:
            last_span = (sec, m.start(), m.end(), m.group(1))
    if not last_span:
        return tex_content, False

    sec, s, e, full = last_span
    new_sec, removed, how = _remove_last_any_bullet(full)
    if removed:
        log_event(f"✂️ [TRIM] Removed last bullet from '{sec}' via {how}.")
        new_sec = _strip_empty_itemize_blocks(new_sec)
        return tex_content[:s] + new_sec + tex_content[e:], True
    return tex_content, False

# ============================================================
# 🔎 Coverage normalization helpers (LaTeX unescape + variants)
# ============================================================

_LATEX_UNESC = [
    (r"\#", "#"), (r"\$", "$"), (r"\%", "%"), (r"\&", "&"),
    (r"\_", "_"), (r"\/", "/"),
]

def _plain_text_for_coverage(tex: str) -> str:
    """
    Produce a plain, human-ish text for coverage matching:
      - strip macros
      - unescape common LaTeX sequences (C\\# -> C#, CI\\/CD -> CI/CD)
      - collapse whitespace
    """
    s = strip_all_macros_keep_text(tex)
    for a, b in _LATEX_UNESC:
        s = s.replace(a, b)
    # very common resume tokens:
    s = s.replace("C\\#", "C#").replace("CI\\/CD", "CI/CD")
    s = s.replace("A\\/B", "A/B").replace("R\\&D", "R&D")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Map canonical token -> acceptable variant spellings for coverage only
_VARIANTS = {
    "kubernetes": ["k8s"],
    "node.js": ["nodejs", "node js", "node"],
    "ci/cd": ["ci cd", "ci-cd"],
    "llms": ["llm", "large language models", "large-language models"],
    "openai api": ["openai"],
    "hugging face transformers": ["hf transformers", "transformers"],
    "postgresql": ["postgres", "postgres sql"],
    "c++": ["cpp"],
    "c#": ["c sharp", "csharp"],
    "sql": ["t-sql", "pl/sql", "ms sql", "postgres", "mysql"],
    "bigquery": ["google bigquery", "big query"],
}

def _expand_variants(token: str) -> List[str]:
    """
    For a canonical token, return a set of variant strings to try when matching.
    Includes punctuation-relaxed forms for robustness (e.g., 'nodejs' vs 'node.js').
    """
    t = canonicalize_token(token).lower().strip()
    alts = _VARIANTS.get(t, [])
    relaxed = {
        t,
        t.replace(".", ""),
        t.replace("/", " "),
        t.replace("-", " "),
    }
    return sorted({*alts, *relaxed})

def _word_count(s: str) -> int:
    return len(re.findall(r"\w+", s))

def _inject_phrase(base: str, tokens: List[str]) -> str:
    """Append a compact phrase with 1–3 tokens; keep ≤24 words; keep sentence natural; avoid brackets."""
    plain = strip_all_macros_keep_text(base).rstrip(".")
    if not tokens:
        return latex_escape_text(plain)

    # prefer 'using ...' (no parentheses), then 'with ...', then 'via ...'
    variants = [
        f" using {', '.join(tokens)}",
        f" with {', '.join(tokens)}",
        f" via {tokens[-1]}",
    ]
    for phr in variants:
        if _word_count(plain + " " + phr) <= 24:
            return latex_escape_text(plain + phr)

    # if everything is too long, append just the last token with 'using'
    tail = f" using {tokens[-1]}"
    if _word_count(plain + " " + tail) <= 24:
        return latex_escape_text(plain + tail)

    return latex_escape_text(plain)


def _alloc_tokens_across_bullets(bullets: List[str], missing: List[str], max_per_bullet: int = 2) -> List[List[str]]:
    """Greedy round-robin distribution so every bullet carries tokens; favors speed/coverage."""
    q = [canonicalize_token(t) for t in missing if str(t).strip()]
    q = list(dict.fromkeys(q))  # dedupe, keep order
    buckets = [[] for _ in bullets]
    i = 0
    for tok in q:
        # find next bullet with capacity
        tries = 0
        while tries < len(bullets) and len(buckets[i]) >= max_per_bullet:
            i = (i + 1) % len(bullets); tries += 1
        if len(buckets[i]) < max_per_bullet:
            buckets[i].append(tok)
            i = (i + 1) % len(bullets)
    return buckets

def _ensure_number_metric(s: str) -> str:
    """If bullet lacks a number/metric/timeframe, append a tiny timeframe to satisfy constraint."""
    plain = strip_all_macros_keep_text(s)
    if re.search(r"(\d+%?|\$?\d|\b(day|week|month|year|quarter)s?\b)", plain, flags=re.I):
        return s
    return s.rstrip(".") + " in 4 weeks"

def inject_missing_keywords_deterministic(tex: str, missing_tokens: List[str]) -> str:
    """
    Force-insert exact tokens **inside sentences** of \\resumeItem bullets (Experience/Projects only).
    Keeps ≤24 words and ensures each bullet has a number/timeframe.
    """
    sections = ("Experience", "Projects")
    for sec in sections:
        pat = section_rx(sec)
        out, pos = [], 0
        for m in pat.finditer(tex):
            out.append(tex[pos:m.start()])
            section = m.group(1)

            items = find_resume_items(section)
            if not items:
                out.append(section); pos = m.end(); continue

            # current bullet texts
            bullets = []
            for (_s, op, cl, _e) in items:
                bullets.append(section[op+1:cl])

            # distribute tokens over bullets
            alloc = _alloc_tokens_across_bullets(bullets, missing_tokens, max_per_bullet=2)

            # patch each bullet
            new_texts = []
            for txt, toks in zip(bullets, alloc):
                base = strip_all_macros_keep_text(txt)
                base = _ensure_number_metric(base)  # guarantee a number/timeframe
                if toks:
                    new_texts.append(_inject_phrase(base, toks))
                else:
                    new_texts.append(latex_escape_text(base))

            section = replace_resume_items(section, new_texts)
            out.append(section); pos = m.end()
        out.append(tex[pos:])
        tex = "".join(out)
    return tex

# ============================================================
# 🧠 Main optimizer — Coursework + Skills + JD-retargeted bullets (High-Recall + Proactive)
# ============================================================

async def optimize_resume_latex(base_tex: str, jd_text: str) -> str:
    """
    JD-only:
      • Replace Experience/Projects bullets with quantified, JD-aligned bullets.
      • Rebuild Skills from JD tokens (short, canonical, includes Soft Skills bucket).
      • Seed coverage before refinement.
    """
    log_event("🟨 [AI] Coursework, Skills, and Experience (JD-only)")
    preamble, body = _split_preamble_body(base_tex)

    # Coursework (JD-driven)
    courses = await extract_coursework_gpt(jd_text, max_courses=24)
    body = replace_relevant_coursework_distinct(body, courses, max_per_line=8)

    # Skills (JD-driven)
    all_skills_raw, protected = await extract_skills_gpt(jd_text)
    all_skills = prune_and_compact_skills(all_skills_raw, protected=protected)
    body = await replace_skills_section(body, all_skills)

    # Experience/Projects: rewrite bullets for JD pillars
    body = await retarget_experience_sections_with_gpt(body, jd_text)

    # Pre-coverage improvement (bullets-only; strict JD tokens)
    try:
        tokens = await get_coverage_targets_from_jd(jd_text, strict=True)
        cov = compute_keyword_coverage_bullets(body, tokens)
        if cov["ratio"] < 1.00:
            log_event(f"⚙️ [PRE-COVERAGE] bullets {cov['ratio']:.2%} → improving.")
            body = await gpt_improve_for_missing_keywords(body, jd_text, cov["missing"])
        else:
            log_event("✅ [PRE-COVERAGE] Bullets already cover all JD tokens.")
    except Exception as e:
        log_event(f"⚠️ [PRE-COVERAGE] Skipped due to error: {e}")

    final = _merge_tex(preamble, body)
    log_event("✅ [AI] Body seeded for full bullets coverage with quantified JD bulleting.")
    return final

# ============================================================
# ✨ Humanize ONLY \resumeItem{…} text (run once after ≥90% coverage)
# ============================================================

async def humanize_experience_bullets(tex_content: str) -> str:
    log_event("🟨 [HUMANIZE] Targeting EXPERIENCE/PROJECTS")

    async def _humanize_block(block: str) -> str:
        items = find_resume_items(block)
        if not items:
            return block
        plain_texts = []
        for (_s, open_b, close_b, _e) in items:
            inner = block[open_b + 1:close_b]
            txt = strip_all_macros_keep_text(inner)
            plain_texts.append(txt[:1000].strip())

        async def rewrite_one(text: str, idx: int) -> str:
            # Use local gateway which handles creds + fallback
            api_base = (getattr(config, "API_BASE_URL", "") or "http://127.0.0.1:8000").rstrip("/")
            url = f"{api_base}/api/superhuman/rewrite"
            payload = {"text": text, "mode": "resume", "tone": "balanced", "latex_safe": True}

            for _attempt in range(2):
                try:
                    async with httpx.AsyncClient(timeout=2000.0) as client:
                        r = await client.post(url, json=payload)
                    if r.status_code == 200:
                        data = r.json()
                        rew = (data.get("rewritten") or "").strip()
                        # strip any accidental local-fallback label and fold to single line
                        rew = _FALLBACK_TAG_RE.sub("", rew).replace("\n", " ").strip()
                        if rew:
                            return latex_escape_text(rew)
                except Exception:
                    await asyncio.sleep(0.4)
            return latex_escape_text(text)

        sem = asyncio.Semaphore(5)
        async def lim(i, t):
            async with sem:
                return await rewrite_one(t, i)
        humanized = await asyncio.gather(*[lim(i, t) for i, t in enumerate(plain_texts, 1)])
        return replace_resume_items(block, humanized)

    for sec_name in ["Experience", "Projects"]:
        pat = section_rx(sec_name)
        out, pos = [], 0
        for m in pat.finditer(tex_content):
            out.append(tex_content[pos:m.start()])
            section = m.group(1)
            s_tag, e_tag = r"\resumeItemListStart", r"\resumeItemListEnd"
            rebuilt, i = [], 0
            while True:
                a = section.find(s_tag, i)
                if a < 0: rebuilt.append(section[i:]); break
                b = section.find(e_tag, a)
                if b < 0: rebuilt.append(section[i:]); break
                rebuilt.append(section[i:a])
                block = section[a:b]
                block = await _humanize_block(block)
                rebuilt.append(block)
                rebuilt.append(section[b:b + len(e_tag)])
                i = b + len(e_tag)
            out.append("".join(rebuilt)); pos = m.end()
        out.append(tex_content[pos:])
        tex_content = "".join(out)
    return tex_content

_EDU_SPLIT_ANCHOR = re.compile(
    r"(%-----------EDUCATION-----------)|\\section\*?\{\s*Education\s*\}",
    re.IGNORECASE
)

def _split_preamble_body(tex: str) -> tuple[str, str]:
    m = _EDU_SPLIT_ANCHOR.search(tex or "")
    if not m:
        # strip a trailing \end{document} if present
        return "", re.sub(r"\\end\{document\}\s*$", "", tex or "")
    start = m.start()
    preamble = (tex or "")[:start]
    body = re.sub(r"\\end\{document\}\s*$", "", (tex or "")[start:])
    return preamble, body

def _merge_tex(preamble: str, body: str) -> str:
    out = (str(preamble).strip() + "\n\n" + str(body).strip()).rstrip()
    # ensure exactly one \end{document}
    out = re.sub(r"\\end\{document\}\s*$", "", out).rstrip()
    out += "\n\\end{document}\n"
    return out

def _sanitize_improved_body(s: str) -> str:
    s = (s or "").replace("```latex", "").replace("```", "").strip()
    s = re.sub(r"(?is)\\documentclass.*?\\begin\{document\}", "", s)
    s = re.sub(r"(?is)\\end\{document\}", "", s)
    return s.strip()

# ============================================================
# 🔍 JD keyword coverage helpers (target = jd_keywords + requirements)
# ============================================================

def _token_regex(token: str) -> re.Pattern:
    """
    Regex for a token that may include punctuation like C++, CI/CD, Node.js.
    Uses alpha-num boundaries rather than \\b to tolerate symbols.
    """
    t = token.lower().strip()
    t = re.escape(t)
    t = t.replace(r"\ ", r"\s+").replace(r"\/", r"\s*\/\s*").replace(r"\.", r"\.")
    return re.compile(rf"(?<![a-z0-9]){t}(?![a-z0-9])", re.IGNORECASE)

def _present_tokens_in_text(text_plain: str, tokens: Iterable[str]) -> Tuple[Set[str], Set[str]]:
    """
    Variant-aware presence check:
      - expands each canonical token into acceptable variants
      - matches using a relaxed boundary regex (_token_regex)
    """
    present, missing = set(), set()
    low_text = text_plain.lower()
    for tok in {canonicalize_token(t).lower().strip() for t in tokens if str(t).strip()}:
        hit = False
        for v in _expand_variants(tok):
            if _token_regex(v).search(low_text):
                hit = True
                break
        (present if hit else missing).add(tok)
    return present, missing

# ============================================================
# 🎯 Bullets-only coverage (Experience/Projects only)
# ============================================================

def _extract_bullets_plain(tex: str, sections: Tuple[str, ...] = ("Experience", "Projects")) -> str:
    """
    Build a plain-text string from \\resumeItem{...} within specified sections only.
    """
    chunks = []
    for sec in sections:
        pat = section_rx(sec)
        for m in pat.finditer(tex):
            section_text = m.group(1)
            for (_s, op, cl, _e) in find_resume_items(section_text):
                inner = section_text[op + 1:cl]
                chunks.append(strip_all_macros_keep_text(inner))
    plain = " ".join(chunks)
    # normalize common LaTeX escapes for tokens like C#, CI/CD, R&D, A/B
    for a, b in _LATEX_UNESC:
        plain = plain.replace(a, b)
    plain = plain.replace("C\\#", "C#").replace("CI\\/CD", "CI/CD")
    plain = plain.replace("A\\/B", "A/B").replace("R\\&D", "R&D")
    return re.sub(r"\s+", " ", plain).strip()

def compute_keyword_coverage_bullets(tex_content: str, tokens_for_coverage: List[str]) -> Dict[str, object]:
    """
    Compute JD keyword coverage **only** over Experience/Projects bullets.
    """
    plain = _extract_bullets_plain(tex_content)
    present, missing = _present_tokens_in_text(plain, tokens_for_coverage)
    total = max(1, len(set(tokens_for_coverage)))
    ratio = len(present) / total
    return {
        "ratio": ratio,
        "present": sorted(present),
        "missing": sorted(missing),
        "total": total
    }

# ============================================================
# 🪄 Coverage token filtering (skip soft/unwinnable items)
# ============================================================

_SKIP_PATTERNS = [
    r"english\s*\(professional\)", r"chinese\s*\(professional\)",
    r"\bcommunication\b", r"\bteamwork\b", r"\bcollaboration\b",
    r"debugging workflows", r"strong interest", r"\bcuriosity\b",
]

def _is_coverage_token(tok: str) -> bool:
    ls = canonicalize_token(tok).lower().strip()
    return not any(re.search(p, ls) for p in _SKIP_PATTERNS)

async def get_coverage_targets_from_jd(jd_text: str, strict: bool = True) -> List[str]:
    """
    Build the set of tokens the score is based on.
    strict=True → include ALL protected tokens (jd_keywords + requirements).
    strict=False → skip low-signal/soft items (legacy behavior).
    """
    _combined, protected = await extract_skills_gpt(jd_text)
    kept = []
    for t in protected:
        ct = canonicalize_token(t)
        if strict:
            kept.append(ct)
        else:
            if _is_coverage_token(ct):
                kept.append(ct)
    return sorted(list({canonicalize_token(t).lower() for t in kept if t}))



def compute_keyword_coverage(tex_content: str, tokens_for_coverage: List[str]) -> Dict[str, object]:
    """
    Compute coverage over fully plain text derived from LaTeX:
      - macros stripped
      - common LaTeX escapes unescaped (so C#, CI/CD, R&D, A/B match)
    """
    plain = _plain_text_for_coverage(tex_content)
    present, missing = _present_tokens_in_text(plain, tokens_for_coverage)
    total = max(1, len(set(tokens_for_coverage)))
    ratio = len(present) / total
    return {
        "ratio": ratio,
        "present": sorted(present),
        "missing": sorted(missing),
        "total": total
    }

# ============================================================
# ✍️ GPT step to weave in missing keywords (truthfully) — Skills + Experience/Projects
# ============================================================
async def gpt_improve_for_missing_keywords(body_tex: str, jd_text: str, missing_tokens: List[str]) -> str:
    r"""
    JD-FIRST REBUILD (2–pass with coverage retry).

    What you MAY do:
      • REWRITE or DELETE misaligned \resumeItem bullets under Experience / Projects.
      • REPLACE them with JD-aligned bullets that *still* use the SAME list macros present
        (\resumeItemListStart/\resumeItemListEnd and \resumeItem{...} lines).
      • Strengthen the \section{Skills} block with concise tokens only (no prose).

    What you MUST NOT do:
      • Do NOT add \documentclass / preamble; return BODY ONLY (Education → end).
      • Do NOT remove list start/end pairs or break LaTeX macro balance.
      • Do NOT invent employers or institutions. If a role is unrelated, keep the heading but
        replace bullets with generic, truthful, task-style bullets (e.g., “Built …, evaluated …”).
      • Every bullet MUST have a number (%, #, $, ×, or timeframe) and be ≤ 24 words.
      • No keyword dumps; keywords must be within natural sentences (verbs + objects).

    Goal:
      Maximize JD keyword coverage using the provided missing_tokens while keeping one-page spirit
      (≤ 3 bullets per block unless it still fits on one page).

    Return:
      STRICT JSON ONLY → {"improved_body": "<LaTeX BODY from Education onward>"}
    """
    # ---- Helpers ----
    def _as_list(x):
        return x if isinstance(x, list) else []

    def _norm_tokens(xs: Iterable[str]) -> List[str]:
        return [canonicalize_token(t) for t in xs if str(t).strip()]

    def _truncate(s: str, n: int) -> str:
        return s[:n] if isinstance(s, str) and len(s) > n else s

    # Cap explicit token list for prompt; we still pass full JD for the model to infer others
    uniq_missing = sorted({canonicalize_token(t) for t in _as_list(missing_tokens)})
    shown_missing = ", ".join(uniq_missing[:80])  # show up to 80 explicitly

    MAX_BODY_CHARS = 12000
    body_snippet = body_tex[-MAX_BODY_CHARS:] if len(body_tex) > MAX_BODY_CHARS else (body_tex or "")

    MAX_JD_CHARS = 12000
    jd_snippet = (jd_text or "")[:MAX_JD_CHARS]

    example = '{"improved_body": "%-----------EDUCATION-----------\\n..."}'

    # Strong, explicit constraints so the model keeps macros intact and weaves keywords into bullets
    rules = (
        "You are rewriting ONLY the LaTeX BODY (Education→end). Keep LaTeX macros balanced and intact.\n"
        f"- Weave the following keywords into bullets via natural sentences (not comma dumps): {shown_missing}.\n"
        "- Preserve section structure. Keep existing \\section titles.\n"
        "- Inside Experience/Projects blocks:\n"
        "  • You may delete misaligned \\resumeItem lines.\n"
        "  • You must keep \\resumeItemListStart / \\resumeItemListEnd pairs.\n"
        "  • Add NEW \\resumeItem{...} lines that match the JD, using truthful, neutral wording.\n"
        "  • Do NOT fabricate employers; if the role is off-fit, keep the heading but write JD-aligned task bullets.\n"
        "- Skills: concise tokens/phrases only (no prose sentences).\n"
        "- Bullet style:\n"
        "  • Start with a strong past-tense verb; include 1–3 exact JD keywords within the sentence.\n"
        "  • Include a metric/timeframe in EVERY bullet (%, #, $, ×, or explicit months/weeks).\n"
        "  • ≤ 24 words per bullet; no first person; no keyword dumps; no semicolon chains.\n"
        "- Keep one-page spirit: ≤ 3 bullets per block unless it still fits.\n"
        "- Return STRICT JSON ONLY with the single key improved_body. No markdown fences."
    )

    prompt = (
        f"{rules}\n\n"
        f"RETURN STRICT JSON ONLY like: {example}\n\n"
        "JOB DESCRIPTION (snippet):\n"
        f"{jd_snippet}\n\n"
        "CURRENT BODY (LaTeX, Education→end; snippet if long):\n"
        f"{body_snippet}"
    )

    # -------- Pass 1 --------
    data = await gpt_json(prompt, temperature=0.0)
    ib_raw = (data or {}).get("improved_body", "")
    ib = _sanitize_improved_body(str(ib_raw)) if ib_raw else ""

    if not ib.strip():
        return body_tex

    # Remove any accidental preamble/document wrappers
    ib = re.sub(r"(?is)\\documentclass.*?\\begin\\{document\\}", "", ib).strip()
    ib = re.sub(r"(?is)\\end\\{document\\}", "", ib).strip()

    # -------- Coverage Check + Retry (minimal patch) --------
    cov = {"present": [], "missing": []}  # ensure defined even if coverage fails
    try:
        tokens = await get_coverage_targets_from_jd(jd_text, strict=True)
        cov = compute_keyword_coverage(ib, tokens)
        still_missing = sorted(set(_norm_tokens(cov.get("missing", []))))
    except Exception:
        still_missing = []

    # If we explicitly had a missing_tokens list, prioritize those for retry
    if uniq_missing:
        present_set = set(_norm_tokens(cov.get("present", []))) if isinstance(cov, dict) else set()
        still_missing = [t for t in uniq_missing if t not in present_set]

    if still_missing:
        retry_rules = (
            "PATCH ONLY the bullet texts (\\resumeItem{...}) to integrate the EXACT tokens listed below, "
            "keeping LaTeX structure identical and all previous constraints:\n"
            f"- Tokens to weave (must appear inside sentences, not as dumps): {', '.join(still_missing[:60])}\n"
            "- Do not add or remove list starts/ends or section headers.\n"
            "- Keep ≤ 24 words and at least one number in EVERY bullet."
        )
        retry_prompt = (
            f"{retry_rules}\n\n"
            f"RETURN STRICT JSON ONLY like: {example}\n\n"
            "CURRENT BODY TO PATCH (LaTeX):\n"
            f"{_truncate(ib, 12000)}"
        )
        data2 = await gpt_json(retry_prompt, temperature=0.0)
        ib2_raw = (data2 or {}).get("improved_body", "")
        ib2 = _sanitize_improved_body(str(ib2_raw)) if ib2_raw else ""
        if ib2.strip():
            ib = re.sub(r"(?is)\\documentclass.*?\\begin\\{document\\}", "", ib2).strip()
            ib = re.sub(r"(?is)\\end\\{document\\}", "", ib).strip()

    return ib or body_tex

# ============================================================
# 🔁 Coverage-driven refinement (aim for ≥ 90%)
# ============================================================

async def _rebuild_skills_safely(tex_content: str, jd_text: str) -> str:
    all_skills_raw, protected = await extract_skills_gpt(jd_text)
    all_skills = prune_and_compact_skills(all_skills_raw, protected=protected)
    return await replace_skills_section(tex_content, all_skills)

# ============================================================
# 🔁 Coverage-driven refinement (aim for ≥ 99%) — Skills + Experience/Projects every round
# ============================================================

async def refine_resume_to_keyword_coverage(
    tex_content: str,
    jd_text: str,
    min_ratio: float = 1.00,
    max_rounds: int = 2,
) -> Tuple[str, Dict[str, object], list]:
    tokens = await get_coverage_targets_from_jd(jd_text, strict=True)
    pre, body = _split_preamble_body(tex_content)
    history = []

    # Seed once: rebuild skills from JD (keeps concise, canonical tokens)
    merged = _merge_tex(pre, body)
    merged = await _rebuild_skills_safely(merged, jd_text)
    pre, body = _split_preamble_body(merged)

    prev_ratio = -1.0
    for rnd in range(1, max_rounds + 1):
        cur_tex = _merge_tex(pre, body)
        cov = compute_keyword_coverage_bullets(cur_tex, tokens)
        history.append({"round": rnd, "coverage": cov["ratio"], "missing": cov["missing"][:40]})
        log_event(f"📊 [COVERAGE r{rnd}] bullets {len(cov['present'])}/{cov['total']} → {cov['ratio']:.1%}")

        if cov["ratio"] >= min_ratio:
            return cur_tex, cov, history

        # 🚀 Try deterministic injector FIRST for instant coverage gains (no API call)
        if cov["missing"]:
            injected = inject_missing_keywords_deterministic(cur_tex, list(cov["missing"]))
            cov_inj = compute_keyword_coverage_bullets(injected, tokens)
            if cov_inj["ratio"] > cov["ratio"]:
                pre, body = _split_preamble_body(injected)
                if cov_inj["ratio"] >= min_ratio:
                    return _merge_tex(pre, body), cov_inj, history
                prev_ratio = cov_inj["ratio"]
                continue  # next round starts from improved state

        if abs(cov["ratio"] - prev_ratio) < 1e-4:
            # one last deterministic push before giving up
            injected = inject_missing_keywords_deterministic(cur_tex, list(cov["missing"]))
            cov_inj = compute_keyword_coverage_bullets(injected, tokens)
            if cov_inj["ratio"] > cov["ratio"]:
                pre, body = _split_preamble_body(injected)
                prev_ratio = cov_inj["ratio"]
                continue
            log_event("⏭️ [COVERAGE] No progress; stopping at current best.")
            break
        prev_ratio = cov["ratio"]

        # 🎯 Single GPT micro-patch (keep progress; do NOT retarget or rebuild again)
        improved_body = await gpt_improve_for_missing_keywords(body, jd_text, cov["missing"])
        pre, body = _split_preamble_body(_merge_tex(pre, improved_body))


    # Final snapshot (no injection)
    final_tex = _merge_tex(pre, body)
    cov = compute_keyword_coverage_bullets(final_tex, tokens)
    return final_tex, cov, history

# ============================================================
# 🚀 Endpoint (iterate to ≥100% keyword coverage, humanize once after)
#     + route aliases for frontend fallbacks (/run, /submit) and legacy (/optimize/*)
# ============================================================
# NOTE: This router is mounted at prefix="/api/optimize" in main.py

@router.post("/")         # POST /api/optimize/
@router.post("/run")      # POST /api/optimize/run
@router.post("/submit")   # POST /api/optimize/submit
async def optimize_endpoint(
    jd_text: str = Form(...),                           # required
    use_humanize: bool = Form(True),
    base_resume_tex: UploadFile | None = File(None),    # accept file uploads correctly
    extra_keywords: str | None = Form(None),            # keep types simple
):
    try:
        # Honor server default switch
        use_humanize = True if getattr(config, "HUMANIZE_DEFAULT_ON", True) else use_humanize

        # Basic input prep
        extras: List[str] = [x.strip() for x in (extra_keywords or "").split(",") if x and x.strip()]
        jd_text = (jd_text or "").strip()
        if not jd_text:
            raise HTTPException(status_code=400, detail="jd_text is required.")

        # ---- Load base .tex (upload if provided, else server default) ----
        raw_tex: str = ""
        if base_resume_tex is not None:
            tex_bytes = await base_resume_tex.read()
            if tex_bytes:
                tex = tex_bytes.decode("utf-8", errors="ignore")
                raw_tex = secure_tex_input(base_resume_tex.filename or "upload.tex", tex)

        if not raw_tex:
            default_path = getattr(config, "DEFAULT_BASE_RESUME", None)
            if isinstance(default_path, (str, bytes)):
                default_path = Path(default_path)
            if not default_path or not isinstance(default_path, Path) or not default_path.exists():
                raise HTTPException(
                    status_code=500,
                    detail=f"Default base resume not found at {default_path}"
                )
            raw_tex = default_path.read_text(encoding="utf-8")
            log_event(f"📄 Using server default base: {default_path}")

        # ---- Company/role for filenames ----
        company_name, role = await extract_company_role(jd_text)

        # Eligibility BEFORE rewrites (strict; mode-aware)
        eligibility = await compute_eligibility_any(raw_tex, jd_text, extra=extras)

        # ---- AI pipeline: seed, refine to coverage ----
        optimized_tex = await optimize_resume_latex(raw_tex, jd_text)
        optimized_tex, coverage_report, coverage_history = await refine_resume_to_keyword_coverage(
            optimized_tex,
            jd_text,
            min_ratio=1.0,
            max_rounds=2,
        )
        cur_tex = optimized_tex  # do NOT rebuild skills again here (preserve injected tokens)

        # ---------- Compile base ----------
        final_tex = render_final_tex(cur_tex)
        pdf_bytes_original = compile_latex_safely(final_tex)
        base_pages = _pdf_page_count(pdf_bytes_original)
        log_event(f"📄 Base PDF pages: {base_pages}")

        # ---------- Output destinations (flat folders) ----------
        safe_company, safe_role = safe_filename(company_name), safe_filename(role)
        paths = build_output_paths(company_name, role)  # /data/Optimized, /data/Humanized, /data/Cover Letters
        saved_paths: List[str] = []

        # ---------- Ensure ≤ 1 page by trimming from Achievements/Projects/Experience ----------
        MAX_TRIMS = 50
        cur_pdf_bytes = pdf_bytes_original
        cur_pages = base_pages
        trim_idx = 0

        while cur_pages > 1 and trim_idx < MAX_TRIMS:
            next_tex, removed = remove_one_achievement_bullet(cur_tex)
            if not removed:
                next_tex, removed = remove_last_bullet_from_sections(cur_tex, sections=("Projects", "Experience"))
                if not removed:
                    log_event("ℹ️ No more bullets to remove; stopping trim loop.")
                    break

            trim_idx += 1
            log_event(f"✂️ [TRIM {trim_idx}] Removed one bullet")
            next_tex_rendered = render_final_tex(next_tex)
            next_pdf_bytes = compile_latex_safely(next_tex_rendered)
            next_pages = _pdf_page_count(next_pdf_bytes)
            log_event(f"📄 [TRIM {trim_idx}] Pages now: {next_pages}")

            cur_tex, cur_pdf_bytes, cur_pages = next_tex, next_pdf_bytes, next_pages
            if cur_pages <= 1:
                log_event(f"✅ Fits on one page after {trim_idx} trims.")
                break

        # ---------- Humanize ONCE if requested and ≥90% coverage ----------
        pdf_bytes_humanized: Optional[bytes] = None
        humanized_tex: Optional[str] = None
        did_humanize = False

        if use_humanize and coverage_report["ratio"] >= 0.90:
            did_humanize = True
            humanized_tex = await humanize_experience_bullets(cur_tex)

            # Re-check and re-inject if humanization paraphrased tokens away
            tokens = await get_coverage_targets_from_jd(jd_text, strict=True)
            cov_h = compute_keyword_coverage_bullets(humanized_tex, tokens)
            if cov_h["ratio"] < 1.0 and cov_h["missing"]:
                humanized_tex = inject_missing_keywords_deterministic(humanized_tex, list(cov_h["missing"]))

            humanized_tex_rendered = render_final_tex(humanized_tex)
            pdf_bytes_humanized = compile_latex_safely(humanized_tex_rendered)

            # If humanized >1 page, mirror trim loop
            h_pages = _pdf_page_count(pdf_bytes_humanized)
            trim_h_idx = 0
            while h_pages > 1 and trim_h_idx < MAX_TRIMS:
                next_h_tex, removed = remove_one_achievement_bullet(humanized_tex)
                if not removed:
                    next_h_tex, removed = remove_last_bullet_from_sections(humanized_tex, sections=("Projects", "Experience"))
                    if not removed:
                        break
                h_rendered = render_final_tex(next_h_tex)
                h_pdf = compile_latex_safely(h_rendered)
                humanized_tex, pdf_bytes_humanized = next_h_tex, h_pdf
                h_pages = _pdf_page_count(pdf_bytes_humanized)
                trim_h_idx += 1

        # ---------- Save final outputs (flat libraries) ----------
        opt_path = paths["optimized"]   # /data/Optimized/Optimized - Sri_{Company}_{Role}.pdf
        hum_path = paths["humanized"]   # /data/Humanized/Humanized - Sri_Kadali_{Company}_{Role}.pdf

        if cur_pdf_bytes:
            opt_path.parent.mkdir(parents=True, exist_ok=True)
            opt_path.write_bytes(cur_pdf_bytes)
            saved_paths.append(str(opt_path))
            log_event(f"💾 [SAVE] Optimized PDF → {opt_path}")

        if did_humanize and pdf_bytes_humanized:
            hum_path.parent.mkdir(parents=True, exist_ok=True)
            hum_path.write_bytes(pdf_bytes_humanized)
            saved_paths.append(str(hum_path))
            log_event(f"💾 [SAVE] Humanized PDF → {hum_path}")
        elif use_humanize and coverage_report["ratio"] >= 0.90 and humanized_tex and not pdf_bytes_humanized:
            # Store failed humanized LaTeX nearby for debugging
            t = hum_path.with_name(f"FAILED_Humanized_{safe_company}_{safe_role}.tex")
            t.write_text(humanized_tex, encoding="utf-8")
            log_event(f"🧾 [DEBUG] Saved failed humanized LaTeX → {t}")

        # ---------- Response ----------
        return JSONResponse({
            "company_name": company_name,
            "role": role,
            "eligibility": eligibility,
            "optimized": {
                "tex": render_final_tex(cur_tex),
                "pdf_b64": base64.b64encode(cur_pdf_bytes or b"").decode("ascii"),
                "filename": str(opt_path) if cur_pdf_bytes else "",
            },
            "humanized": {
                "tex": render_final_tex(humanized_tex) if (did_humanize and humanized_tex) else "",
                "pdf_b64": base64.b64encode(pdf_bytes_humanized or b"").decode("ascii") if (did_humanize and pdf_bytes_humanized) else "",
                "filename": str(hum_path) if (did_humanize and pdf_bytes_humanized) else "",
            },
            # backward/flat keys used by the frontend cache
            "tex_string": render_final_tex(cur_tex),
            "pdf_base64": base64.b64encode(cur_pdf_bytes or b"").decode("ascii"),
            "pdf_base64_humanized": base64.b64encode(pdf_bytes_humanized or b"").decode("ascii")
                if (did_humanize and pdf_bytes_humanized) else None,
            "saved_paths": saved_paths,
            "coverage_ratio": coverage_report["ratio"],
            "coverage_present": coverage_report["present"],
            "coverage_missing": coverage_report["missing"],
            "coverage_history": coverage_history,
            "did_humanize": did_humanize,
            "extra_keywords": extras,
        })
    except Exception as e:
        log_event(f"💥 [PIPELINE] Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
