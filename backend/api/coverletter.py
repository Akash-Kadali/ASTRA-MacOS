# ============================================================
#  HIREX v2.1.0 — Cover Letter Generation Endpoint (FINAL)
#  ------------------------------------------------------------
#  Improvements vs 2.0.x:
#   • Scrubs fake-looking step numbers ((1), 1), 1.) in prose
#   • Replaces em/en/double/single dash separators with commas
#   • Strong LaTeX-safe escaping (&, %, $, #, _, {, }, ~, ^, \)
#   • Removes STAR scaffolding; fixes mid-word linebreaks
#   • Precise injection between salutation and signoff
#   • Saves to stable per-(Company,Role) context file (dedupe)
#   • Response includes memory_id=id=stable context key
#   • Saves final PDF to data/samples/Cover Letters/...
# ============================================================

from __future__ import annotations

import base64
import json
import re
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List
import difflib

import httpx
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI

from backend.core import config
from backend.core.utils import log_event, safe_filename, ensure_dir
from backend.core.compiler import compile_latex_safely
from backend.core.security import secure_tex_input

# --- shared helpers (try backend.api.*, then api.*) ---
try:
    from backend.api.render_tex import render_final_tex  # type: ignore
except Exception:  # pragma: no cover
    from api.render_tex import render_final_tex  # type: ignore

try:
    from backend.api.latex_parse import inject_cover_body as _shared_inject  # type: ignore
except Exception:  # pragma: no cover
    try:
        from api.latex_parse import inject_cover_body as _shared_inject  # type: ignore
    except Exception:
        _shared_inject = None  # type: ignore

router = APIRouter(prefix="/api/coverletter", tags=["coverletter"])
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

_DEFAULT_OAI_MODEL = "gpt-4o-mini"
_EXTRACT_MODEL = getattr(config, "COVERLETTER_MODEL", _DEFAULT_OAI_MODEL) or _DEFAULT_OAI_MODEL
_DRAFT_MODEL = getattr(config, "COVERLETTER_MODEL", _DEFAULT_OAI_MODEL) or _DEFAULT_OAI_MODEL

_DISABLE_SHARED_INJECTOR = True  # known-buggy guard


# -----------------------------
# Utilities
# -----------------------------
def _json_from_text(text: str, default: dict) -> dict:
    if not text:
        return default
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return default
    try:
        return json.loads(m.group(0))
    except Exception:
        return default


def _latex_escape_light(text: str) -> str:
    """
    Minimal, text-mode LaTeX escaping for generated/body/header content.
    - '&' -> 'and'
    - Escapes: %, $, #, _, {, }, ~, ^, backslash
    - Idempotent
    """
    if not text:
        return ""
    text = text.replace("&", " and ")
    repl = {
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\string~",
        "^": r"\string^",
        "\\": r"\textbackslash{}",
    }
    out = "".join(repl.get(ch, ch) for ch in text)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def _strip_star_labels(text: str) -> str:
    """Remove STAR scaffolding like '(situation/task)', 'Actions:', 'result and impact', etc., and tidy spaces."""
    if not text:
        return ""
    text = re.sub(
        r"(?i)\(\s*(?:situation|task|actions?|result(?:\s+and\s+impact)?|impact)"
        r"(?:\s*/\s*(?:task|actions?|result|impact))?\s*\)",
        "",
        text,
    )
    text = re.sub(
        r"(?im)^\s*(?:situation(?:\s*/\s*task)?|task|actions?|result(?:\s+and\s+impact)?|impact)\s*[:\-]\s*",
        "",
        text,
    )
    text = re.sub(
        r"(?i)\b(?:situation(?:\s*/\s*task)?|task|actions?|result(?:\s+and\s+impact)?|impact)\s*:\s*",
        "",
        text,
    )
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_body_whitespace(text: str) -> str:
    """
    Fix word-breaks and preserve paragraphs:
      - 'un-\nstructured' -> 'unstructured'
      - Join accidental intra-word breaks: 'f\n ields' -> 'fields'
      - Single newline -> space; double newline -> paragraph
    """
    if not text:
        return ""
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)             # hyphenated break
    text = re.sub(r"([A-Za-z])\s*\n\s*([A-Za-z])", r"\1\2", text)  # intra-word break
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)                   # single newline -> space
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _debullettify_and_dedash(text: str) -> str:
    """
    Remove enumerations and dashy separators to keep prose natural:
      - '(1)', '(2)' anywhere -> ''
      - Sentence/phrase-leading '1) ' or '1. ' -> ''
      - '—'/'–'/'--' or spaced ' - ' as separators -> comma + space
      - Collapse duplicate commas/spaces
    """
    if not text:
        return ""

    # Remove paren-numbers like (1), (2) etc. (keep years like (2025) by limiting to 1–2 digits)
    text = re.sub(r"\(\s*[0-9]{1,2}\s*\)\s*", "", text)

    # Remove leading enumerations at sentence/phrase starts: "1) step", "2. step"
    text = re.sub(r"(^|[.?!]\s+)\d{1,2}[.)]\s*", r"\1", text)

    # Replace em/en dashes and double hyphens with commas
    text = re.sub(r"\s*(?:—|–|--)\s*", ", ", text)

    # Replace spaced single dash used as a separator with a comma
    text = re.sub(r"\s-\s", ", ", text)

    # Trim excess commas/spaces
    text = re.sub(r"\s*,\s*,\s*", ", ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _postprocess_body(text: str) -> str:
    """Cleanup + humanize punctuation + LaTeX safety in the right order."""
    text = secure_tex_input(text or "")
    text = _strip_star_labels(text)
    text = _normalize_body_whitespace(text)
    text = _debullettify_and_dedash(text)
    return _latex_escape_light(text)


async def chat_text(system: str, user: str, model: str) -> str:
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()


async def chat_json(user_prompt: str, model: str) -> dict:
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
    )
    content = (resp.choices[0].message.content or "").strip()
    return _json_from_text(content, {})


# -----------------------------
# Extract Company + Role
# -----------------------------
async def extract_company_role(jd_text: str) -> Tuple[str, str]:
    jd_excerpt = (jd_text or "").strip()[:4000]
    prompt = (
        "Extract company name and the exact role title from the job description below.\n"
        'Return STRICT JSON (no prose) like {"company":"…","role":"…"}.\n\n'
        f"JD:\n{jd_excerpt}"
    )
    try:
        data = await chat_json(prompt, model=_EXTRACT_MODEL)
        return (data.get("company") or "Company").strip(), (data.get("role") or "Role").strip()
    except Exception as e:
        log_event("coverletter_extract_fail", {"error": str(e)})
        return "Company", "Role"


# ============================================================
# Draft Cover-Letter Body — enforced & grounded generation
# ============================================================

# Config knobs
_SENT_MIN, _SENT_MAX = 12, 28                   # sentence length band (words)
_TOTAL_KW_MIN, _TOTAL_KW_MAX = 5, 9            # total JD keywords to weave
_PER_SENT_KW_MAX = 2                           # keyword density cap per sentence
_LENGTH_BANDS = {"short": (120, 180), "standard": (180, 280), "long": (300, 400)}

_BUZZ_DEFAULT = [
    "passionate", "dynamic", "cutting edge", "team player",
    "synergy", "results-driven", "fast-paced", "leverage synergies",
]
_VERB_VARIANTS = ["built", "designed", "shipped", "created", "developed",
                  "implemented", "delivered", "launched", "scaled", "optimized"]

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\./_+]*")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CAMEL = re.compile(r"[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]+")
_TOOLISH = re.compile(r"[A-Za-z]+[0-9]+|[A-Za-z]+\.[A-Za-z]+|[-_/]")
_STOPWORDS = set("""
a an the and or but if while for with to of in on by from as at into over under
is are was were be been being this that these those i you he she they we it
""".split())

def _norm(s: str) -> str:
    return (s or "").replace("&", " and ").replace("—", ", ").replace("–", ", ").replace("--", ", ").strip()

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(text)]

def _extract_terms(text: str) -> set[str]:
    terms = set()
    for tok in set(_WORD.findall(text or "")):
        raw = tok.strip()
        if not raw:
            continue
        low = raw.lower()
        if low in _STOPWORDS or len(low) < 2:
            continue
        if _TOOLISH.search(raw) or _CAMEL.search(raw) or "-" in raw or "." in raw or "/" in raw:
            terms.add(raw)
        elif low in {"pytorch", "tensorflow", "spark", "airflow", "docker", "kubernetes",
                     "bigquery", "snowflake", "hive", "flink", "scikit", "sklearn",
                     "xgboost", "lightgbm", "postgres", "mysql", "redis", "elasticsearch",
                     "fastapi", "flask", "ray", "rag", "retrieval"}:
            terms.add(raw)
    return set(sorted(terms, key=str.lower))

def _extract_responsibilities(jd_text: str) -> List[str]:
    verbs = r"(own|lead|drive|design|build|ship|develop|maintain|scale|optimiz|evaluate|deploy|integrate|collaborate|partner)"
    lines = re.split(r"[\n;]|(?<=[.!?])\s+", jd_text or "")
    reps = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if re.search(rf"^\s*-?\s*{verbs}\b", s, re.IGNORECASE) or re.search(rf"\b{verbs}\b", s, re.IGNORECASE):
            s = re.sub(r"^\s*[-*•]\s*", "", s)
            s = re.sub(r"\s{2,}", " ", s).strip().rstrip(";")
            if 6 <= len(s.split()) <= 25:
                reps.append(s)
    # dedupe preserving order
    seen = set()
    out = []
    for r in reps:
        k = r.lower()
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out[:12]

def _overlap_priority(jd_terms: set[str], resume_terms: set[str]) -> tuple[List[str], List[str], List[str]]:
    jd_low, res_low = {t.lower(): t for t in jd_terms}, {t.lower(): t for t in resume_terms}
    shared = [jd_low[k] for k in jd_low.keys() & res_low.keys()]
    jd_only = [jd_low[k] for k in jd_low.keys() - res_low.keys()]
    res_only = [res_low[k] for k in res_low.keys() - jd_low.keys()]
    shared.sort(key=str.lower); jd_only.sort(key=str.lower); res_only.sort(key=str.lower)
    return shared, jd_only, res_only

def _dedupe_verbs_local(text: str) -> str:
    sentences = _SENT_SPLIT.split(text.strip())
    seen = set()
    for i, s in enumerate(sentences):
        m = re.match(r"^([A-Za-z]+)\b", s.strip())
        if not m:
            continue
        v = m.group(1).lower()
        if v in seen and v in _VERB_VARIANTS:
            idx = _VERB_VARIANTS.index(v)
            newv = _VERB_VARIANTS[(idx + 3) % len(_VERB_VARIANTS)]
            sentences[i] = re.sub(r"^[A-Za-z]+\b", newv.capitalize(), s.strip(), count=1)
        seen.add(v)
    return " ".join(sentences)

def _clean_text_local(s: str, banned_phrases: Optional[List[str]] = None) -> str:
    txt = _norm(s)
    txt = re.sub(r"^\s*(?:[#`>\-\*•]|\d+[.)])\s+", "", txt, flags=re.MULTILINE)
    banned = set((banned_phrases or []) + _BUZZ_DEFAULT)
    for b in sorted(banned, key=len, reverse=True):
        txt = re.sub(rf"\b{re.escape(b)}\b", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\bGPA\b[:\s]?\d+(\.\d+)?", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\bcourse(s|work)?\b.*?(completed|including|such as).*$", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\b(visa|relocation|sponsorship|available from)\b.*?$", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"[\[\]\{\}]+", "", txt)
    txt = re.sub(r"\s+,", ",", txt)
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    return txt

def _enforce_word_band_local(text: str, length: str) -> str:
    lo, hi = _LENGTH_BANDS.get(length, (180, 280))
    words = text.split()
    if len(words) <= hi and len(words) >= lo:
        return text
    sentences = _SENT_SPLIT.split(text.strip())
    out = []
    for s in sentences:
        candidate = " ".join(out + [s])
        if len(candidate.split()) <= hi:
            out.append(s)
        else:
            break
    trimmed = " ".join(out).strip()
    if len(trimmed.split()) < lo:
        extra = [x for x in sentences[len(out):] if x.strip()]
        if extra:
            trimmed = (trimmed + " " + extra[0]).strip()
    return trimmed

def _sentence_stats(text: str) -> List[int]:
    return [len(s.split()) for s in _SENT_SPLIT.split(text.strip()) if s.strip()]

def _enforce_sentence_band(text: str) -> str:
    sentences = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    fixed = []
    i = 0
    while i < len(sentences):
        s = sentences[i]
        n = len(s.split())
        if n > _SENT_MAX:
            split = re.split(r"; |, and |, ", s, maxsplit=1)
            if len(split) == 2:
                a, b = split
                if a and b:
                    sentences.insert(i+1, b.strip().capitalize())
                    s = a.strip()
                    n = len(s.split())
        if n < _SENT_MIN and i + 1 < len(sentences):
            s = (s.rstrip(",") + ", " + sentences[i+1].lstrip().lower())
            i += 1
        fixed.append(s)
        i += 1
    return " ".join(fixed)

def _shape_paragraphs(text: str, mode: str) -> str:
    sents = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    if not sents:
        return text.strip()
    if mode == "short":
        cut = max(1, min(len(sents)-1, len(sents)//3))
        return (" ".join(sents[:cut]).strip() + "\n\n" + " ".join(sents[cut:]).strip())
    n = len(sents)
    i1 = max(1, min(n-2, n//5))
    i2 = max(i1+1, min(n-1, (n*4)//5))
    return (" ".join(sents[:i1]).strip()
            + "\n\n" + " ".join(sents[i1:i2]).strip()
            + "\n\n" + " ".join(sents[i2:]).strip())

def _keywords_in_text(text: str, keywords: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    low = text.lower()
    for k in keywords:
        if not k:
            continue
        pat = re.escape(k.lower())
        c = len(re.findall(rf"\b{pat}\b", low))
        if c:
            counts[k] = c
    return counts

def _per_sentence_kw_counts(text: str, keywords: List[str]) -> List[int]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    res = []
    for s in sents:
        c = sum(1 for k in keywords if re.search(rf"\b{re.escape(k.lower())}\b", s.lower()))
        res.append(c)
    return res

def _allowed_terms(jd_terms: set[str], resume_terms: set[str]) -> set[str]:
    return set(t.lower() for t in jd_terms | resume_terms | set(_tokenize(" ".join(jd_terms | resume_terms))))

def _oov_terms(text: str, allowed: set[str]) -> set[str]:
    cand = set()
    for tok in set(_WORD.findall(text)):
        low = tok.lower()
        if low in _STOPWORDS:
            continue
        if _TOOLISH.search(tok) or _CAMEL.search(tok) or "-" in tok or "." in tok or "/" in tok:
            if low not in allowed:
                cand.add(tok)
    return cand

def _choose_kw_list(shared: List[str], jd_only: List[str], max_n: int = 14) -> List[str]:
    out = shared[:]
    for t in jd_only:
        if t not in out:
            out.append(t)
        if len(out) >= max_n:
            break
    return out

def _has_specific_company_detail(text: str, facts: List[str]) -> bool:
    if not facts:
        return True
    low = text.lower()
    for f in facts:
        f = (f or "").strip()
        if not f:
            continue
        parts = [p for p in re.split(r"\s+", f.lower()) if p and p not in _STOPWORDS]
        if len(parts) >= 2 and " ".join(parts) in low:
            return True
        if any(len(p) > 5 and re.search(rf"\b{re.escape(p)}\b", low) for p in parts):
            return True
    return False

def _has_why_now(text: str) -> bool:
    return bool(re.search(r"\b(recent|recently|now|right now|this quarter|this year|launch|launched|rollout|roadmap|momentum)\b", text.lower()))

async def draft_cover_body(
    jd_text: str,
    resume_text: str,
    company: str,
    role: str,
    tone: str,
    length: str,
    company_facts: Optional[List[str]] = None,
    first_30_day_ideas: Optional[List[str]] = None,
    banned_phrases: Optional[List[str]] = None,
) -> str:
    """
    BODY only; enforced structure + grounding:
      - Extract JD keywords & responsibilities; prefer JD–resume overlap
      - Enforce 5–9 total keywords, ≤2 per sentence
      - Sentence length band; paragraph shaping by length
      - OOV grounding against JD∪resume terms; fuzzy repair
      - Validate specific company detail and a “why now” reason
    """
    try:
        tone = (tone or "balanced").strip().lower()
        length = (length or "standard").strip().lower()
        if length not in _LENGTH_BANDS:
            length = "standard"

        # Pre-extraction and grounding
        jd_terms = _extract_terms(jd_text or "")
        resume_terms = _extract_terms(resume_text or "")
        shared, jd_only, _ = _overlap_priority(jd_terms, resume_terms)
        preferred_kw_pool = _choose_kw_list(shared, jd_only, max_n=20)

        if len(preferred_kw_pool) < _TOTAL_KW_MIN:
            toks = [t for t in _tokenize(jd_text) if t not in _STOPWORDS and len(t) > 3]
            freq: Dict[str, int] = {}
            for t in toks:
                freq[t] = freq.get(t, 0) + 1
            for t, _n in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
                if t not in preferred_kw_pool:
                    preferred_kw_pool.append(t)
                if len(preferred_kw_pool) >= 20:
                    break

        jd_resps = _extract_responsibilities(jd_text or "")
        allowed = _allowed_terms(jd_terms, resume_terms)

        # Prompt with JSON scaffold and strict shape
        facts_str = ""
        if company_facts:
            slim = [f.strip() for f in company_facts if f and f.strip()]
            if slim:
                facts_str = "Company facts (may reference 1): " + " | ".join(slim[:3])

        ideas_str = ""
        if first_30_day_ideas:
            slim = [f.strip() for f in first_30_day_ideas if f and f.strip()]
            if slim:
                ideas_str = "Seed first-30-day ideas (optional, choose at most 1): " + " | ".join(slim[:3])

        length_hint = {
            "short": "Limit to ~120–180 words; 2 paragraphs.",
            "standard": "Target ~180–280 words; 3 paragraphs.",
            "long": "Allow ~300–400 words; 3 paragraphs.",
        }[length]

        sys_prompt = f"""
You will produce TWO outputs in order:

(1) A STRICT JSON object:
{{
 "thesis": "1–2 sentences that name {company} and {role} and a 'why now'",
 "story": "4–6 sentences: challenge → actions (tools) → outcome (numbers)",
 "mapping": ["exact JD responsibility 1", "exact JD responsibility 2", "optional 3"],
 "plan": "1–2 sentences: realistic 30–60 day experiment using their stack",
 "ask": "1 sentence interview ask",
 "keywords_used": ["k1","k2","..."]  // 5–9 items chosen ONLY from the ALLOWED_KEYWORDS list below
}}

(2) Then plain BODY paragraphs only (no bullets, no headings, no code), merging the JSON fields into:
- {('2 paragraphs' if length=='short' else '3 paragraphs')} with the shape:
  P1 thesis, P2 story+mapping, {('P3 omitted for short' if length=='short' else 'P3 plan+ask')}

Rules:
- First-person singular; energetic, not hypey. Plain English. No clichés.
- Ground everything strictly in the JD and resume. Do NOT invent employers, tools, results, or dates.
- Use "and" instead of "&". No em/en dashes; use commas or conjunctions.
- No GPA, course lists, relocation/visa/availability notes.
- Weave exactly 5–9 keywords from ALLOWED_KEYWORDS total, max 2 per sentence.
- Explicitly reuse 2–3 responsibilities from JD_RESPONSIBILITIES in "mapping".
- {length_hint}
- After the JSON, output only the paragraphs (no JSON, no commentary).

ALLOWED_KEYWORDS (prioritize overlap with resume):
{preferred_kw_pool}

JD_RESPONSIBILITIES (pick 2–3 to map):
{jd_resps}

Quality checks (you must satisfy):
- Names company, role, and a “why now” tied to a specific detail if possible.
- One measurable outcome in the story (%, #, $, latency, adoption, risk).
- Keyword total 5–9; ≤2 per sentence.
- Paragraph count as specified; no bullets or step numbers.

{facts_str}
{ideas_str}
""".strip()

        user_prompt = (
            f"JOB DESCRIPTION (<=4k):\n{(jd_text or '')[:4000]}\n\n"
            f"RESUME (raw; may contain LaTeX, <=4k):\n{(resume_text or '')[:4000]}\n\n"
            "Return JSON first, then the body paragraphs."
        )

        draft = await chat_text(sys_prompt, user_prompt, model=_DRAFT_MODEL)

        # remove initial JSON blob if present
        body_v1 = re.sub(r"^\s*\{.*?\}\s*", "", draft.strip(), flags=re.DOTALL)
        body_v1 = _clean_text_local(body_v1, banned_phrases=banned_phrases)

        def _stats(text: str) -> Dict[str, object]:
            kw_counts = _keywords_in_text(text, preferred_kw_pool)
            per_sent = _per_sentence_kw_counts(text, preferred_kw_pool)
            oov = _oov_terms(text, allowed)
            has_detail = _has_specific_company_detail(text, company_facts or [])
            has_why = _has_why_now(text)
            s_lens = _sentence_stats(text)
            return {
                "total_keywords_used": len([k for k,v in kw_counts.items() if v > 0]),
                "kw_counts": kw_counts,
                "kw_per_sentence": per_sent,
                "oov_terms": list(sorted(oov)),
                "has_company_detail": has_detail,
                "has_why_now": has_why,
                "sentence_lengths": s_lens,
                "paragraphs": len([p for p in text.split("\n") if p.strip()]),
            }

        stats = _stats(body_v1)

        async def _repair(reason: str, text_in: str,
                          enforce_add: List[str] = None,
                          enforce_drop: List[str] = None,
                          target_paras: int = 3) -> str:
            enforce_add = enforce_add or []
            enforce_drop = enforce_drop or []
            add_str = ", ".join(enforce_add) if enforce_add else "none"
            drop_str = ", ".join(enforce_drop) if enforce_drop else "none"
            r_prompt = f"""
Rewrite the cover-letter BODY for "{role}" at "{company}" to FIX: {reason}.
Constraints:
- Keep all facts grounded in JD and resume (no new employers/tools/results/dates).
- Keep tone first-person, plain English, no clichés, no bullets or lists.
- Use EXACTLY {_TOTAL_KW_MIN}-{_TOTAL_KW_MAX} keywords total from this list, max {_PER_SENT_KW_MAX} per sentence:
  {preferred_kw_pool}
- Prefer overlap keywords; avoid overstuffing any sentence.
- Paragraphs: {target_paras} total ({'P1 thesis, P2 story+mapping' if target_paras==2 else 'P1 thesis, P2 story+mapping, P3 plan+ask'}).
- Add these keywords if missing (optional): {add_str}
- Remove or replace these terms if present: {drop_str}
Return only the body paragraphs.
"""
            revised = await chat_text(r_prompt, f"JD:\n{(jd_text or '')[:3000]}\n\nResume:\n{(resume_text or '')[:3000]}\n\nDraft:\n{text_in}", model=_DRAFT_MODEL)
            return _clean_text_local(revised, banned_phrases=banned_phrases)

        # OOV grounding
        if stats["oov_terms"]:
            replacements = {}
            for bad in stats["oov_terms"]:
                cand = difflib.get_close_matches(bad.lower(), [t.lower() for t in preferred_kw_pool], n=1, cutoff=0.72)
                if cand:
                    replacements[bad] = cand[0]
            if replacements:
                body_v1 = re.sub(
                    "|".join(map(re.escape, replacements.keys())),
                    lambda m: replacements[m.group(0)],
                    body_v1
                )
            else:
                body_v1 = await _repair("remove or replace out-of-vocabulary tool names with JD/resume tools", body_v1, target_paras=(2 if length=="short" else 3))
            stats = _stats(body_v1)

        # Total keyword count 5–9
        total_kw_used = stats["total_keywords_used"]
        if total_kw_used < _TOTAL_KW_MIN:
            missing = [k for k in preferred_kw_pool if k not in stats["kw_counts"]][:(_TOTAL_KW_MIN - total_kw_used)]
            body_v1 = await _repair(f"raise total distinct keyword count to between {_TOTAL_KW_MIN} and {_TOTAL_KW_MAX}", body_v1, enforce_add=missing, target_paras=(2 if length=="short" else 3))
            stats = _stats(body_v1)
        elif total_kw_used > _TOTAL_KW_MAX:
            extras = sorted(stats["kw_counts"], key=lambda k: (k.lower() not in [s.lower() for s in shared], stats["kw_counts"][k]), reverse=True)
            drop = extras[:max(0, total_kw_used - _TOTAL_KW_MAX)]
            body_v1 = await _repair(f"reduce total distinct keyword count to at most {_TOTAL_KW_MAX}", body_v1, enforce_drop=drop, target_paras=(2 if length=="short" else 3))
            stats = _stats(body_v1)

        # Per-sentence density cap ≤ 2
        if any(c > _PER_SENT_KW_MAX for c in stats["kw_per_sentence"]):
            body_v1 = await _repair(f"reduce keyword density to ≤{_PER_SENT_KW_MAX} per sentence", body_v1, target_paras=(2 if length=="short" else 3))
            stats = _stats(body_v1)

        # Sentence length band
        if any(n < _SENT_MIN or n > _SENT_MAX for n in stats["sentence_lengths"]):
            body_v1 = _enforce_sentence_band(body_v1)
            stats = _stats(body_v1)

        # Specific company detail + why-now
        if not _has_specific_company_detail(body_v1, company_facts or []):
            body_v1 = await _repair("include one specific company detail from facts and tie it to 'why now'", body_v1, target_paras=(2 if length=="short" else 3))
            stats = _stats(body_v1)
        if not _has_why_now(body_v1):
            body_v1 = await _repair("add a concise 'why now' reason anchored to a product, launch, metric, or initiative", body_v1, target_paras=(2 if length=="short" else 3))
            stats = _stats(body_v1)

        # Paragraph count/shape control
        paras_target = 2 if length == "short" else 3
        body_v1 = _shape_paragraphs(body_v1, length)
        par_count = len([p for p in body_v1.split("\n") if p.strip()])
        if par_count != paras_target:
            body_v1 = await _repair(f"ensure exactly {paras_target} paragraphs with the required shape", body_v1, target_paras=paras_target)
            body_v1 = _shape_paragraphs(body_v1, length)

        # Final local post-processing
        body_v1 = _dedupe_verbs_local(body_v1)
        body_v1 = _enforce_word_band_local(body_v1, length)
        body_v1 = re.sub(r"(^|\n)\s*(?:[-*•]|\d+[.)])\s+", " ", body_v1)

        return _postprocess_body(body_v1)

    except Exception as e:
        log_event("coverletter_draft_fail", {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Body generation failed: {e}")


# -----------------------------
# Humanize via internal service
# -----------------------------
async def humanize_text(body_text: str, tone: str) -> str:
    api_base = (getattr(config, "API_BASE_URL", "") or "").rstrip("/") or "http://127.0.0.1:8000"
    url = f"{api_base}/api/superhuman/rewrite"
    payload = {"text": body_text, "mode": "coverletter", "tone": tone, "latex_safe": True}
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("rewritten") or data.get("text") or body_text
    except Exception as e:
        log_event("superhuman_handoff_fail", {"error": str(e)})
        return body_text


# -----------------------------
# Header placeholder filling
# -----------------------------
def _fill_header_fields(
    tex: str,
    *,
    company: str,
    role: str,
    candidate: str,
    date_str: str,
    email: str = "",
    phone: str = "",
    citystate: str = "",
) -> str:
    def esc(v: str) -> str:
        return _latex_escape_light(secure_tex_input(v or ""))

    subst = {
        "COMPANY": company,
        "ROLE": role,
        "CANDIDATE_NAME": candidate,
        "NAME": candidate,
        "DATE": date_str,
        "EMAIL": email,
        "PHONE": phone,
        "CITYSTATE": citystate,
    }
    for k, v in subst.items():
        tex = tex.replace(f"{{{{{k}}}}}", esc(v))
        tex = tex.replace(f"%<<{k}>>%", esc(v))

    patterns = {
        r"(\\def\\Company\{)(.*?)(\})": company,
        r"(\\def\\Role\{)(.*?)(\})": role,
        r"(\\def\\CandidateName\{)(.*?)(\})": candidate,
        r"(\\def\\Date\{)(.*?)(\})": date_str,
    }
    for pat, val in patterns.items():
        tex = re.sub(pat, lambda m: f"{m.group(1)}{esc(val)}{m.group(3)}", tex, flags=re.I)

    tex = re.sub(r"(^|\n)\s*Company\s*$", lambda m: f"{m.group(1)}{esc(company)}", tex, flags=re.M)
    tex = re.sub(r"(^|\n)\s*Your Name\s*$", lambda m: f"{m.group(1)}{esc(candidate)}", tex, flags=re.M)
    return tex


# -----------------------------
# Precise body injection helpers
# -----------------------------
def _inject_between_salutation_and_signoff(base_tex: str, body_tex: str) -> Optional[str]:
    pat = r"(Dear[^\n]*?,\s*\n)([\s\S]*?)(\n\s*Sincerely,\s*\\\\[\s\S]*?$)"
    if re.search(pat, base_tex, flags=re.I):
        return re.sub(pat, lambda m: f"{m.group(1)}{body_tex}\n{m.group(3)}", base_tex, flags=re.I)
    return None


# -----------------------------
# Inject Body into LaTeX Template
# -----------------------------
def inject_body_into_template(base_tex: str, body_tex: str) -> str:
    swapped = _inject_between_salutation_and_signoff(base_tex, body_tex)
    if swapped is not None:
        return swapped

    if _shared_inject and not _DISABLE_SHARED_INJECTOR:
        try:
            return _shared_inject(base_tex, body_tex)
        except Exception as e:  # pragma: no cover
            log_event("shared_inject_fail", {"error": str(e)})

    safe_body = re.sub(r"\\documentclass[\s\S]*?\\begin\{document\}", "", body_tex or "", flags=re.I)
    safe_body = re.sub(r"\\end\{document\}\s*$", "", safe_body, flags=re.I).strip()

    anchor_pat = r"(%-+BODY-START-+%)(.*?)(%-+BODY-END-+%)"
    if re.search(anchor_pat, base_tex, flags=re.S):
        return re.sub(anchor_pat, lambda m: f"{m.group(1)}\n{safe_body}\n{m.group(3)}", base_tex, flags=re.S)

    if re.search(r"\\end\{document\}\s*$", base_tex, flags=re.I):
        return re.sub(
            r"\\end\{document\}\s*$",
            lambda m: f"\n% (Auto-inserted by HIREX)\n{safe_body}\n\\end{{document}}\n",
            base_tex,
            flags=re.I,
        )

    return base_tex.rstrip() + f"\n\n% (Auto-inserted by HIREX)\n{safe_body}\n\\end{{document}}\n"


# -----------------------------
# Main Endpoint
@router.post("")
async def generate_coverletter(
    jd_text: str = Form(...),
    resume_tex: str = Form(""),
    use_humanize: bool = Form(True),
    tone: str = Form("balanced"),
    length: str = Form("standard"),
):
    if not (config.OPENAI_API_KEY or "").strip():
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY missing in environment.")
    if not (jd_text or "").strip():
        raise HTTPException(status_code=400, detail="jd_text is required.")

    company, role = await extract_company_role(jd_text)

    company_slug = safe_filename(company)
    role_slug = safe_filename(role)
    context_key = f"{company_slug}__{role_slug}"

    body_text = await draft_cover_body(
        jd_text, resume_tex, company, role, tone, length
    )

    if use_humanize:
        body_text = await humanize_text(body_text, tone)
        body_text = _postprocess_body(body_text)

    base_path = config.BASE_COVERLETTER_PATH
    try:
        with open(base_path, encoding="utf-8") as f:
            base_tex = f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Base cover-letter template not found at {base_path}")

    today_str = datetime.now().strftime("%B %d, %Y")
    candidate = getattr(config, "CANDIDATE_NAME", "Sri Akash Kadali")
    applicant_email = getattr(config, "APPLICANT_EMAIL", "kadali18@umd.edu")
    applicant_phone = getattr(config, "APPLICANT_PHONE", "+1 240-726-9356")
    applicant_city = getattr(config, "APPLICANT_CITYSTATE", "College Park, Maryland")

    base_tex = _fill_header_fields(
        base_tex,
        company=company,
        role=role,
        candidate=candidate,
        date_str=today_str,
        email=applicant_email,
        phone=applicant_phone,
        citystate=applicant_city,
    )

    try:
        injected = inject_body_into_template(base_tex, body_text)
    except re.error as e:
        log_event("inject_regex_error", {"error": str(e)})
        injected = f"{base_tex}\n\n% (Fallback inject due to regex error)\n{body_text}\n"
        if not injected.strip().endswith("\\end{document}"):
            injected += "\n\\end{document}\n"

    final_tex = render_final_tex(injected)
    pdf_bytes = compile_latex_safely(final_tex) or b""
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    # Save final PDF into samples path per your spec
    out_pdf_path = config.get_sample_coverletter_pdf_path(company, role)
    ensure_dir(out_pdf_path.parent)
    if pdf_bytes:
        out_pdf_path.write_bytes(pdf_bytes)

    # Persist into stable context (dedup by Company__Role)
    ctx_dir = config.get_contexts_dir()
    ensure_dir(ctx_dir)
    ctx_path = ctx_dir / f"{context_key}.json"

    existing: Dict[str, Any] = {}
    if ctx_path.exists():
        try:
            existing = json.loads(ctx_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    context_payload: Dict[str, Any] = {
        **existing,
        "key": context_key,
        "title_for_memory": f"{company_slug}_{role_slug}_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
        "company": company,
        "role": role,
        "jd_text": jd_text or existing.get("jd_text", ""),
        "cover_letter": {
            **(existing.get("cover_letter") or {}),
            "tex": final_tex,
            "pdf_path": str(out_pdf_path),
            "pdf_b64": pdf_b64,
            "tone": tone,
            "length": length,
            "humanized": bool(use_humanize),
        },
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    ctx_path.write_text(json.dumps(context_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    log_event(
        "coverletter_saved",
        {
            "company": company,
            "role": role,
            "tone": tone,
            "use_humanize": use_humanize,
            "length": length,
            "chars": len(body_text or ""),
            "pdf_path": str(out_pdf_path),
            "context_path": str(ctx_path),
            "context_key": context_key,
        },
    )

    return JSONResponse(
        {
            "company": company,
            "role": role,
            "tone": tone,
            "use_humanize": use_humanize,
            "tex_string": final_tex,
            "pdf_base64": pdf_b64,
            "pdf_path": str(out_pdf_path),
            "context_key": context_key,
            "context_path": str(ctx_path),
            "id": context_key,
            "memory_id": context_key,
        }
    )
