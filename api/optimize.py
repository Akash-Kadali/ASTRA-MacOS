"""
Resume optimizer API (FastAPI) — v2.9.2

CHANGES vs v2.9.1:
 FIX    Metric believability: exactly 1 metric per experience block (4 total),
        other 2 bullets per block use qualitative outcomes only.
 FIX    Replaced _odd_int with _casual_int — round-ish, human-sounding numbers.
 FIX    Intern-scale metrics: no K/M/GB, no sub-10ms latency, hedging language
        ("about", "~", "roughly") required on all numbers.
 ADD    verify_metric_believability post-filter: caps metric bullets at 4,
        rewrites unrealistic metrics, adds hedging language.
 UPDATE enforce_metric_diversity simplified for 4-metric-only constraint.
 UPDATE METRIC_TEMPLATES rewritten with intern-scale casual phrasing.
 UPDATE Bullet prompt METRIC RULES: 1-of-3 metric rule, no odd-digit rule.
 UPDATE score_bullet_quality_rubric: non-metric bullets score 3/3 if they
        have qualitative outcomes.
 ALL    All v2.9.2 features preserved (AI-tell filter, structure diversity,
        human voice rubric, JD echo verification).
"""
import base64
import json
import re
import asyncio
import threading
import random as _random
import traceback
import tempfile
import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set, Any

from fastapi import APIRouter, UploadFile, Form, File, HTTPException
from fastapi.responses import JSONResponse

from backend.core import config
from backend.core.compiler import compile_latex_safely
from backend.core.security import secure_tex_input
from backend.core.utils import log_event, safe_filename, build_output_paths
from backend.api.render_tex import render_final_tex

router = APIRouter(prefix="/api/optimize", tags=["optimize"])


# ── Enhanced logging helper ──────────────────────────────────
def _log(msg: str, data: Any = None):
    """Print to terminal AND call log_event so nothing is lost."""
    if data:
        print(f"{msg} :: {data}", flush=True)
    else:
        print(msg, flush=True)
    log_event(msg, **(data if isinstance(data, dict) else {}))

# ── Genuine JD-Optimized Style ───────────────────────────────
GENUINE_JD_OPTIMIZED_STYLE = """
Generate strong JD-targeted resume content, but keep it genuine and interview-defensible.

The output should sound like a real intern or early-career engineer wrote it carefully.
Do not sound like a resume optimizer, marketing copy, or ChatGPT.

Core rules:
- Optimize for the JD through relevant themes, not copied JD phrasing.
- Do not copy wording from the existing resume bullets.
- Use the existing resume only for structure, company names, dates, and broad experience context.
- Prefer newly written bullets grounded in believable intern-level work.
- Every bullet must be explainable in a 2-minute interview answer.
- Use concrete technical artifacts: dashboards, scripts, components, notebooks, models, checks, reports, pipelines, APIs, test cases.
- Avoid vague phrases like "actionable insights," "data-driven experiences," and "stakeholder visibility."
- Avoid hype words and senior-level claims.
"""
# ── OpenAI client ────────────────────────────────────────────
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

_openai_client: Optional["OpenAI"] = None
_openai_lock = threading.Lock()


def get_openai_client() -> "OpenAI":
    global _openai_client
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not available.")
    if _openai_client is None:
        with _openai_lock:
            if _openai_client is None:
                _openai_client = OpenAI(api_key=getattr(config, "OPENAI_API_KEY", ""))
    return _openai_client


# ── GPT helper ───────────────────────────────────────────────

def _json_from_text(text: str, default: Any):
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return default
    try:
        return json.loads(m.group(0))
    except Exception:
        return default


async def gpt_json(
    prompt: str, temperature: float = 0.0, model: str = "gpt-5.4-nano",
) -> dict:
    client = get_openai_client()
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "timeout": 120,
    }
    try:
        kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
    except TypeError:
        kwargs.pop("response_format", None)
        resp = client.chat.completions.create(**kwargs)
    content = (resp.choices[0].message.content or "").strip()
    return _json_from_text(content or "{}", {})

# ── Candidate Capability Inventory ───────────────────────────
async def build_candidate_capability_inventory(
    original_tex: str,
    target_role: str,
    jd_text: str,
) -> Dict[str, Any]:
    prompt = f"""
You are building a candidate capability inventory for JD-targeted resume generation.

Use the existing resume only to understand:
- companies/institutions
- dates
- broad role history
- likely tools and domains
- projects and achievements

Do NOT copy existing bullet wording.
Do NOT reuse suspicious phrases from the resume.
Do NOT preserve weak or awkward bullets.
Do NOT include experience company names in any bullet content.
Bullets should describe WHAT was done, not WHERE.

Create a compact inventory of what the candidate can plausibly defend in interviews.

TARGET ROLE:
{target_role}

JOB DESCRIPTION:
{jd_text[:3500]}

EXISTING RESUME / LATEX:
{original_tex[:7000]}

Return STRICT JSON:
{{
  "candidate_level": "intern|new_grad|early_career",
  "defensible_tools": ["..."],
  "defensible_domains": ["..."],
  "defensible_project_types": ["..."],
  "experience_inventory": [
    {{
      "company": "...",
      "role_context": "...",
      "likely_work": ["..."],
      "defensible_tools": ["..."],
      "safe_metrics": ["..."],
      "avoid_claims": ["..."]
    }}
  ],
  "jd_overlap": {{
    "strong_overlap": ["..."],
    "moderate_overlap": ["..."],
    "risky_or_unsupported": ["..."]
  }},
  "global_avoid_phrases": [
    "actionable insights",
    "data-driven experiences",
    "T3",
    "T4",
    "Dell T"
  ]
}}
"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        _log(f"⚠️ [CAPABILITY INVENTORY] Failed: {e}")
        return {}
    
# ═══════════════════════════════════════════════════════════════
# ROLE ARCHETYPE + TONE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

ROLE_ARCHETYPES = {
    "software_engineer": {
        "name": "Software Engineer",
        "bullet_focus": "systems built, scale handled, reliability, performance",
        "phrasing_style": "Plain professional bullets focused on concrete work and believable outcomes",
        "result_types": "latency reduced, uptime improved, throughput increased",
        "typical_verbs": ["Built", "Wrote", "Set up", "Shipped", "Fixed"],
        "avoid": "ML-specific jargon unless JD mentions it",
    },
    "data_scientist": {
        "name": "Data Scientist",
        "bullet_focus": "hypotheses tested, insights found, models validated, business impact", 
        "phrasing_style": "Plain professional bullets focused on concrete work and believable outcomes",
        "result_types": "insight discovered, model accuracy, business metric moved",
        "typical_verbs": ["Dug into", "Found", "Trained", "Plotted", "Cleaned"],
        "avoid": "pure engineering language, deployment details unless relevant",
    },
    "ml_engineer": {
        "name": "ML Engineer",
        "bullet_focus": "pipelines built, models deployed, latency/throughput, training infra",
        "phrasing_style": "Plain professional bullets focused on concrete work and believable outcomes",
        "result_types": "inference latency, model serving throughput, pipeline reliability",
        "typical_verbs": ["Built", "Trained", "Shipped", "Sped up", "Scripted"],
        "avoid": "pure research language, business analysis language",
    },
    "cloud_infrastructure": {
        "name": "Cloud / Infrastructure Engineer",
        "bullet_focus": "infra provisioned, automation, cost optimization, reliability",
        "phrasing_style": "Plain professional bullets focused on concrete work and believable outcomes",
        "result_types": "cost reduced, provisioning time cut, reliability improved",
        "typical_verbs": ["Set up", "Scripted", "Shipped", "Moved", "Fixed"],
        "avoid": "ML model details, data science language",
    },
    "research": {
        "name": "Research Scientist / Engineer",
        "bullet_focus": "novel methods, ablations, benchmarks, publications",
        "phrasing_style": "Plain professional bullets focused on concrete work and believable outcomes",
        "result_types": "benchmark improved, novel approach validated, ablation completed",
        "typical_verbs": ["Tested", "Tried", "Built", "Found", "Wrote up"],
        "avoid": "production/deployment language, business metrics",
    },
    "data_engineer": {
        "name": "Data Engineer",
        "bullet_focus": "pipelines built, data quality, throughput, schema design",
        "phrasing_style": "Plain professional bullets focused on concrete work and believable outcomes",
        "result_types": "data freshness, pipeline throughput, data quality",
        "typical_verbs": ["Built", "Pulled", "Scripted", "Sped up", "Fixed"],
        "avoid": "ML model training details, research language",
    },
    "general_tech": {
        "name": "General Technical Role",
        "bullet_focus": "problems solved, systems built, efficiency gained",
        "phrasing_style": "Plain professional bullets focused on concrete work and believable outcomes",
        "result_types": "efficiency gained, problem solved, process improved",
        "typical_verbs": ["Built", "Wrote", "Fixed", "Scripted", "Sped up"],
        "avoid": "nothing specific",
    },
}


async def classify_role_and_tone(jd_text: str, target_role: str) -> Dict[str, Any]:
    prompt = f"""Classify this job and analyze its writing tone.

JOB TITLE: {target_role}

JOB DESCRIPTION (first 2500 chars):
{jd_text[:2500]}

Return STRICT JSON:
{{
    "archetype": "software_engineer|data_scientist|ml_engineer|cloud_infrastructure|research|data_engineer|general_tech",
    "confidence": 0.85,
    "reasoning": "1-2 sentences why this archetype fits",
    "tone": {{
        "register": "formal|casual|technical|business|academic",
        "pace": "fast-paced startup language|measured enterprise language|academic careful language",
        "vocabulary_style": "uses buzzwords freely|precise technical terms|plain language",
        "example_phrases": ["3-4 characteristic phrases from the JD that reveal its tone"]
    }}
}}
"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        key = data.get("archetype", "general_tech")
        if key not in ROLE_ARCHETYPES:
            key = "general_tech"
        tone = data.get("tone", {})
        result = {
            "key": key,
            **ROLE_ARCHETYPES[key],
            "confidence": data.get("confidence", 0.5),
            "reasoning": data.get("reasoning", ""),
            "tone_register": tone.get("register", "technical"),
            "tone_pace": tone.get("pace", "measured"),
            "tone_vocabulary": tone.get("vocabulary_style", "precise technical terms"),
            "tone_examples": tone.get("example_phrases", []),
        }
        _log(f"🎭 [ROLE+TONE] {target_role} → {key}, tone={result['tone_register']}/{result['tone_pace']}")
        return result
    except Exception as e:
        _log(f"⚠️ [ROLE+TONE] Failed: {e}")
        return {"key": "general_tech", **ROLE_ARCHETYPES["general_tech"],
                "tone_register": "technical", "tone_pace": "measured",
                "tone_vocabulary": "precise", "tone_examples": []}


# ═══════════════════════════════════════════════════════════════
# EXPERIENCE TITLE REWRITING — v2.8.1 UPDATED
# ═══════════════════════════════════════════════════════════════

ALLOWED_INTERN_TITLES = [
    "Machine Learning Intern",
    "Software Engineer Intern",
    "AI Engineer Intern",
    "Data Science Intern",
]

_ARCHETYPE_TO_INTERN_TITLE = {
    "software_engineer": "Software Engineer Intern",
    "data_scientist": "Data Science Intern",
    "ml_engineer": "Machine Learning Intern",
    "cloud_infrastructure": "Software Engineer Intern",
    "research": "AI Engineer Intern",
    "data_engineer": "Data Science Intern",
    "general_tech": "Software Engineer Intern",
}

async def classify_best_intern_title(
    jd_text: str, target_role: str, role_archetype: Dict[str, Any],
) -> str:
    archetype_key = role_archetype.get("key", "general_tech")

    prompt = f"""You are a resume strategist. A candidate is applying for:
ROLE: {target_role}

JOB DESCRIPTION (first 2500 chars):
{jd_text[:2500]}

The candidate's previous internship titles need to be rewritten to best match this JD.
You MUST choose EXACTLY one of these four titles VERBATIM — do not modify, combine, or invent new titles:

1. "Machine Learning Intern" — best when JD focuses on: model training, ML pipelines,
   NLP, computer vision, deep learning, model evaluation, feature engineering,
   recommendation systems, or ML research.

2. "Software Engineer Intern" — best when JD focuses on: backend/frontend development,
   system design, APIs, microservices, databases, cloud infrastructure, DevOps, CI/CD,
   distributed systems, web applications, mobile development, or general programming.

3. "AI Engineer Intern" — best when JD focuses on: LLM applications, AI agents, RAG pipelines,
   prompt engineering, generative AI, AI infrastructure, fine-tuning foundation models,
   AI product development, or applied AI systems that combine ML with engineering.

4. "Data Science Intern" — best when JD focuses on: data analysis, A/B testing,
   experimentation, statistical modeling, analytics, business intelligence, data pipelines,
   SQL-heavy work, dashboards, reporting, data visualization, or exploratory data analysis.

CRITICAL: Your chosen_title MUST be one of these EXACT strings, copied character-for-character:
- "Machine Learning Intern"
- "Software Engineer Intern"
- "AI Engineer Intern"
- "Data Science Intern"
Do NOT add words like "Research", "Engineering", "Senior", etc. Pick the closest match from the 4 options above.

Return STRICT JSON:
{{
    "chosen_title": "Machine Learning Intern|Software Engineer Intern|AI Engineer Intern|Data Science Intern",
    "reasoning": "1-2 sentences explaining why this title is the best match",
    "jd_signals": ["3-5 specific phrases from the JD that drove this decision"]
}}
"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        chosen = data.get("chosen_title", "").strip()
        reasoning = data.get("reasoning", "")

        if chosen not in ALLOWED_INTERN_TITLES:
            chosen_lower = chosen.lower()
            _log(f"   ⚠️ [TITLE] GPT returned non-allowed title: '{chosen}', mapping to closest match")
            if "data scien" in chosen_lower or "data analy" in chosen_lower or "analytics" in chosen_lower:
                chosen = "Data Science Intern"
            elif "ai " in chosen_lower or "artificial intelligence" in chosen_lower:
                chosen = "AI Engineer Intern"
            elif "software" in chosen_lower or "swe" in chosen_lower:
                chosen = "Software Engineer Intern"
            elif "machine learning" in chosen_lower or "ml " in chosen_lower:
                chosen = "Machine Learning Intern"
            else:
                chosen = _ARCHETYPE_TO_INTERN_TITLE.get(archetype_key, "Software Engineer Intern")

        _log(f"🏷️ [TITLE] JD best-fit intern title: '{chosen}' — {reasoning}")
        return chosen

    except Exception as e:
        _log(f"⚠️ [TITLE] GPT classification failed: {e}, using archetype fallback")
        return _ARCHETYPE_TO_INTERN_TITLE.get(archetype_key, "Software Engineer Intern")


def _find_experience_role_positions(tex: str) -> List[Dict[str, Any]]:
    exp_pat = section_rx("Experience")
    m = exp_pat.search(tex)
    if not m:
        _log("⚠️ [TITLE] No Experience section found")
        return []

    section_start = m.start()
    section_text = m.group(1)

    subheading_pat = re.compile(r"\\resumeSubheading\s*")
    results = []

    for sh_match in subheading_pat.finditer(section_text):
        sh_end = sh_match.end()

        brace_groups = []
        pos = sh_end
        for _ in range(4):
            while pos < len(section_text) and section_text[pos] in " \t\n\r":
                pos += 1
            if pos >= len(section_text) or section_text[pos] != "{":
                break
            depth = 0
            start_pos = pos
            while pos < len(section_text):
                if section_text[pos] == "{":
                    depth += 1
                elif section_text[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        brace_groups.append({
                            "start": start_pos,
                            "end": pos + 1,
                            "content": section_text[start_pos + 1:pos],
                        })
                        pos += 1
                        break
                pos += 1

        if len(brace_groups) < 4:
            continue

        role_group_idx = None
        for idx, bg in enumerate(brace_groups):
            content_lower = bg["content"].lower()
            if "intern" in content_lower:
                role_group_idx = idx
                break

        if role_group_idx is None:
            role_keywords = ["engineer", "developer", "analyst", "scientist",
                             "assistant", "associate", "fellow", "researcher",
                             "architect", "designer", "specialist", "coordinator",
                             "programmer", "technician"]
            for idx, bg in enumerate(brace_groups):
                content_lower = bg["content"].lower()
                if any(kw in content_lower for kw in role_keywords):
                    role_group_idx = idx
                    break

        if role_group_idx is None:
            role_group_idx = 0

        company_name = ""
        for idx, bg in enumerate(brace_groups):
            if idx == role_group_idx:
                continue
            content_lower = bg["content"].lower()
            if any(kw in content_lower for kw in ["iit", "nit", "institute", "university", "lab"]):
                company_name = bg["content"]
                break
        if not company_name:
            for idx, bg in enumerate(brace_groups):
                if idx == role_group_idx:
                    continue
                content_lower = bg["content"].lower()
                if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|20\d{2})\b", content_lower):
                    continue
                if re.search(r"\b[A-Z]{2}\b", bg["content"]) and "," in bg["content"]:
                    continue
                company_name = bg["content"]
                break

        rg = brace_groups[role_group_idx]
        results.append({
            "absolute_start": section_start + rg["start"],
            "absolute_end": section_start + rg["end"],
            "content_start": section_start + rg["start"] + 1,
            "content_end": section_start + rg["end"] - 1,
            "original_title": rg["content"],
            "group_index": role_group_idx,
            "company_name": company_name,
        })

    _log(f"🏷️ [TITLE] Found {len(results)} experience role positions")
    for r in results:
        _log(f"   → '{r['original_title']}' @ company='{r['company_name']}'")

    return results

SAFE_TITLE_MAP = {
    "software engineer intern": "Software Engineer Intern",
    "software engineering intern": "Software Engineering Intern",
    "machine learning engineer intern": "Machine Learning Engineer Intern",
    "machine learning intern": "Machine Learning Intern",
    "research intern": "Research Intern",
    "ai/ml intern": "AI/ML Intern",
    "ai engineer intern": "AI Engineer Intern",
    "data science intern": "Data Science Intern",
    "data analyst intern": "Data Analyst Intern",
}


def normalize_safe_intern_title(original: str) -> str:
    cleaned = re.sub(r"\s+", " ", original or "").strip()
    key = cleaned.lower()
    return SAFE_TITLE_MAP.get(key, cleaned)

LOCKED_TITLES = {
    "ayar labs": "Machine Learning Engineer Intern",
}

def rewrite_experience_titles_per_block_v2(
    tex: str, jd_title: str, experience_companies: List[str],
) -> str:
    positions = _find_experience_role_positions(tex)
    if not positions:
        _log("⚠️ [TITLE] No role positions found to rewrite")
        return tex

    for block_idx, pos_info in enumerate(reversed(positions)):
        actual_idx = len(positions) - 1 - block_idx
        original = pos_info["original_title"]
        cs = pos_info["content_start"]
        ce = pos_info["content_end"]
        company = pos_info.get("company_name", "")

        original_lower = original.lower().strip()
        if re.search(r"\b[A-Z]{2}\b", original) and "," in original:
            _log(f"   ⏭️ Skipping '{original}' — looks like a location")
            continue
        if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|20\d{2})\b",
                      original_lower):
            _log(f"   ⏭️ Skipping '{original}' — looks like a date")
            continue

        company_lower = company.lower().strip()
        if company_lower in LOCKED_TITLES:
            new_title = LOCKED_TITLES[company_lower]
            _log(f"   🔒 [TITLE] Locked: {company} → {new_title}")
        else:
            new_title = jd_title
            _log(f"   🏷️ [TITLE] JD-matched rewrite: '{original}' → '{new_title}'")

        new_title_escaped = latex_escape_for_macro_arg(new_title)
        tex = tex[:cs] + new_title_escaped + tex[ce:]

    return tex

async def rewrite_experience_titles_per_block(
    tex: str, jd_text: str, target_role: str, role_archetype: Dict[str, Any],
    jd_tasks: List[Dict[str, Any]],
) -> str:
    jd_title = await classify_best_intern_title(jd_text, target_role, role_archetype)
    exp_companies = await _extract_experience_companies(tex)
    tex = rewrite_experience_titles_per_block_v2(tex, jd_title, exp_companies)
    _log(f"✅ [TITLE] Titles normalized (locked titles preserved, others kept/cleaned)")
    return tex


# ═══════════════════════════════════════════════════════════════
# JD TASK DECOMPOSITION
# ═══════════════════════════════════════════════════════════════

async def decompose_jd_into_tasks(
    jd_text: str, target_company: str, target_role: str,
    role_archetype: Dict[str, Any],
) -> List[Dict[str, Any]]:
    archetype_key = role_archetype.get("key", "general_tech")
    archetype_focus = role_archetype.get("bullet_focus", "problems solved")

    prompt = f"""You are a hiring manager at {target_company} for {target_role}.
Extract the SPECIFIC TASKS this person will do day-to-day.

Use JD tasks to understand what the role values, not to copy wording into the resume.
Extract practical day-to-day work themes and required capabilities.

JOB DESCRIPTION:
{jd_text[:3500]}

ROLE ARCHETYPE: {archetype_key} — focuses on: {archetype_focus}

Extract tasks at a SPECIFIC level. "Build ML models" is too vague —
extract "Train and deploy a transformer-based NER model for extracting entities from
clinical text" if that's what the JD actually describes.

Return STRICT JSON:
{{
    "tasks": [
        {{
            "task_id": 1,
            "task_description": "Concrete task from the JD, paraphrased naturally",
            "task_category": "build_system|analyze_data|train_model|deploy_service|build_pipeline|optimize_performance|write_tests|design_architecture|automate_process|collaborate|research|monitor",
            "implied_technologies": ["2-3 specific technologies"],
            "resume_angle": "How an intern-level candidate could credibly show related experience",
            "priority": "high|medium|low",
            "ats_keywords": ["short keywords only"],
            "risk_if_forced": "What would sound fake if exaggerated"
        }},
        ... (10-12 tasks)
    ],
    "role_summary": "2-3 sentence summary",
    "domain_context": "industry/domain context"
}}

RULES:
- Tasks are WORK ACTIVITIES, not skill names
- At least 3 tasks must be "high" priority
- task_category MUST be one of the listed values
"""
    try:
        data = await gpt_json(prompt, temperature=0.2)
        tasks = []
        for t in (data.get("tasks", []) or [])[:12]:
            if isinstance(t, dict) and t.get("task_description"):
                tasks.append({
                    "task_id": t.get("task_id", len(tasks) + 1),
                    "task_description": str(t["task_description"]).strip(),
                    "task_category": t.get("task_category", "build_system"),
                    "implied_technologies": t.get("implied_technologies", [])[:3],
                    "what_good_looks_like": t.get("resume_angle", t.get("what_good_looks_like", "")),
                    "priority": t.get("priority", "medium"),
                    "key_jd_phrases": t.get("ats_keywords", t.get("key_jd_phrases", []))[:5],
                    "exact_jd_sentence": "",
                    "ats_keywords_in_task": t.get("ats_keywords", t.get("ats_keywords_in_task", []))[:5],
                    "ideal_bullet_template": t.get("resume_angle", t.get("ideal_bullet_template", "")),
                    "risk_if_forced": t.get("risk_if_forced", ""),
                })
        _log(f"📋 [JD TASKS] {len(tasks)} tasks extracted")
        return tasks
    except Exception as e:
        _log(f"⚠️ [JD TASKS] Failed: {e}")
        return [{"task_id": i + 1, "task_description": f"Task {i + 1}",
                 "task_category": "build_system", "implied_technologies": ["Python"],
                 "what_good_looks_like": "", "priority": "medium", "key_jd_phrases": [],
                 "exact_jd_sentence": "", "ats_keywords_in_task": [],
                 "ideal_bullet_template": "", "risk_if_forced": ""}
                for i in range(6)]


# ═══════════════════════════════════════════════════════════════
# JD KEY PHRASE EXTRACTION
# ═══════════════════════════════════════════════════════════════

async def extract_jd_key_phrases(jd_text: str) -> List[str]:
    prompt = f"""
Extract JD themes for natural resume alignment.

Do NOT extract phrases for keyword stuffing.
Do NOT require exact JD wording in bullets.

JOB DESCRIPTION:
{jd_text[:3000]}

Return STRICT JSON:
{{
  "jd_themes": [
    {{
      "theme": "short natural theme",
      "related_keywords": ["keyword1", "keyword2"],
      "use_naturally_as": "plain-language way to reflect this in a resume bullet"
    }}
  ]
}}

Rules:
- Extract 10-15 themes.
- Prefer day-to-day work themes over generic skills.
- Use these themes to guide bullet direction, not to copy JD language.
- Avoid long exact JD phrases.
"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        # Support both new jd_themes format and old key_phrases for backward compat
        themes = data.get("jd_themes", [])
        if themes and isinstance(themes, list):
            phrases = []
            for t in themes:
                if isinstance(t, dict):
                    theme = str(t.get("theme", "")).strip().lower()
                    if theme:
                        phrases.append(theme)
                    for kw in (t.get("related_keywords", []) or []):
                        kw_s = str(kw).strip().lower()
                        if kw_s and kw_s not in phrases:
                            phrases.append(kw_s)
                elif isinstance(t, str):
                    phrases.append(t.strip().lower())
            _log(f"🔤 [JD THEMES] Extracted {len(phrases)} theme keywords")
            return phrases[:25]
        # Fallback to old format
        phrases = [str(p).strip().lower() for p in (data.get("key_phrases", []) or []) if str(p).strip()]
        _log(f"🔤 [JD PHRASES] Extracted {len(phrases)} key phrases (legacy format)")
        return phrases[:25]
    except Exception:
        return []


def check_phrase_coverage(tex_content: str, key_phrases: List[str]) -> Tuple[List[str], List[str]]:
    plain = strip_all_macros_keep_text(tex_content).lower()
    present = [p for p in key_phrases if p in plain]
    missing = [p for p in key_phrases if p not in plain]
    return present, missing


# ═══════════════════════════════════════════════════════════════
# ACTION VERB MANAGEMENT
# ═══════════════════════════════════════════════════════════════

HUMAN_ACTION_VERBS = {
    "ml_engineer": {
        "build": ["Built", "Created", "Wrote", "Implemented", "Developed", "Assembled"],
        "train": ["Trained", "Fine-tuned", "Retrained", "Tuned", "Calibrated", "Validated"],
        "optimize": ["Improved", "Reduced", "Trimmed", "Tuned", "Compressed", "Accelerated"],
        "deploy": ["Deployed", "Released", "Rolled out", "Launched", "Moved to production"],
        "analyze": ["Analyzed", "Profiled", "Investigated", "Traced", "Identified", "Diagnosed"],
        "automate": ["Automated", "Scripted", "Scheduled", "Integrated", "Replaced manual work"],
        "debug": ["Fixed", "Debugged", "Resolved", "Patched", "Diagnosed", "Cleaned up"],
        "data": ["Cleaned", "Pulled", "Merged", "Parsed", "Normalized", "Reformatted", "Joined", "Filtered"],
        "research": ["Tested", "Compared", "Benchmarked", "Evaluated", "Measured", "Explored"],
        "collaborate": ["Worked with", "Coordinated with", "Supported", "Partnered with", "Reviewed with"],
        "monitor": ["Tracked", "Monitored", "Logged", "Checked", "Flagged"],
        "document": ["Documented", "Summarized", "Outlined", "Recorded", "Mapped out"],
        "refactor": ["Refactored", "Rewrote", "Reworked", "Restructured", "Simplified", "Cleaned up"],
        "integrate": ["Integrated", "Connected", "Linked", "Merged", "Piped into"],
        "scale": ["Scaled", "Expanded", "Extended", "Increased", "Improved"],
    },

    "software_engineer": {
        "build": ["Built", "Wrote", "Created", "Implemented", "Developed", "Coded"],
        "optimize": ["Improved", "Reduced", "Tuned", "Trimmed", "Accelerated"],
        "deploy": ["Deployed", "Released", "Rolled out", "Launched", "Pushed"],
        "debug": ["Fixed", "Debugged", "Resolved", "Patched", "Diagnosed", "Cleaned up"],
        "automate": ["Automated", "Scripted", "Scheduled", "Integrated", "Replaced manual steps"],
        "refactor": ["Refactored", "Rewrote", "Reworked", "Simplified", "Restructured", "Cleaned up"],
        "test": ["Tested", "Validated", "Verified", "Stress-tested", "Load-tested", "Wrote tests for"],
        "data": ["Queried", "Pulled", "Cleaned", "Migrated", "Loaded", "Replicated"],
        "collaborate": ["Worked with", "Reviewed", "Coordinated with", "Supported", "Co-authored"],
        "integrate": ["Integrated", "Connected", "Linked", "Merged", "Piped into"],
        "document": ["Documented", "Summarized", "Outlined", "Recorded", "Commented", "Annotated"],
        "monitor": ["Tracked", "Monitored", "Logged", "Checked", "Flagged"],
    },

    "data_scientist": {
        "analyze": ["Analyzed", "Investigated", "Examined", "Studied", "Explored", "Identified"],
        "build": ["Built", "Wrote", "Created", "Implemented", "Developed", "Assembled"],
        "train": ["Trained", "Fit", "Fine-tuned", "Tuned", "Validated", "Calibrated"],
        "visualize": ["Plotted", "Charted", "Graphed", "Visualized", "Presented"],
        "data": ["Cleaned", "Pulled", "Merged", "Parsed", "Filtered", "Joined", "Normalized", "Reformatted"],
        "communicate": ["Presented", "Shared", "Explained", "Reported", "Briefed", "Summarized"],
        "optimize": ["Improved", "Reduced", "Raised", "Trimmed", "Tuned"],
        "automate": ["Automated", "Scripted", "Scheduled", "Built", "Integrated"],
        "research": ["Tested", "Compared", "Benchmarked", "Evaluated", "Measured", "Explored"],
        "collaborate": ["Worked with", "Partnered with", "Coordinated with", "Supported", "Reviewed with"],
    },

    "research": {
        "experiment": ["Tested", "Compared", "Measured", "Benchmarked", "Evaluated", "Validated", "Replicated"],
        "build": ["Built", "Wrote", "Implemented", "Created", "Developed", "Prototyped"],
        "analyze": ["Analyzed", "Investigated", "Examined", "Studied", "Traced", "Identified"],
        "optimize": ["Improved", "Reduced", "Simplified", "Tuned", "Accelerated"],
        "document": ["Documented", "Drafted", "Summarized", "Outlined", "Recorded", "Reported", "Presented"],
        "data": ["Cleaned", "Curated", "Labeled", "Annotated", "Collected", "Gathered", "Filtered", "Organized"],
        "collaborate": ["Worked with", "Discussed with", "Presented to", "Reviewed with", "Co-authored with"],
    },

    "data_engineer": {
        "build": ["Built", "Created", "Wrote", "Implemented", "Developed", "Assembled"],
        "data": ["Pulled", "Moved", "Migrated", "Loaded", "Synced", "Replicated", "Ingested", "Streamed"],
        "optimize": ["Improved", "Reduced", "Tuned", "Trimmed", "Accelerated"],
        "automate": ["Automated", "Scripted", "Scheduled", "Integrated", "Replaced manual steps"],
        "monitor": ["Tracked", "Monitored", "Logged", "Checked", "Flagged"],
        "debug": ["Fixed", "Debugged", "Resolved", "Patched", "Diagnosed", "Cleaned up"],
        "collaborate": ["Worked with", "Coordinated with", "Supported", "Partnered with", "Reviewed with"],
        "refactor": ["Refactored", "Rewrote", "Reworked", "Restructured", "Simplified", "Cleaned up"],
    },

    "cloud_infrastructure": {
        "build": ["Built", "Configured", "Provisioned", "Created", "Implemented", "Deployed"],
        "automate": ["Automated", "Scripted", "Scheduled", "Integrated", "Replaced manual steps"],
        "deploy": ["Deployed", "Released", "Rolled out", "Launched", "Pushed"],
        "optimize": ["Improved", "Reduced", "Right-sized", "Tuned", "Trimmed"],
        "monitor": ["Tracked", "Monitored", "Logged", "Checked", "Flagged"],
        "debug": ["Fixed", "Debugged", "Resolved", "Patched", "Diagnosed", "Cleaned up"],
        "migrate": ["Migrated", "Moved", "Transferred", "Ported", "Converted", "Transitioned"],
        "collaborate": ["Worked with", "Coordinated with", "Supported", "Partnered with", "Reviewed with"],
    },

    "general_tech": {
        "build": ["Built", "Wrote", "Created", "Implemented", "Developed", "Coded"],
        "optimize": ["Improved", "Reduced", "Tuned", "Trimmed", "Accelerated"],
        "debug": ["Fixed", "Debugged", "Resolved", "Patched", "Diagnosed", "Cleaned up"],
        "automate": ["Automated", "Scripted", "Scheduled", "Integrated", "Replaced manual work"],
        "data": ["Pulled", "Cleaned", "Merged", "Parsed", "Moved", "Loaded", "Filtered", "Normalized"],
        "deploy": ["Deployed", "Released", "Rolled out", "Launched", "Pushed"],
        "collaborate": ["Worked with", "Coordinated with", "Supported", "Reviewed with", "Partnered with"],
        "document": ["Documented", "Outlined", "Summarized", "Recorded", "Commented"],
        "research": ["Tested", "Compared", "Explored", "Evaluated", "Measured", "Benchmarked"],
    },
}
TASK_CAT_TO_VERB_CAT = {
    "build_system": "build", "analyze_data": "analyze",
    "train_model": "train", "deploy_service": "deploy",
    "build_pipeline": "build", "optimize_performance": "optimize",
    "write_tests": "build", "design_architecture": "build",
    "automate_process": "automate", "collaborate": "collaborate",
    "research": "research", "monitor": "monitor",
}

SAFE_INTERN_VERBS = [
    "Built", "Rebuilt", "Created", "Wrote", "Updated",
    "Improved", "Fixed", "Tested", "Validated", "Analyzed",
    "Cleaned", "Connected", "Documented", "Reviewed", "Supported",
]

BANNED_VERBS = {
    "assembled", "spearheaded", "leveraged", "utilized", "orchestrated",
    "pioneered", "championed", "harnessed", "galvanized",
    "operationalized", "productionized", "architected",
}


def clean_action_verb_pool():
    for archetype, groups in HUMAN_ACTION_VERBS.items():
        for category, verbs in groups.items():
            cleaned = []
            for v in verbs:
                if v.lower() in BANNED_VERBS:
                    continue
                cleaned.append("Rebuilt" if v == "Assembled" else v)
            groups[category] = list(dict.fromkeys(cleaned))


# Clean on module load
clean_action_verb_pool()
_used_verbs_global: Set[str] = set()


def reset_verb_tracking():
    global _used_verbs_global
    _used_verbs_global.clear()

# ═══════════════════════════════════════════════════════════════
# STRUCTURE DIVERSITY TRACKER
# ═══════════════════════════════════════════════════════════════

_used_structures: List[str] = []


def reset_structure_tracking():
    global _used_structures
    _used_structures.clear()


def validate_structures(structures: List[str], block_index: int) -> bool:
    """Validate structure choices for a block of 3 bullets.
    Returns True if valid, False if needs regeneration.
    """
    global _used_structures
    # Rule 1: No two bullets in same block use the same structure
    if len(structures) != len(set(structures)):
        _log(f"⚠️ [STRUCTURE] Block {block_index}: duplicate structures within block: {structures}")
        return False
    # Rule 2: No structure appears more than twice across all 12 bullets
    from collections import Counter
    projected = _used_structures + structures
    counts = Counter(projected)
    for struct, count in counts.items():
        if count > 2:
            _log(f"⚠️ [STRUCTURE] Block {block_index}: structure '{struct}' would appear {count} times (max 2)")
            return False
    return True


def record_structures(structures: List[str]):
    """Record structures used after validation passes."""
    global _used_structures
    _used_structures.extend(structures)
    _log(f"📐 [STRUCTURE] Recorded: {structures} (total: {len(_used_structures)})")

_current_archetype_key: str = "general_tech"


def set_current_archetype(key: str):
    global _current_archetype_key
    _current_archetype_key = key if key in HUMAN_ACTION_VERBS else "general_tech"


def get_diverse_verb(category: str, fallback: str = "Built") -> str:
    global _used_verbs_global, _current_archetype_key
    archetype_verbs = HUMAN_ACTION_VERBS.get(_current_archetype_key, HUMAN_ACTION_VERBS["general_tech"])
    verbs = archetype_verbs.get(category, archetype_verbs.get("build", []))
    available = [v for v in verbs if v.lower() not in _used_verbs_global]
    if not available:
        # Try other sub-categories in same archetype
        for sub_cat, sub_verbs in archetype_verbs.items():
            if sub_cat == category:
                continue
            available = [v for v in sub_verbs if v.lower() not in _used_verbs_global]
            if available:
                break
    if not available:
        # Last resort: all verbs across all archetypes
        all_v = [v for arch in HUMAN_ACTION_VERBS.values() for sub in arch.values() for v in sub]
        available = [v for v in all_v if v.lower() not in _used_verbs_global]
    chosen = _random.choice(available) if available else fallback
    _used_verbs_global.add(chosen.lower())
    return chosen


# ═══════════════════════════════════════════════════════════════
# INTERN-SCALE METRIC HELPERS
# ═══════════════════════════════════════════════════════════════

def _casual_int(lo: int, hi: int) -> int:
    """Return a round-ish, human-sounding number. No odd-digit rule."""
    candidates = []
    for x in range(lo, hi + 1):
        if x <= 10 or x % 5 == 0 or x % 10 == 0:
            candidates.append(x)
    if not candidates:
        candidates = list(range(lo, hi + 1))
    return _random.choice(candidates)

METRIC_RULES = """
Metrics are optional and should be rare.
Use at most one metric per experience block.
Use at most four metrics across the full resume.
Use a metric only when it sounds believable and intern-scale.
Use approximate wording: about, roughly, ~, around, under.
Avoid exact suspicious numbers, huge scale, sub-10ms latency, and production-scale claims.
If no believable metric fits, use a qualitative outcome.
"""

METRIC_TEMPLATES: Dict[str, List[str]] = {
    "build_system": [
        "saved about {n} hours of repeated setup work each week",
        "reduced the update process from roughly {a} steps to {b}",
        "made about {n} dashboard views easier to maintain",
    ],
    "analyze_data": [
        "found about {n} reporting mismatches during review",
        "checked roughly {n} records across {m} files",
        "cut review time from about {a} hours to under {b}",
    ],
    "train_model": [
        "tested about {n} model variants during ablation",
        "reduced experiment review time from roughly {a} hours to under {b}",
        "compared results across {m} baseline settings",
    ],
    "deploy_service": [
        "cut cold-start time from about {a} seconds to under {b}",
        "brought rollback time from roughly {a} minutes to under {b}",
        "reduced failed deployments from about {a} per month to {b}",
    ],
    "build_pipeline": [
        "cut pipeline runtime from about {a} hours to roughly {b} minutes",
        "reduced manual intervention from about {a} times a week to {b}",
        "processed roughly {n} files daily without data loss",
    ],
    "optimize_performance": [
        "cut batch processing from about {a} minutes to under {b}",
        "sped things up about {n}x compared to the old approach",
        "got response time from roughly {a} seconds to about {b}",
    ],
    "automate_process": [
        "saved the team about {n} hours a week",
        "replaced {a} manual steps with {b} automated ones",
        "automated roughly {n} previously manual checks per sprint",
    ],
    "research": [
        "tested about {n} architectural variants in one sweep",
        "validated across {n} experimental conditions",
        "compared results across {m} baseline settings",
    ],
    "monitor": [
        "cut mean detection time from about {a} minutes to roughly {b}",
        "tracked about {n} metrics across {m} dashboards",
        "caught {n} drift incidents before they reached production",
    ],
    "collaborate": [
        "delivered {n} features across a {m}-week sprint cycle",
        "unblocked {n} downstream teams by clarifying data contracts",
    ],
    "design_architecture": [
        "cut provisioning time from roughly {a} hours to about {b} minutes",
        "enabled {n} teams to deploy independently",
    ],
    "write_tests": [
        "brought test coverage from about {a} percent up to roughly {b}",
        "caught {n} regressions before merge over a {m}-week window",
    ],
    "default": [
        "saved roughly {n} hours of manual work",
        "reduced a repeated process from about {a} steps to {b}",
        "checked about {n} cases during review",
    ],
}
ENABLE_AUTO_REWRITE = False
SEVERE_REWRITE_ISSUES = {
    "placeholder",
    "artifact",
    "broken_grammar",
    "impossible_metric",
    "latex_error",
}


def should_auto_rewrite(issue_type: str) -> bool:
    return ENABLE_AUTO_REWRITE or issue_type in SEVERE_REWRITE_ISSUES
_used_metric_types: List[str] = []


def reset_metric_type_tracking():
    global _used_metric_types
    _used_metric_types.clear()

def pick_metric_hint(task_category: str) -> str:
    """Pick an intern-scale, casual metric hint. No odd-digit rule, no K/M/GB."""
    global _used_metric_types
    templates = METRIC_TEMPLATES.get(task_category, METRIC_TEMPLATES["build_system"])

    def _classify_metric_type(tpl: str) -> str:
        if "from" in tpl and "to" in tpl:
            return "x_to_y"
        if "%" in tpl:
            return "percentage"
        if "hours" in tpl or "minutes" in tpl or "seconds" in tpl:
            return "time"
        if "x " in tpl or "x the" in tpl:
            return "multiplier"
        return "count"

    scored = []
    for tpl in templates:
        mtype = _classify_metric_type(tpl)
        recent_count = _used_metric_types[-4:].count(mtype) if _used_metric_types else 0
        score = _random.random() - (recent_count * 0.5)
        scored.append((score, tpl, mtype))
    scored.sort(key=lambda x: x[0], reverse=True)
    tpl = scored[0][1]
    mtype = scored[0][2]
    _used_metric_types.append(mtype)

    # Intern-scale numbers: small, round-ish, casual
    a = _casual_int(3, 45)
    b = _casual_int(1, max(1, a // 2))
    n = _casual_int(2, 30)
    m = _casual_int(2, 8)
    a_d = _casual_int(60, 78)
    b_d = _casual_int(max(a_d + 2, 80), 95)

    try:
        return tpl.format(n=n, m=m, a=a, b=b, a_d=a_d, b_d=b_d)
    except (KeyError, ValueError, IndexError):
        return tpl



# ═══════════════════════════════════════════════════════════════
# PROGRESSION
# ═══════════════════════════════════════════════════════════════

def get_progression_context(block_index: int, total_blocks: int = 4) -> Dict[str, str]:
    if block_index == 0:
        return {"complexity": "advanced", "autonomy": "independently with periodic reviews"}
    elif block_index == total_blocks - 1:
        return {"complexity": "foundational", "autonomy": "under close guidance"}
    return {"complexity": "intermediate", "autonomy": "with regular mentorship"}


# ═══════════════════════════════════════════════════════════════
# CAPITALIZATION
# ═══════════════════════════════════════════════════════════════

_cap_cache: Dict[str, str] = {}


async def fix_capitalization_gpt(text: str) -> str:
    if not text or len(text.strip()) < 3:
        return text
    key = text.lower().strip()
    if key in _cap_cache:
        return _cap_cache[key]
    prompt = f"""Fix capitalization of technical terms. Return STRICT JSON: {{"fixed":"..."}}
Text: "{text}" """
    try:
        data = await gpt_json(prompt, temperature=0.0)
        fixed = data.get("fixed", text).strip()
        if len(key) < 50:
            _cap_cache[key] = fixed
        return fixed
    except Exception:
        return text


async def fix_capitalization_batch(items: List[str]) -> List[str]:
    if not items:
        return []
    uncached = [i for i in items if i.lower().strip() not in _cap_cache]
    if not uncached:
        return [_cap_cache.get(i.lower().strip(), i) for i in items]
    prompt = f"""Fix capitalization of technical keywords. Return STRICT JSON: {{"fixed":[...]}}
Keywords: {json.dumps(uncached)}"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        fl = data.get("fixed", uncached)
        if len(fl) != len(uncached):
            fl = uncached
        for o, f in zip(uncached, fl):
            _cap_cache[o.lower().strip()] = str(f).strip()
        return [_cap_cache.get(i.lower().strip(), i) for i in items]
    except Exception:
        return items


def _ensure_cap(s: str) -> str:
    s = (s or "").strip()
    return s[0].upper() + s[1:] if s and s[0].isalpha() and s[0].islower() else s


def fix_skill_capitalization_sync(skill: str) -> str:
    skill = (skill or "").strip()
    if not skill:
        return ""
    return _cap_cache.get(skill.lower().strip(), _ensure_cap(skill))


# ═══════════════════════════════════════════════════════════════
# PLACEHOLDER WORD SANITIZER
# ═══════════════════════════════════════════════════════════════

_PLACEHOLDER_PATTERNS = re.compile(
    r'\b('
    r'XYZ|ABC|DEF|GHI|JKL|MNO|PQR|STU|VWX|YZA|'
    r'Foo|Bar|Baz|Qux|Quux|Corge|Grault|Garply|Waldo|Fred|Plugh|'
    r'Lorem|Ipsum|Dolor|Amet|Consectetur|'
    r'Acme|Initech|Hooli|Pied Piper|Globex|'
    r'John Doe|Jane Doe|John Smith|Jane Smith|'
    r'PLACEHOLDER|TODO|FIXME|TBD|N/A|XXX|'
    r'widget[s]?|gadget[s]?|thingy|doohickey|'
    r'some company|some tool|some framework|some library|'
    r'Company A|Company B|Tool X|Tool Y|Service Z|'
    r'the system|the platform|the tool|the service|the application'
    r')\b',
    re.IGNORECASE
)

_ARTIFACT_PATTERNS = re.compile(
    r'\bT[0-9]\b|\bDell\s+T\b|\brebuild and enhance\b|\bdeveloped T[0-9]\b|\bwrote T[0-9]\b|\bassembled the\b',
    re.IGNORECASE
)


def deterministic_artifact_cleanup(text: str) -> str:
    fixes = [
        (r"\bWrote\s+T[0-9]\s+to\s+", "Wrote code to "),
        (r"\bDeveloped\s+T[0-9]\s+collaboration\s+with\b", "Worked with"),
        (r"\bDell\s+T\b", "the team"),
        (r"\bassembled the rebuild and enhance\b", "rebuilt"),
        (r"\bAssembled the\b", "Rebuilt the"),
    ]
    out = text
    for pat, repl in fixes:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()

def deterministic_company_name_cleanup(bullet: str, experience_companies: List[str], target_company: str = "") -> str:
    """Strip experience and target company names from bullet text."""
    if not bullet:
        return bullet
    all_companies = [c for c in experience_companies if c] if experience_companies else []
    if target_company and target_company.strip() and len(target_company.strip()) >= 3:
        all_companies.append(target_company.strip())
    if not all_companies:
        return bullet
    out = bullet
    for company in all_companies:
        if not company or len(company.strip()) < 3:
            continue
        patterns = [
            rf'\s*(?:,\s*)?(?:—\s*)?(?:at|at the|during my|during the|while at)\s+{re.escape(company)}\s*(?:internship|intern|role|position|fellowship)?\s*[,.]?\s*',
            rf'\s+at\s+{re.escape(company)}\b[,.]?\s*',
        ]
        for pat in patterns:
            out = re.sub(pat, ' ', out, flags=re.IGNORECASE)
    out = re.sub(r'\s+', ' ', out).strip()
    out = re.sub(r'\s+\.', '.', out)
    out = re.sub(r'\s+,', ',', out)
    return out

_METRIC_PLACEHOLDER_PAT = re.compile(
    r'\b[XxNn]\s*(%|percent|hours?|minutes?|seconds?|ms|times?|x\b|days?|weeks?|steps?)',
    re.IGNORECASE
)
_METRIC_PLACEHOLDER_STANDALONE = re.compile(
    r'(?:by|roughly|about|approximately|around|~)\s+[XxNn](%|(?:\s*percent))',
    re.IGNORECASE
)


def deterministic_metric_placeholder_cleanup(bullet: str) -> str:
    """Replace unfilled metric placeholders like 'X%', 'roughly X%', 'N hours' with casual numbers."""
    if not bullet:
        return bullet
    out = bullet

    # "roughly X%" → "roughly 25%"
    out = re.sub(r'(?i)((?:by|roughly|about|approximately|around|~)\s+)[Xx]\s*(%)', 
                 lambda m: m.group(1) + str(_casual_int(15, 40)) + m.group(2), out)
    # "X%" standalone
    out = re.sub(r'(?i)\bX\s*(%)', lambda m: str(_casual_int(15, 40)) + m.group(1), out)
    # "X hours/minutes/seconds"
    out = re.sub(r'(?i)\b[Xx]\s+(hours?|minutes?|seconds?)',
                 lambda m: str(_casual_int(2, 10)) + ' ' + m.group(1), out)
    # "X times" or "Xx"
    out = re.sub(r'(?i)\b[Xx]\s+(times?)\b',
                 lambda m: str(_casual_int(2, 5)) + ' ' + m.group(1), out)
    # "N steps" / "N files" / "N records" etc.
    out = re.sub(r'(?i)\b[Nn]\s+(steps?|files?|records?|checks?|variants?|teams?|features?)',
                 lambda m: str(_casual_int(3, 15)) + ' ' + m.group(1), out)

    return out


def has_metric_placeholder(bullet: str) -> bool:
    """Check if a bullet contains unfilled metric placeholders like X%, N hours."""
    return bool(_METRIC_PLACEHOLDER_PAT.search(bullet) or _METRIC_PLACEHOLDER_STANDALONE.search(bullet))

_PLACEHOLDER_PHRASES = [
    "xyz company", "abc corporation", "xyz tool", "abc framework",
    "company xyz", "company abc", "tool xyz", "platform xyz",
    "the xyz", "an xyz", "a xyz", "this xyz",
]


async def sanitize_placeholder_words(
    text: str, context: str = "", company: str = "", role: str = "",
) -> str:
    if not text:
        return text
    text_lower = text.lower()
    has_placeholder_phrase = any(p in text_lower for p in _PLACEHOLDER_PHRASES)
    matches = list(_PLACEHOLDER_PATTERNS.finditer(text))
    if not matches and not has_placeholder_phrase:
        return text
    _log(f"🔧 [SANITIZE] Found {len(matches)} placeholder matches in text")
    prompt = f"""The following text contains PLACEHOLDER WORDS or malformed resume-generator artifacts that need to be replaced with REAL, contextually appropriate words.

TEXT:
"{text[:500]}"

CONTEXT: Resume bullet for {role} role at {company}.
{('ADDITIONAL CONTEXT: ' + context[:300]) if context else ''}

The text may contain malformed resume-generator artifacts such as T3, T4, Dell T, "assembled the", or "rebuild and enhance".
Remove or rewrite those fragments naturally.
Do not invent new facts.
Do not add new metrics.
Do not copy JD wording.
Replace EVERY placeholder with a SPECIFIC, REAL term that fits the context.
Return STRICT JSON: {{"fixed": "the corrected text with all placeholders replaced"}}
"""
    try:
        data = await gpt_json(prompt, temperature=0.2)
        fixed = data.get("fixed", "").strip()
        if fixed and len(fixed) >= len(text) * 0.5:
            remaining = list(_PLACEHOLDER_PATTERNS.finditer(fixed))
            if len(remaining) < len(matches):
                _log(f"✅ [SANITIZE] Replaced {len(matches) - len(remaining)} placeholders")
                return fixed
        return text
    except Exception as e:
        _log(f"⚠️ [SANITIZE] Failed: {e}")
        return text

async def sanitize_all_bullets(
    all_bullets: List[List[str]], target_company: str, target_role: str,
    experience_companies: List[str],
) -> List[List[str]]:
    result = []
    for block_idx, block in enumerate(all_bullets):
        fixed_block = []
        ec = experience_companies[block_idx] if block_idx < len(experience_companies) else "Company"
        for bullet in block:
            # Deterministic artifact cleanup first
            bullet = deterministic_artifact_cleanup(bullet)
            # Fix unfilled metric placeholders (X%, N hours, etc.)
            bullet = deterministic_metric_placeholder_cleanup(bullet)
            # Strip company names from bullet text
            bullet = deterministic_company_name_cleanup(bullet, experience_companies, target_company)
            if _PLACEHOLDER_PATTERNS.search(bullet) or _ARTIFACT_PATTERNS.search(bullet) or has_metric_placeholder(bullet):
                fixed = await sanitize_placeholder_words(
                    bullet, context=f"Intern at {ec}", company=target_company, role=target_role)
                fixed_block.append(fixed)
            else:
                fixed_block.append(bullet)
        result.append(fixed_block)
    return result

# ═══════════════════════════════════════════════════════════════
# AI-TELL POST-FILTER — v2.9.2
# ═══════════════════════════════════════════════════════════════

AI_TELL_WORDS = frozenset({
    "spearheaded", "leveraged", "utilized", "orchestrated", "pioneered",
    "championed", "synthesized", "facilitated", "harnessed", "bolstered",
    "fortified", "galvanized", "operationalized", "productionized",
    "architected", "conceptualized", "ideated", "actualized", "streamlined",
    "elevated", "augmented", "endeavored", "fostered", "cultivated",
    "garnered", "propelled", "catapulted", "revolutionized", "empowered",
    "spurred", "devised", "instituted", "commenced", "helmed", "forged",
    "crafted", "navigated", "steered", "charted",
})

AI_TELL_PHRASES = [
    "resulting in", "thereby", "thus enabling", "which enabled",
    "comprehensive", "robust and scalable", "end-to-end solution",
    "cutting-edge", "state-of-the-art", "best-in-class", "world-class",
    "mission-critical", "enterprise-grade", "production-grade",
    "seamless integration", "holistic approach",
]


def _find_ai_tells(bullet: str) -> List[str]:
    """Find all AI-tell words and phrases in a bullet."""
    bl = bullet.lower()
    matches = []
    for word in bl.split():
        clean = re.sub(r'[^a-z]', '', word)
        if clean in AI_TELL_WORDS:
            matches.append(clean)
    for phrase in AI_TELL_PHRASES:
        if phrase in bl:
            matches.append(phrase)
    return matches


async def ai_tell_post_filter(
    all_bullets: List[List[str]], target_role: str, target_company: str,
) -> List[List[str]]:
    """Scan all bullets for AI-sounding language and rewrite offenders."""
    flat = [b for block in all_bullets for b in block]
    rewrites = 0

    for idx, bullet in enumerate(flat):
        matches = _find_ai_tells(bullet)
        #  Only rewrite if 2+ AI-tell words, or 1 severe phrase
        severe_phrases = {"cutting-edge", "mission-critical", "enterprise-grade",
                          "robust and scalable", "end-to-end solution", "holistic approach"}
        has_severe = any(m in severe_phrases for m in matches)
        if len(matches) < 2 and not has_severe:
            continue

        _log(f"🤖 [AI-TELL] Bullet {idx} has AI tells: {matches[:3]}")
        verb = re.sub(r"\\[#$%&_{}]", "", bullet.split()[0]) if bullet.split() else "Built"

        try:
            fix = await gpt_json(
                f"Rewrite this bullet in plain professional resume language.\n"
                f"Keep the same factual content.\n"
                f"Remove hype words.\n"
                f"Do not add tools.\n"
                f"Do not add metrics.\n"
                f"Do not add outcomes.\n"
                f"Do not copy JD language.\n"
                f"Keep it 18-30 words.\n"
                f"Start with \"{verb}\".\n"
                f"CURRENT: \"{bullet[:200]}\"\n"
                f'Return STRICT JSON: {{"bullet": "..."}}',
                temperature=0.35)
            new_b = fix.get("bullet", "")
            if new_b and len(new_b.split()) >= 15:
                # Verify the rewrite doesn't still have AI tells
                new_matches = _find_ai_tells(new_b)
                if len(new_matches) < len(matches):
                    new_b = await fix_capitalization_gpt(new_b)
                    new_b = adjust_bullet_length(new_b)
                    if not new_b.endswith("."):
                        new_b = new_b.rstrip(".,;: ") + "."
                    flat[idx] = latex_escape_text(new_b)
                    rewrites += 1
                    _log(f"✅ [AI-TELL] Bullet {idx} rewritten ({len(matches)} tells → {len(new_matches)})")
        except Exception as e:
            _log(f"⚠️ [AI-TELL] Rewrite failed for bullet {idx}: {e}")

    _log(f"🤖 [AI-TELL] {rewrites} bullets rewritten out of {len(flat)}")

    result, i = [], 0
    for block in all_bullets:
        result.append(flat[i:i + len(block)])
        i += len(block)
    return result


# ═══════════════════════════════════════════════════════════════
# METRIC BELIEVABILITY VERIFIER — v2.9.2 NEW
# ═══════════════════════════════════════════════════════════════

_METRIC_NUMBER_PAT = re.compile(r'\d[\d,]*\.?\d*')
_LARGE_NUMBER_PAT = re.compile(r'\b\d{2,}[KMG]B?\b|\b\d{5,}\b', re.I)
_SUB_10MS_PAT = re.compile(r'\b[1-9]ms\b|\bunder\s+10\s*ms\b|\b[1-9]\s*ms\b', re.I)
_HEDGE_WORDS = re.compile(r'\b(about|roughly|approximately|around|~)\b', re.I)


def _bullet_has_number(bullet: str) -> bool:
    """Check if a bullet contains any numerical metric."""
    bl = bullet.lower()
    # Strip LaTeX escapes for checking
    clean = re.sub(r'\\[#$%&_{}]', '', bl)
    return bool(_METRIC_NUMBER_PAT.search(clean))

def _count_metric_bullets(bullets: List[str]) -> List[int]:
    """Return indices of bullets that contain numerical metrics."""
    return [i for i, b in enumerate(bullets) if _bullet_has_number(b)]

async def verify_metric_believability(
    all_bullets: List[List[str]], target_role: str, target_company: str,
    jd_text: str,
) -> List[List[str]]:
    """Post-filter: ensure at most 4 bullets have numbers, and all metrics are intern-scale.

    1. If more than 4 bullets have numbers, rewrite extras to remove numbers.
    2. For remaining ≤4 metric bullets, check intern-scale believability.
    3. Add hedging language where missing.
    """
    flat = [b for block in all_bullets for b in block]
    if len(flat) < 6:
        return all_bullets

    metric_indices = _count_metric_bullets(flat)
    _log(f"🔍 [METRIC BELIEVE] {len(metric_indices)} bullets have numerical metrics")

    # --- Step 1: If more than 4 bullets have numbers, remove extras ---
    if len(metric_indices) > 4:
        # Keep the first bullet of each block (indices 0, 3, 6, 9) if they have metrics
        # Otherwise keep the first 4 metric bullets found
        preferred = {0, 3, 6, 9}
        keep = [i for i in metric_indices if i in preferred][:4]
        if len(keep) < 4:
            extras_pool = [i for i in metric_indices if i not in keep]
            keep.extend(extras_pool[:4 - len(keep)])
        remove_metrics_from = [i for i in metric_indices if i not in keep]

        _log(f"⚠️ [METRIC BELIEVE] Removing numbers from {len(remove_metrics_from)} bullets: {remove_metrics_from}")

        for idx in remove_metrics_from:
            verb = re.sub(r"\\[#$%&_{}]", "", flat[idx].split()[0]) if flat[idx].split() else "Built"
            try:
                fix = await gpt_json(
                    f'Rewrite this resume bullet to REMOVE all numerical metrics.\n'
                    f'Replace the metric with a qualitative outcome: what happened as a result?\n'
                    f'Examples of good qualitative outcomes:\n'
                    f'  "which the team adopted as the default pipeline"\n'
                    f'  "that replaced the previous manual workflow"\n'
                    f'  "so the on-call engineer could skip the morning data check"\n'
                    f'CURRENT: "{flat[idx][:200]}"\n'
                    f'Start with "{verb}". 24-34 words. NO numbers, NO percentages, NO latency figures.\n'
                    f'Return STRICT JSON: {{"bullet": "..."}}',
                    temperature=0.35)
                new_b = fix.get("bullet", "")
                if new_b and len(new_b.split()) >= 15 and not _bullet_has_number(new_b):
                    new_b = await fix_capitalization_gpt(new_b)
                    new_b = adjust_bullet_length(new_b)
                    if not new_b.endswith("."):
                        new_b = new_b.rstrip(".,;: ") + "."
                    flat[idx] = latex_escape_text(new_b)
                    _log(f"✅ [METRIC BELIEVE] Bullet {idx}: numbers removed, qualitative outcome added")
            except Exception as e:
                _log(f"⚠️ [METRIC BELIEVE] Failed to remove numbers from bullet {idx}: {e}")

        # Recount after removals
        metric_indices = _count_metric_bullets(flat)

    # --- Step 2: Check remaining metric bullets for intern-scale believability ---
    jd_lower = jd_text.lower()
    jd_mentions_scale = any(kw in jd_lower for kw in [
        "large-scale", "petabyte", "millions", "billions", "high-throughput",
        "distributed", "100k", "1m", "10m", "big data", "data lake",
    ])

    for idx in metric_indices:
        bullet = flat[idx]
        clean = re.sub(r'\\[#$%&_{}]', '', bullet)
        issues = []

        # Check for unrealistically large numbers (unless JD justifies it)
        if not jd_mentions_scale and _LARGE_NUMBER_PAT.search(clean):
            issues.append("unrealistic_scale")

        # Check for sub-10ms latency claims
        if _SUB_10MS_PAT.search(clean):
            issues.append("sub_10ms_latency")

        # Check for missing hedging language on numbers
        if not _HEDGE_WORDS.search(clean) and _METRIC_NUMBER_PAT.search(clean):
            issues.append("no_hedging")

        if not issues:
            continue

        _log(f"⚠️ [METRIC BELIEVE] Bullet {idx} issues: {issues}")

        verb = re.sub(r"\\[#$%&_{}]", "", flat[idx].split()[0]) if flat[idx].split() else "Built"
        issue_instructions = []
        if "unrealistic_scale" in issues:
            issue_instructions.append("Scale down the numbers to intern-level (no K/M/GB unless justified). Use small human-scale numbers.")
        if "sub_10ms_latency" in issues:
            issue_instructions.append("No sub-10ms latency claims. Use realistic numbers like 'from ~500ms to about 200ms' or 'from 4 seconds to under 1'.")
        if "no_hedging" in issues:
            issue_instructions.append("Add hedging: 'about', '~', or 'roughly' before numbers. Real interns estimate, they don't measure to exact digits.")

        try:
            fix = await gpt_json(
                f'Fix this resume bullet for metric believability.\n'
                f'ISSUES: {"; ".join(issue_instructions)}\n'
                f'CURRENT: "{flat[idx][:200]}"\n'
                f'Start with "{verb}". 24-34 words. Keep the technical content.\n'
                f'Return STRICT JSON: {{"bullet": "..."}}',
                temperature=0.3)
            new_b = fix.get("bullet", "")
            if new_b and len(new_b.split()) >= 15:
                # Verify the fix actually addressed the issues
                new_clean = re.sub(r'\\[#$%&_{}]', '', new_b)
                still_bad = False
                if "unrealistic_scale" in issues and _LARGE_NUMBER_PAT.search(new_clean):
                    still_bad = True
                if "sub_10ms_latency" in issues and _SUB_10MS_PAT.search(new_clean):
                    still_bad = True
                if not still_bad:
                    new_b = await fix_capitalization_gpt(new_b)
                    new_b = adjust_bullet_length(new_b)
                    if not new_b.endswith("."):
                        new_b = new_b.rstrip(".,;: ") + "."
                    flat[idx] = latex_escape_text(new_b)
                    _log(f"✅ [METRIC BELIEVE] Bullet {idx} fixed: {issues}")
        except Exception as e:
            _log(f"⚠️ [METRIC BELIEVE] Fix failed for bullet {idx}: {e}")

    _log(f"✅ [METRIC BELIEVE] Final: {len(_count_metric_bullets(flat))} bullets with metrics")

    result, i = [], 0
    for block in all_bullets:
        result.append(flat[i:i + len(block)])
        i += len(block)
    return result

# ═══════════════════════════════════════════════════════════════
# METRIC DIVERSITY ENFORCER
# ═══════════════════════════════════════════════════════════════

def _classify_bullet_metric(bullet: str) -> str:
    bl = bullet.lower()
    if re.search(r'\d+\s*%', bl):
        return "percentage"
    if re.search(r'from\s+\d+.*?to\s+\d+', bl):
        return "x_to_y"
    if re.search(r'\d+x\s', bl):
        return "multiplier"
    if re.search(r'\d+\s*(ms|seconds?|minutes?|hours?|days?)\b', bl):
        return "time"
    if re.search(r'\d+\s*(K|M|GB|TB|MB)\b', bl, re.I):
        return "count"
    if re.search(r'\d+', bl):
        return "numeric"
    return "none"

async def enforce_metric_diversity(
    all_bullets: List[List[str]], bullet_plan: List[Dict],
    target_role: str, jd_text: str,
) -> List[List[str]]:
    """Simplified: since we only have ~4 metric bullets, just ensure no two use the same format."""
    flat = [b for block in all_bullets for b in block]
    if len(flat) < 6:
        return all_bullets

    # Find bullets that actually have metrics
    metric_bullets = []
    for i, b in enumerate(flat):
        mt = _classify_bullet_metric(b)
        if mt != "none":
            metric_bullets.append((i, mt))

    if len(metric_bullets) <= 1:
        _log(f"✅ [METRIC DIV] {len(metric_bullets)} metric bullet(s), no diversity issue")
        return all_bullets

    # Check for duplicate metric formats
    seen_types: Dict[str, List[int]] = {}
    for idx, mt in metric_bullets:
        if mt not in seen_types:
            seen_types[mt] = []
        seen_types[mt].append(idx)

    # Find types that appear more than once
    duplicates = {mt: indices for mt, indices in seen_types.items() if len(indices) > 1}
    if not duplicates:
        _log(f"✅ [METRIC DIV] All {len(metric_bullets)} metrics use different formats: {[mt for _, mt in metric_bullets]}")
        return all_bullets

    _log(f"⚠️ [METRIC DIV] Duplicate metric formats: {duplicates}")

    # Desired types to cycle through
    desired_types = ["x_to_y", "percentage", "count", "time"]
    used_types = set()
    for idx, mt in metric_bullets:
        if mt not in duplicates or duplicates[mt][0] == idx:
            # Keep the first occurrence of each type
            used_types.add(mt)

    # Rewrite duplicate occurrences (skip the first of each type)
    for mt, indices in duplicates.items():
        for rewrite_idx in indices[1:]:  # skip first occurrence
            # Pick a type we haven't used yet
            available = [t for t in desired_types if t not in used_types]
            if not available:
                available = desired_types  # fallback: allow some overlap
            desired = available[0]
            used_types.add(desired)

            plan = bullet_plan[rewrite_idx] if rewrite_idx < len(bullet_plan) else {}
            tc = plan.get("task_category", "build_system")
            verb = re.sub(r"\\[#$%&_{}]", "", flat[rewrite_idx].split()[0]) if flat[rewrite_idx].split() else "Built"
            tech = plan.get("primary_technology", "Python")

            type_instructions = {
                "x_to_y": "Use a 'from X to Y' format (e.g., 'cut runtime from ~4 hours to under 1')",
                "percentage": "Use a casual percentage (e.g., 'about 30% faster')",
                "count": "Use a count (e.g., '~200 test images', 'about 5 team members')",
                "time": "Use time saved (e.g., 'saved the team roughly 6 hours a week')",
            }

            try:
                fix = await gpt_json(
                    f'Rewrite this resume bullet replacing the metric format.\n'
                    f'CURRENT: "{flat[rewrite_idx][:200]}"\n'
                    f'NEW METRIC FORMAT: {type_instructions.get(desired, "Use a different metric type")}\n'
                    f'Keep starting verb "{verb}". Mention {tech}. 24-34 words.\n'
                    f'Use hedging: "about", "~", "roughly". Intern-scale numbers only.\n'
                    f'Return STRICT JSON: {{"bullet": "..."}}',
                    temperature=0.35)
                new_b = fix.get("bullet", "")
                if new_b and len(new_b.split()) >= 15:
                    new_mt = _classify_bullet_metric(new_b)
                    if new_mt != mt:  # actually changed the format
                        new_b = await fix_capitalization_gpt(new_b)
                        new_b = adjust_bullet_length(new_b)
                        if not new_b.endswith("."):
                            new_b = new_b.rstrip(".,;: ") + "."
                        flat[rewrite_idx] = latex_escape_text(new_b)
                        _log(f"✅ [METRIC DIV] idx={rewrite_idx}: {mt} → {new_mt}")
            except Exception:
                pass

    result, i = [], 0
    for block in all_bullets:
        result.append(flat[i:i + len(block)])
        i += len(block)
    return result

# ═══════════════════════════════════════════════════════════════
# SKILL VALIDATION — GPT + JD context
# ═══════════════════════════════════════════════════════════════

_validated_cache: Dict[str, bool] = {}

_HARD_REJECTS = frozenset({
    "phd", "ph.d", "ms", "m.s", "msc", "bs", "b.s", "bsc",
    "bachelor", "master", "masters", "degree", "university", "college",
    "experience", "years", "year", "months", "weeks",
    "required", "preferred", "plus", "bonus", "nice to have",
    "strong", "excellent", "good", "proficient", "familiar", "advanced", "basic",
    "knowledge", "understanding", "ability", "skills", "skill",
    "iso", "nist", "gdpr", "hipaa", "sox", "pci", "cmmi", "itil",
    "compliance", "certified", "certification",
    "iso 42001", "nist ai rmf", "ai rmf", "rmf",
    "real-time applications", "computational efficiency",
    "clinical decision support", "end-to-end", "cross-functional",
    "data driven", "business intelligence",
})

_EXPERIENCE_PHRASING = re.compile(
    r"^(experience\s+(with|in|of|at|using|building|developing|working|leading|managing))"
    r"|^(understanding\s+of)"
    r"|^(knowledge\s+of)"
    r"|^(familiarity\s+with)"
    r"|^(exposure\s+to)"
    r"|^(awareness\s+of)"
    r"|^(background\s+in)"
    r"|^(proficiency\s+in)"
    r"|^(working\s+knowledge\s+of)",
    re.IGNORECASE,
)

_SOFT_SKILL_TERMS = frozenset({
    "teamwork", "team work", "communication", "communication skills",
    "interpersonal skills", "interpersonal", "leadership", "problem solving",
    "problem-solving", "critical thinking", "time management", "adaptability",
    "creativity", "collaboration", "presentation", "presentation skills",
    "stakeholder management", "project management", "organizational skills",
    "attention to detail", "analytical skills", "analytical thinking",
    "work ethic", "self-motivated", "fast learner", "quick learner",
    "detail-oriented", "results-driven", "proactive", "innovative thinking",
    "verbal communication", "written communication",
})


async def is_valid_skill(keyword: str, jd_snippet: str = "") -> bool:
    kl = keyword.lower().strip()
    cache_key = kl + ("|" + jd_snippet[:80] if jd_snippet else "")
    if cache_key in _validated_cache:
        return _validated_cache[cache_key]
    if (kl in _HARD_REJECTS
            or re.match(r"^(iso|nist|pci|gdpr|hipaa|sox)\s*[\d/]", kl)
            or len(keyword.split()) >= 6
            or re.match(r"^\d+\+?\s*(years?|months?|yrs?)", kl)):
        _validated_cache[cache_key] = False
        log_event(f"  🔍 skill '{keyword}' → ❌ (hard-reject)")
        return False
    if _EXPERIENCE_PHRASING.match(kl):
        _validated_cache[cache_key] = False
        log_event(f"  🔍 skill '{keyword}' → ❌ (experience description, not a skill name)")
        return False
    if kl in _SOFT_SKILL_TERMS:
        in_jd = bool(jd_snippet) and kl in jd_snippet.lower()
        _validated_cache[cache_key] = True if in_jd else (not bool(jd_snippet))
        icon = "✅" if _validated_cache[cache_key] else "❌"
        log_event(f"  🔍 skill '{keyword}' → {icon} (soft skill, in_jd={in_jd})")
        return _validated_cache[cache_key]
    jd_context = (f"\n\nJOB CONTEXT (use this to judge relevance):\n{jd_snippet}" if jd_snippet else "")
    prompt = f"""You are a senior technical recruiter reviewing a resume Skills section.
Decide whether "{keyword}" is a legitimate skill worth listing on a resume.{jd_context}

ACCEPT: Programming languages, ML/AI frameworks, data tools, cloud platforms, DevOps tools,
databases, ML concepts, protocols, technical methodologies, domain-specific concepts,
soft skills if they appear in the JD, any specific tool/technology named in the JD.

REJECT: Phrases starting with "Experience with/in/of", domain knowledge claims without a tool,
sentences, degree/experience requirements, compliance standards without tech, overly generic words.

When genuinely uncertain: ACCEPT.
Return STRICT JSON only: {{"is_skill": true, "reason": "one short phrase"}}"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        ok = bool(data.get("is_skill", True))
        _validated_cache[cache_key] = ok
        log_event(f"  🔍 skill '{keyword}' → {'✅' if ok else '❌'} ({data.get('reason', '')})")
        return ok
    except Exception:
        _validated_cache[cache_key] = True
        return True


async def filter_valid_skills(keywords: List[str], jd_snippet: str = "") -> List[str]:
    if not keywords:
        return []
    results = await asyncio.gather(*[is_valid_skill(k, jd_snippet) for k in keywords])
    return [k for k, ok in zip(keywords, results) if ok]


def clear_skill_validation_cache():
    global _validated_cache
    _validated_cache = {}


# ═══════════════════════════════════════════════════════════════
# EXTRACT ALL JD SKILLS (including soft skills)
# ═══════════════════════════════════════════════════════════════

async def extract_all_jd_skills(jd_text: str) -> List[str]:
    prompt = f"""Extract ALL skills (technical AND soft/interpersonal) from this job description.
Include everything: programming languages, frameworks, tools, platforms, methodologies,
soft skills like teamwork, communication, leadership, problem-solving, etc.

JOB DESCRIPTION:
{jd_text[:3500]}

Return STRICT JSON:
{{
    "technical_skills": ["Python", "PyTorch", "Docker", ...],
    "soft_skills": ["Communication", "Teamwork", "Leadership", ...]
}}
"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        tech = [str(s).strip() for s in (data.get("technical_skills") or []) if str(s).strip()]
        soft = [str(s).strip() for s in (data.get("soft_skills") or []) if str(s).strip()]
        all_skills = tech + soft
        _log(f"📋 [ALL JD SKILLS] {len(tech)} technical + {len(soft)} soft = {len(all_skills)}")
        return all_skills
    except Exception as e:
        _log(f"⚠️ [ALL JD SKILLS] Failed: {e}")
        return []

_company_cache: Dict[str, Dict] = {}


async def get_company_context_gpt(
    name: str, jd_text: str = "", target_role: str = "",
) -> Dict[str, Any]:
    nl = (name or "").lower().strip()

    if nl in _company_cache:
        return _company_cache[nl]

    jd_context = ""
    if jd_text and target_role:
        jd_context = f"""
The candidate is applying for {target_role}.

JD CONTEXT (what the candidate is targeting):
{jd_text[:3000]}

IMPORTANT: When listing realistic_technologies, FAVOR technologies mentioned in the JD.
If the JD mentions PyTorch and this company could plausibly use PyTorch, include PyTorch.
The goal is to maximize overlap between the company's believable tech stack and the JD's requirements.
"""

    prompt = f"""Analyze "{name}" for resume context.{jd_context}

Return STRICT JSON:
{{"type":"industry_internship|research_internship|internship","domain":"2-4 words",
"context":"1-2 sentences","technical_vocabulary":["5-8 terms"],
"realistic_technologies":["6-10 tools, PRIORITIZING tools that overlap with the JD"],
"unrealistic_technologies":["3-5 tools this company would NOT use"],
"jd_overlap_areas":["2-3 areas where this company's work plausibly overlaps with the JD's requirements"]}}
Be REALISTIC about what an intern would do here."""
    try:
        data = await gpt_json(prompt, temperature=0.2)
        result = {
            "type": data.get("type", "internship"),
            "domain": data.get("domain", "Technology"),
            "context": data.get("context", "Technical internship."),
            "technical_vocabulary": data.get("technical_vocabulary", []),
            "realistic_technologies": data.get("realistic_technologies", ["Python"]),
            "unrealistic_technologies": data.get("unrealistic_technologies", []),
            "jd_overlap_areas": data.get("jd_overlap_areas", []),
        }
        _company_cache[nl] = result
        _log(f"🏢 [COMPANY] {name}: {result['type']}")
        return result
    except Exception:
        fb = {"type": "internship", "domain": "Technology", "context": "",
              "technical_vocabulary": [], "realistic_technologies": ["Python"],
              "unrealistic_technologies": []}
        _company_cache[nl] = fb
        return fb


_core_cache: Dict[str, Dict] = {}


async def extract_company_core_requirements(company: str, role: str, jd: str) -> Dict[str, Any]:
    ck = f"{company.lower()}__{role.lower()}"
    if ck in _core_cache:
        return _core_cache[ck]
    if not company.strip() or company.lower() in {"company", "unknown"}:
        out = {"core_areas": [], "core_keywords": [], "notes": "Generic."}
        _core_cache[ck] = out
        return out
    prompt = (f"Infer key expectations for {company} / {role} not explicitly in JD.\n"
              f'Return STRICT JSON: {{"core_areas":["..."],"core_keywords":["..."],"notes":"..."}}\n'
              f"JD:\n{jd[:2500]}")
    try:
        data = await gpt_json(prompt, temperature=0.0)
        areas = await fix_capitalization_batch(
            [str(x).strip() for x in (data.get("core_areas", []) or []) if str(x).strip()])
        kws = await fix_capitalization_batch(
            [str(x).strip() for x in (data.get("core_keywords", []) or []) if str(x).strip()])
        seen: Set[str] = set()
        da, dk = [], []
        for a in areas:
            if a.lower() not in seen:
                seen.add(a.lower()); da.append(a)
        for k in kws:
            if k.lower() not in seen:
                seen.add(k.lower()); dk.append(k)
        out = {"core_areas": da[:8], "core_keywords": dk[:18], "notes": data.get("notes", "")}
        _core_cache[ck] = out
        return out
    except Exception:
        out = {"core_areas": [], "core_keywords": [], "notes": "Fallback."}
        _core_cache[ck] = out
        return out


# ═══════════════════════════════════════════════════════════════
# IDEAL CANDIDATE
# ═══════════════════════════════════════════════════════════════

_ideal_cache: Dict[str, Dict] = {}


async def profile_ideal_candidate(jd: str, company: str, role: str) -> Dict[str, Any]:
    ck = f"{company.lower()}__{role.lower()}"
    if ck in _ideal_cache:
        return _ideal_cache[ck]
    prompt = f"""Senior recruiter at {company} hiring {role}. JD:\n{jd[:3000]}
What does this job REALLY need? Return STRICT JSON:
{{"ideal_profile_summary":"2-3 sentences","implicit_requirements":[{{"requirement":"...","importance_rank":1}}],"top_3_must_haves":["..."],"differentiation_factors":["..."]}}"""
    try:
        data = await gpt_json(prompt, temperature=0.3, model="gpt-5.4-nano")
        result = {
            "ideal_profile_summary": data.get("ideal_profile_summary", ""),
            "implicit_requirements": (data.get("implicit_requirements") or [])[:6],
            "top_3_must_haves": (data.get("top_3_must_haves") or [])[:3],
            "differentiation_factors": (data.get("differentiation_factors") or [])[:4],
        }
        _ideal_cache[ck] = result
        return result
    except Exception:
        fb = {"ideal_profile_summary": "", "implicit_requirements": [],
              "top_3_must_haves": [], "differentiation_factors": []}
        _ideal_cache[ck] = fb
        return fb


# ═══════════════════════════════════════════════════════════════
# LATEX UTILITIES — v2.8.1: Added latex_escape_for_macro_arg
# ═══════════════════════════════════════════════════════════════

LATEX_ESC = {"#": r"\#", "%": r"\%", "$": r"\$", "&": r"\&",
             "_": r"\_", "{": r"\{", "}": r"\}"}
UNICODE_NORM = {"\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2022": "-",
                "\u00b7": "-", "\u25cf": "-", "\u2192": "->", "\u21d2": "=>",
                "\u00d7": "x", "\u00b0": " degrees ", "\u00A0": " ",
                "\uf0b7": "-", "\x95": "-"}


def latex_escape_text(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    for a, b in UNICODE_NORM.items():
        s = s.replace(a, b)
    for ch in ["%", "$", "&", "_", "#", "{", "}"]:
        s = re.sub(rf"(?<!\\){re.escape(ch)}", LATEX_ESC[ch], s)
    s = re.sub(r"(?<!\\)\^", r"\^{}", s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    s = re.sub(r"\\(?![a-zA-Z#$%&_{}^])", "", s)
    return s


def latex_escape_for_macro_arg(s: str) -> str:
    """v2.8.1 NEW: Escape text for safe use inside \\newcommand-defined macro arguments.

    Standard \\# works in body text but can break inside macro arguments
    like \\resumeSubheading{...}{...}{...}{...} depending on how the
    template processes its parameters (e.g., \\edef, \\write, \\MakeUppercase).

    This function applies standard escaping first, then wraps every \\#
    in protective braces: {\\#}. This is safe in all LaTeX contexts.

    Use this instead of latex_escape_text() when the text will be placed
    inside a brace group that is a macro argument — particularly:
    - \\resumeSubheading{HERE}{HERE}{HERE}{HERE}
    - \\textbf{HERE} (when inside another macro's argument)
    - Any \\newcommand-defined macro's parameters
    """
    s = latex_escape_text(s)
    # Wrap \# in braces for protection inside macro arguments.
    # {\\#} is safe because it creates a group containing \\#,
    # which prevents LaTeX from misinterpreting # as a parameter token.
    s = s.replace("\\#", "{\\#}")
    return s


def strip_all_macros_keep_text(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s).replace("{", "").replace("}", "")
    for a, b in UNICODE_NORM.items():
        s = s.replace(a, b)
    return s.strip()


MIN_WORDS, MAX_WORDS = 22, 34


def get_word_count(t: str) -> int:
    return len((t or "").split())


def adjust_bullet_length(text: str) -> str:
    text = (text or "").strip()
    words = text.split()
    n = len(words)
    if n <= MAX_WORDS:
        return text.rstrip(".,;:") + "."
    candidate = " ".join(words[:MAX_WORDS])
    ws = MAX_WORDS - 10
    window = " ".join(words[ws:MAX_WORDS])
    lc = window.rfind(",")
    if lc > 0:
        combined = (" ".join(words[:ws]) + " " + window[:lc]).strip()
        if get_word_count(combined) >= MIN_WORDS:
            return combined.rstrip(".,;:") + "."
    for conn in [" and ", " with ", " using ", " via ", " by ", " to ", " for "]:
        idx = candidate.rfind(conn)
        if idx > 0 and get_word_count(candidate[:idx]) >= MIN_WORDS:
            return candidate[:idx].rstrip(".,;:") + "."
    return candidate.rstrip(".,;:") + "."


def find_resume_items(block: str) -> List[Tuple[int, int, int, int]]:
    out, i, macro, n = [], 0, r"\resumeItem", len(r"\resumeItem")
    while True:
        i = block.find(macro, i)
        if i < 0:
            break
        j = i + n
        while j < len(block) and block[j].isspace():
            j += 1
        if j >= len(block) or block[j] != "{":
            i = j; continue
        ob, depth, k = j, 0, j
        while k < len(block):
            if block[k] == "{":
                depth += 1
            elif block[k] == "}":
                depth -= 1
                if depth == 0:
                    out.append((i, ob, k, k + 1)); i = k + 1; break
            k += 1
        else:
            break
    return out


def section_rx(name: str) -> re.Pattern:
    words = [w for w in re.split(r"\W+", name) if len(w) > 2] or [name]
    la = "".join(rf"(?=[^{{}}]*\b{re.escape(w)}\b)" for w in words)
    return re.compile(
        rf"(\\section\*?\{{{la}[^}}]*\}}[\s\S]*?)(?=\\section\*?\{{|\\end\{{document\}}|$)",
        re.IGNORECASE)


def _count_experience_bullets(tex: str) -> int:
    exp_pat = section_rx("Experience")
    total = 0
    for m in exp_pat.finditer(tex):
        total += len(find_resume_items(m.group(1)))
    return total


# ═══════════════════════════════════════════════════════════════
# BRACE BALANCE HELPER
# ═══════════════════════════════════════════════════════════════

def _brace_balance(s: str) -> int:
    depth = 0
    i = 0
    while i < len(s):
        if s[i] == '\\':
            i += 2
            continue
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
        i += 1
    return depth

# ═══════════════════════════════════════════════════════════════
# PDF METADATA INJECTION
# ═══════════════════════════════════════════════════════════════

def _meta_escape(s: str) -> str:
    if not s:
        return ""
    for ch in "\\{}#%&~^_$":
        s = s.replace(ch, "")
    return re.sub(r"\s+", " ", s).strip()


def _find_matching_brace(tex: str, open_pos: int) -> int:
    if open_pos >= len(tex) or tex[open_pos] != "{":
        return -1
    depth, k = 0, open_pos
    while k < len(tex):
        if tex[k] == "{":
            depth += 1
        elif tex[k] == "}":
            depth -= 1
            if depth == 0:
                return k
        k += 1
    return -1


def inject_pdf_metadata(
    tex: str,
    target_company: str,
    target_role: str,
    skills_list: List[str],
    courses: List[str],
) -> str:
    skill_kw_str = ", ".join(_meta_escape(_ensure_cap(s)) for s in skills_list[:40] if s)

    author = "Sri Akash Kadali"
    title = f"{author} - Resume - {_meta_escape(target_role)} at {_meta_escape(target_company)}"
    subject = f"Resume for {_meta_escape(target_role)} at {_meta_escape(target_company)}"
    keywords = _meta_escape(skill_kw_str)

    new_hs = (
        "\\hypersetup{\n"
        "  pdfauthor={" + author + "},\n"
        "  pdftitle={" + title + "},\n"
        "  pdfsubject={" + subject + "},\n"
        "  pdfkeywords={" + keywords + "},\n"
        "  pdfcreator={Sri Akash Kadali},\n"
        "  pdfproducer={Sri Akash Kadali},\n"
        "  hidelinks,\n"
        "  colorlinks=false,\n"
        "}\n"
    )

    hs_start = tex.find(r"\hypersetup{")
    while hs_start >= 0:
        brace_pos = hs_start + len(r"\hypersetup")
        close = _find_matching_brace(tex, brace_pos)
        if close >= 0:
            end = close + 1
            if end < len(tex) and tex[end] == '\n':
                end += 1
            tex = tex[:hs_start] + tex[end:]
        else:
            break
        hs_start = tex.find(r"\hypersetup{")

    has_hyperref = bool(re.search(r'\\usepackage(\[[^\]]*\])?\{hyperref\}', tex))

    if not has_hyperref:
        last_pkg = None
        bd = tex.find(r"\begin{document}")
        for m in re.finditer(r'^\\usepackage(\[[^\]]*\])?\{[^}]+\}[^\n]*$', tex, re.M):
            if bd < 0 or m.end() < bd:
                last_pkg = m
        if last_pkg:
            insert_pos = last_pkg.end()
            tex = tex[:insert_pos] + "\n\\usepackage{hyperref}\n" + tex[insert_pos:]
        elif bd >= 0:
            tex = tex[:bd] + "\\usepackage{hyperref}\n" + tex[bd:]

    bd = tex.find(r"\begin{document}")
    if bd >= 0:
        inject_block = (
            "\n% --- PDF Metadata (v2.8.2) ---\n"
            + new_hs
            + "% --- End PDF Metadata ---\n"
        )
        tex = tex[:bd] + inject_block + tex[bd:]

    _log(f"📄 [METADATA] Injected PDF metadata: author={author}, {len(skills_list)} skill keywords")
    return tex


# ═══════════════════════════════════════════════════════════════
# JD ANALYSIS
# ═══════════════════════════════════════════════════════════════

async def extract_company_role(jd: str) -> Tuple[str, str]:
    try:
        data = await gpt_json(
            f'Return STRICT JSON: {{"company":"...","role":"..."}}\nJD:\n{jd}',
            temperature=0.0)
        return data.get("company", "Company"), data.get("role", "Role")
    except Exception:
        return "Company", "Role"


async def extract_keywords_with_priority(jd: str) -> Dict[str, Any]:
    p1 = f"""Extract ALL technical keywords from this JD. Return STRICT JSON:
{{"must_have":["..."],"should_have":["..."],"nice_to_have":["..."],
"key_responsibilities":["5-7 duties"],"domain_context":"..."}}
JD:\n{jd}"""
    try:
        data = await gpt_json(p1, temperature=0.0)
        must_r = [str(k).strip() for k in data.get("must_have", []) if str(k).strip()]
        should_r = [str(k).strip() for k in data.get("should_have", []) if str(k).strip()]
        nice_r = [str(k).strip() for k in data.get("nice_to_have", []) if str(k).strip()]
        all_raw = must_r + should_r + nice_r
        if all_raw:
            af = await fix_capitalization_batch(all_raw)
            i = 0
            must = af[i:i + len(must_r)]; i += len(must_r)
            should = af[i:i + len(should_r)]; i += len(should_r)
            nice = af[i:i + len(nice_r)]
        else:
            must, should, nice = [], [], []
        seen: Set[str] = set()

        def dd(lst):
            o = []
            for x in lst:
                x = str(x).strip()
                if x and x.lower() not in seen:
                    seen.add(x.lower()); o.append(x)
            return o

        must, should, nice = dd(must), dd(should), dd(nice)
        return {"must_have": must, "should_have": should, "nice_to_have": nice,
                "all_keywords": must + should + nice,
                "responsibilities": list(data.get("key_responsibilities", [])),
                "domain": data.get("domain_context", "Technology")}
    except Exception:
        return {"must_have": [], "should_have": [], "nice_to_have": [],
                "all_keywords": [], "responsibilities": [], "domain": "Technology"}


# ═══════════════════════════════════════════════════════════════
# SKILL RANKING BY JD RELEVANCE
# ═══════════════════════════════════════════════════════════════

MAX_SKILLS = 30
MAX_SKILLS_TIGHT = 20
MAX_SKILLS_EMERGENCY = 12


def rank_skills_by_jd_relevance(
    skills_raw: List[str], must_have: List[str], should_have: List[str],
    nice_to_have: List[str], core_keywords: List[str], jd_text: str,
    max_skills: int = MAX_SKILLS,
) -> List[str]:
    jd_lower = jd_text.lower()
    must_set = {k.lower() for k in must_have}
    should_set = {k.lower() for k in should_have}
    nice_set = {k.lower() for k in nice_to_have}
    core_set = {k.lower() for k in core_keywords}
    seen: Set[str] = set()
    scored: List[Tuple[int, int, str]] = []
    for idx, skill in enumerate(skills_raw):
        s = (skill or "").strip()
        if not s or s.lower() in seen:
            continue
        seen.add(s.lower())
        sl = s.lower()
        if sl in must_set: score = 100
        elif sl in core_set: score = 80
        elif sl in should_set: score = 60
        elif sl in nice_set: score = 40
        elif sl in jd_lower: score = 20
        else: score = 1
        scored.append((score, -idx, s))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    result = [s for _, _, s in scored[:max_skills]]
    _log(f"📊 [SKILLS RANK] {len(skills_raw)} raw → {len(result)} after ranking (cap={max_skills})")
    return result


# ═══════════════════════════════════════════════════════════════
# SKILLS SECTION RENDERING — v2.8.1: CATEGORIZED w/ safe escaping
# ═══════════════════════════════════════════════════════════════

SKILL_CATEGORY_ORDER = [
    "Languages",
    "Frameworks/Libraries",
    "ML/AI",
    "Databases",
    "Cloud/DevOps",
    "Tools",
    "Other",
]

_SKILL_CATEGORY_RULES: Dict[str, List[str]] = {
    "Languages": [
        "python", "java", "c++", "c", "javascript", "typescript", "go", "golang",
        "rust", "scala", "kotlin", "ruby", "r", "matlab", "sql", "bash", "shell",
        "perl", "swift", "dart", "lua", "php", "haskell", "julia", "assembly",
        "html", "css", "c#", "objective-c", "vhdl", "verilog",
    ],
    "Frameworks/Libraries": [
        "react", "angular", "vue", "next.js", "fastapi", "flask", "django", "spring",
        "express", "node.js", "pandas", "numpy", "scipy", "matplotlib", "seaborn",
        "plotly", "streamlit", "gradio", "langchain", "llamaindex", "crewai",
        "bootstrap", "tailwind", "jquery", "redux", "celery", "asyncio",
    ],
    "ML/AI": [
        "pytorch", "tensorflow", "keras", "scikit-learn", "xgboost", "lightgbm",
        "hugging face", "transformers", "bert", "gpt", "llama", "mistral",
        "opencv", "yolo", "detectron", "spacy", "nltk", "mlflow", "wandb",
        "dvc", "ray", "deepspeed", "onnx", "tensorrt", "triton", "vllm",
        "stable diffusion", "rag", "fine-tuning", "rlhf", "lora", "qlora",
        "langsmith", "openai", "anthropic", "gemini",
    ],
    "Databases": [
        "postgresql", "mysql", "sqlite", "mongodb", "redis", "elasticsearch",
        "dynamodb", "cassandra", "neo4j", "pinecone", "weaviate", "chromadb",
        "milvus", "qdrant", "snowflake", "bigquery", "redshift", "clickhouse",
        "supabase", "firebase", "cockroachdb", "timescaledb",
    ],
    "Cloud/DevOps": [
        "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ansible",
        "jenkins", "github actions", "gitlab ci", "circleci", "helm", "istio",
        "prometheus", "grafana", "cloudwatch", "datadog", "sagemaker",
        "vertex ai", "azure ml", "ec2", "s3", "lambda", "ecs", "eks", "gke",
        "cloud run", "cloud functions", "airflow", "prefect", "dagster",
        "kafka", "rabbitmq", "pulsar", "nginx", "traefik", "ci/cd",
    ],
    "Tools": [
        "git", "github", "gitlab", "bitbucket", "jira", "confluence", "slack",
        "postman", "swagger", "linux", "unix", "vim", "vscode", "jupyter",
        "colab", "latex", "markdown", "figma", "notion", "trello",
        "dbeaver", "pgadmin", "compass", "insomnia", "wireshark",
    ],
}


def _categorize_skill_fallback(skill: str) -> str:
    sl = skill.lower().strip()
    for category, keywords in _SKILL_CATEGORY_RULES.items():
        for kw in keywords:
            if kw == sl or kw in sl or sl in kw:
                return category
    return "Other"


async def categorize_skills_gpt(skills: List[str], jd_text: str = "") -> Dict[str, List[str]]:
    if not skills:
        return {}

    prompt = f"""You are a resume formatting expert. Categorize these technical skills
into EXACTLY the following subheading categories for a resume Skills section:

CATEGORIES (use these exact names):
- "Languages" — Programming languages ONLY (Python, C++, Java, SQL, etc.)
- "Frameworks/Libraries" — Software frameworks, libraries, SDKs (PyTorch, React, FastAPI, Pandas, etc.)
- "ML/AI" — Machine learning concepts, AI tools, model types (Transformers, RAG, Fine-tuning, BERT, etc.)
- "Databases" — Database systems (PostgreSQL, MongoDB, Redis, Pinecone, etc.)
- "Cloud/DevOps" — Cloud platforms, CI/CD, orchestration, MLOps (AWS, Docker, Kubernetes, Airflow, etc.)
- "Tools" — Developer tools, version control, IDEs, utilities (Git, Linux, Jupyter, etc.)
- "Other" — Anything that doesn't fit above (soft skills, domain concepts, methodologies)

SKILLS TO CATEGORIZE:
{json.dumps(skills[:40])}

{('JD CONTEXT: ' + jd_text[:800]) if jd_text else ''}

RULES:
- Every skill must appear in exactly ONE category
- Use the EXACT category names listed above
- If a skill fits multiple categories, pick the MOST SPECIFIC one
- "Other" is the catch-all — use it sparingly
- Do not include vague concepts like "AI", "data-driven", "business insights", or "statistical thinking" unless they are concrete tools or methods

Return STRICT JSON:
{{
    "Languages": ["Python", ...],
    "Frameworks/Libraries": ["PyTorch", ...],
    "ML/AI": ["Transformers", ...],
    "Databases": ["PostgreSQL", ...],
    "Cloud/DevOps": ["AWS", ...],
    "Tools": ["Git", ...],
    "Other": ["..."]
}}
"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        result: Dict[str, List[str]] = {}
        categorized_skills: Set[str] = set()

        for cat in SKILL_CATEGORY_ORDER:
            items = data.get(cat, [])
            if isinstance(items, list):
                clean = []
                for s in items:
                    s = str(s).strip()
                    if s and s.lower() not in categorized_skills:
                        categorized_skills.add(s.lower())
                        clean.append(_ensure_cap(s))
                if clean:
                    result[cat] = clean

        for skill in skills:
            if skill.lower().strip() not in categorized_skills:
                cat = _categorize_skill_fallback(skill)
                if cat not in result:
                    result[cat] = []
                result[cat].append(_ensure_cap(skill))

        if "Other" in result and not result["Other"]:
            del result["Other"]

        total = sum(len(v) for v in result.values())
        _log(f"📊 [SKILL CATEGORIES] {total} skills across {len(result)} categories: "
             f"{', '.join(f'{k}({len(v)})' for k, v in result.items())}")
        return result

    except Exception as e:
        _log(f"⚠️ [SKILL CATEGORIES] GPT failed: {e}, using fallback")
        result: Dict[str, List[str]] = {}
        for skill in skills:
            cat = _categorize_skill_fallback(skill)
            if cat not in result:
                result[cat] = []
            result[cat].append(_ensure_cap(skill))
        return result

def render_skills_section_categorized(categorized_skills: Dict[str, List[str]]) -> str:
    """Render Skills section in the exact single-item multiline style requested."""
    if not categorized_skills:
        return ""

    lines = []
    lines.append("\\section{Skills}")
    lines.append("\\begin{itemize}[leftmargin=0.15in, label={}]")

    rendered_rows = []

    for cat in SKILL_CATEGORY_ORDER:
        if cat not in categorized_skills or not categorized_skills[cat]:
            continue
        skills_str = ", ".join(latex_escape_text(s) for s in categorized_skills[cat])
        cat_esc = latex_escape_for_macro_arg(cat)
        rendered_rows.append(f"\\textbf{{{cat_esc}:}} \\small{{{skills_str}}} \\\\")

    for cat, skills in categorized_skills.items():
        if cat not in SKILL_CATEGORY_ORDER and skills:
            skills_str = ", ".join(latex_escape_text(s) for s in skills)
            cat_esc = latex_escape_for_macro_arg(cat)
            rendered_rows.append(f"\\textbf{{{cat_esc}:}} \\small{{{skills_str}}} \\\\")

    if rendered_rows:
        lines.append("\\item " + rendered_rows[0])
        lines.extend(rendered_rows[1:])

    lines.append("\\end{itemize}")
    return "\n".join(lines)

async def replace_skills_section(
    body: str, skills: List[str], jd_text: str = "",
    categorized: Optional[Dict[str, List[str]]] = None,
) -> str:
    if not categorized:
        return body
    nb = render_skills_section_categorized(categorized)

    if not nb:
        return body

    patterns = [
        re.compile(r"\\section\*?\{Skills\}[\s\S]*?(?=\\section\*?\{|\\end\{document\}|$)", re.I),
        re.compile(r"\\section\*?\{Technical Skills\}[\s\S]*?(?=\\section\*?\{|\\end\{document\}|$)", re.I),
    ]

    for pat in patterns:
        if pat.search(body):
            return pat.sub(lambda _: nb + "\n", body, count=1)

    m = re.search(r"%-----------TECHNICAL SKILLS-----------", body, re.I)
    if m:
        return body[:m.end()] + "\n" + nb + "\n" + body[m.end():]

    return body


def strip_undergraduate_degree(tex: str) -> str:
    _log("🎓 [EDUCATION] Stripping undergraduate degree and relevant coursework...")

    edu_pat = section_rx("Education")
    m = edu_pat.search(tex)
    if not m:
        _log("⚠️ [EDUCATION] No Education section found")
        return tex

    section_text = m.group(1)

    # Split education section into full entry blocks:
    # each block = \resumeSubheading ... optional \resumeItemListStart ... \resumeItemListEnd
    block_pat = re.compile(
        r'(\\resumeSubheading\s*'
        r'\{[\s\S]*?\}\s*\{[\s\S]*?\}\s*\{[\s\S]*?\}\s*\{[\s\S]*?\}'
        r'(?:[\s\S]*?\\resumeItemListStart[\s\S]*?\\resumeItemListEnd)?'
        r')',
        re.MULTILINE
    )

    blocks = block_pat.findall(section_text)
    if not blocks:
        return tex

    kept = []
    for block in blocks:
        bl = block.lower()

        is_undergrad = any(k in bl for k in [
            "bachelor", "b.s.", "b.tech", "btech", "b.e.", "undergraduate",
            "iiit vadodara", "iiit", "b.sc", "bsc",
        ])

        # Remove coursework lines from kept entries
        block = re.sub(
            r'[ \t]*\\item\s*\{?\\textbf\{Relevant Coursework:?\}\}?[^\\\n]*(?:\\\\)?\n?',
            '',
            block,
            flags=re.IGNORECASE
        )
        block = re.sub(
            r'\\resumeItemListStart\s*\\resumeItemListEnd',
            '',
            block
        )

        if not is_undergrad:
            kept.append(block.strip())

    if not kept:
        _log("⚠️ [EDUCATION] All education blocks removed unexpectedly; keeping original")
        return tex

    new_section = (
        "\\section{Education}\n"
        "\\resumeSubHeadingListStart\n"
        + "\n\n".join(kept) +
        "\n\\resumeSubHeadingListEnd\n"
    )

    return tex[:m.start()] + new_section + tex[m.end():]

# ═══════════════════════════════════════════════════════════════
# ATS SELF-SIMULATION PASS
# ═══════════════════════════════════════════════════════════════

async def ats_self_simulation_pass(
    body_tex: str, jd_text: str, all_keywords: List[str], must_have: List[str],
) -> Tuple[List[str], List[str]]:
    resume_plain = strip_all_macros_keep_text(body_tex).lower()
    present = [k for k in all_keywords if k.lower() in resume_plain]
    missing_must = [k for k in must_have if k.lower() not in resume_plain]
    prompt = f"""You are an ATS scanner reviewing a resume against a job description.

JOB DESCRIPTION (first 3000 chars):
{jd_text[:3000]}

RESUME PLAIN TEXT:
{resume_plain[:3000]}

KEYWORDS PRESENT: {json.dumps(present[:25])}
MUST-HAVE MISSING: {json.dumps(missing_must[:10])}

Identify 8-12 technical keywords from the JD that are absent or underrepresented.
Real terms only — languages, frameworks, tools, platforms. Soft skills OK if in JD.

Beyond keywords, also check: does this resume demonstrate EACH of the JD's
top 5 responsibilities? List any JD responsibility that NO bullet addresses.

Return STRICT JSON: {{
    "missing_keywords": ["keyword1", ...],
    "uncovered_responsibilities": ["responsibility from JD that no bullet demonstrates"],
    "reasoning": "brief"
}}
Max 12 keywords, max 5 uncovered responsibilities."""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        raw = [str(k).strip() for k in (data.get("missing_keywords") or []) if str(k).strip()]

        uncovered = [str(r).strip() for r in (data.get("uncovered_responsibilities") or []) if str(r).strip()]
        if uncovered:
            _log(f"⚠️ [ATS SIM] {len(uncovered)} uncovered JD responsibilities: {uncovered[:3]}")

        if not raw:
            return [], uncovered[:5]
        validated = await filter_valid_skills(raw, jd_text[:500])
        if validated:
            validated = await fix_capitalization_batch(validated)
        _log(f"🤖 [ATS SIM] {len(validated)} under-represented keywords → adding to Skills")
        return validated[:12], uncovered[:5]
    except Exception as e:
        _log(f"⚠️ [ATS SIM] Failed: {e}")
        return [], []


async def plan_all_12_bullets(
    jd_text: str, target_company: str, target_role: str,
    jd_tasks: List[Dict[str, Any]], jd_keywords: List[str],
    ideal_candidate: Dict[str, Any], role_archetype: Dict[str, Any],
    experience_companies: List[str],
) -> Dict[str, Any]:
    top_3 = ideal_candidate.get("top_3_must_haves", [])
    implicit_reqs = ideal_candidate.get("implicit_requirements", [])
    archetype_key = role_archetype.get("key", "general_tech")

    task_summaries = []
    for t in jd_tasks[:12]:
        task_summaries.append(
            f"  T{t['task_id']}: [{t['priority'].upper()}] [{t['task_category']}] "
            f"{t['task_description']} (tech: {', '.join(t.get('implied_technologies', [])[:2])})")
    tasks_str = "\n".join(task_summaries)
    companies_str = ", ".join(experience_companies[:4]) or "Company 1-4"

    prompt = f"""Plan 12 resume bullets across 4 experience blocks (3 each).
Candidate had internships at: {companies_str}
Applying for {target_role} at {target_company}. Archetype: {archetype_key}.

JD TEXT (first 3000 chars):
{jd_text[:3000]}

JD TASKS:
{tasks_str}

JD KEYWORDS: {json.dumps(jd_keywords[:25])}

IDEAL CANDIDATE:
- Must-haves: {json.dumps(top_3)}
- Implicit: {json.dumps([r.get('requirement', '') for r in implicit_reqs[:4]])}

BLOCK ASSIGNMENT RULES:
- Blocks 0 and 1 (6 bullets total): These are the two most recent internships. Bullets
  should naturally demonstrate work related to the JD's top-priority tasks. Use JD technology
  names where the candidate can defensibly claim experience, but do not force technologies
  the candidate hasn't used. Describe adjacent or related work if exact JD tech isn't defensible.
- Block 2 (3 bullets): Medium-priority JD tasks. Address supporting tasks like
  data pipelines, testing, monitoring, or automation that the JD mentions.
- Block 3 (3 bullets): Foundational work. Demonstrate foundational skills the JD
  assumes — version control, basic data processing, scripting, documentation.

IMPORTANT: Bullets should sound like believable intern work. Do not force exact JD
phrases into bullets. Use JD themes naturally. Every bullet must be something the
candidate can explain in a 2-minute interview answer.

ANTI-PATTERN CHECK: If a bullet sounds like marketing copy or keyword stuffing, rewrite it
to sound like a real intern describing their work.

Return STRICT JSON:
{{
    "bullet_plan": [
        {{
            "bullet_index": 0, "block_index": 0,
            "experience_company": "company name",
            "assigned_jd_task": "the JD task this bullet mirrors",
            "task_id": 1,
            "task_category": "build_system|analyze_data|train_model|deploy_service|build_pipeline|optimize_performance|automate_process|collaborate|research|monitor",
            "bullet_seed": "1-sentence past-tense description, SPECIFIC to this JD",
            "primary_technology": "specific technology FROM THE JD",
            "supporting_keywords": ["2-3 JD keywords"],
            "result_type": "metric|qualitative_insight|efficiency_gain|reliability|collaboration",
            "has_metric": true,
            "metric_hint": "specific non-percentage metric"
        }},
        ... (exactly 12)
    ],
    "task_coverage": {{"high_priority_covered": 3, "total_covered": 10, "uncovered": []}}
}}

RULES:
1. ALL high-priority tasks MUST be covered
2. Each bullet addresses a DIFFERENT task
3. Exactly 1 bullet per block gets a metric (4 metrics total). The other 2 use qualitative outcomes.
4. Block 0 = most recent (advanced), Block 3 = oldest (simpler)
5. METRIC DIVERSITY: The 4 metric bullets should each use a different format.
6. All bullets must be interview-defensible and intern-scale.
7. Do not copy JD sentences verbatim into bullet seeds.
"""
    try:
        data = await gpt_json(prompt, temperature=0.25)
        plan = data.get("bullet_plan", [])
        while len(plan) < 12:
            idx = len(plan)
            block = idx // 3
            t = jd_tasks[idx % len(jd_tasks)] if jd_tasks else {}
            tc = t.get("task_category", "build_system")
            plan.append({
                "bullet_index": idx, "block_index": block,
                "assigned_jd_task": t.get("task_description", "Technical work"),
                "jd_line_reference": "",
                "task_id": t.get("task_id", 1), "task_category": tc,
                "bullet_seed": t.get("what_good_looks_like", "Technical contribution"),
                "primary_technology": (t.get("implied_technologies", ["Python"]) or ["Python"])[0],
                "supporting_keywords": [], "jd_phrases_to_mirror": [],
                "result_type": "qualitative_insight",
                "has_metric": idx % 3 == 0, "metric_hint": pick_metric_hint(tc) if idx % 3 == 0 else "",
            })
        _log("📋 [MASTER PLAN] 12 bullets planned")
        return {"bullet_plan": plan[:12], "task_coverage": data.get("task_coverage", {})}
    except Exception as e:
        _log(f"⚠️ [PLAN] Failed: {e}")
        plan = []
        for i in range(12):
            block = i // 3
            t = jd_tasks[i % len(jd_tasks)] if jd_tasks else {}
            tc = t.get("task_category", "build_system")
            plan.append({
                "bullet_index": i, "block_index": block,
                "assigned_jd_task": t.get("task_description", "Technical work"),
                "jd_line_reference": "",
                "task_id": t.get("task_id", 1), "task_category": tc,
                "bullet_seed": t.get("what_good_looks_like", "Technical contribution"),
                "primary_technology": (t.get("implied_technologies", ["Python"]) or ["Python"])[0],
                "supporting_keywords": [], "jd_phrases_to_mirror": [],
                "result_type": "qualitative_insight",
                "has_metric": i % 3 == 0, "metric_hint": pick_metric_hint(tc) if i % 3 == 0 else "",
            })
        return {"bullet_plan": plan, "task_coverage": {}}


_global_kw_assignments: Dict[str, int] = {}


def reset_keyword_assignment_tracking():
    global _global_kw_assignments
    _global_kw_assignments.clear()

def build_genuine_jd_bullet_prompt(
    experience_company: str,
    original_title: str,
    target_company: str,
    target_role: str,
    jd_tasks: List[Dict[str, Any]],
    bullet_plans: List[Dict[str, Any]],
    candidate_inventory: Dict[str, Any],
    role_archetype: Dict[str, Any],
    verbs: List[str],
    jd_text: str = "",
) -> str:
    return f"""
{GENUINE_JD_OPTIMIZED_STYLE}

Write exactly 3 resume bullets for this experience.

Company / institution:
{experience_company}

Original or current title:
{original_title}

Target company:
{target_company}

Target role:
{target_role}

Candidate capability inventory:
{json.dumps(candidate_inventory, indent=2)[:4000]}

Relevant JD tasks/themes:
{json.dumps(jd_tasks[:8], indent=2)[:3000]}

Bullet plan:
{json.dumps(bullet_plans[:3], indent=2)[:2000]}

Required starting verbs, in order:
1. {verbs[0] if len(verbs) > 0 else "Built"}
2. {verbs[1] if len(verbs) > 1 else "Wrote"}
3. {verbs[2] if len(verbs) > 2 else "Tested"}

Rules:
- Generate new bullet wording from scratch.
- Do not copy old resume bullet phrasing.
- Strongly align with the JD, but use natural themes rather than copied JD phrases.
- Use only defensible tools and domains from the candidate inventory.
- If a JD technology is not defensible, describe adjacent work instead.
- Each bullet must mention a concrete artifact: dashboard, component, script, notebook, model, report, check, pipeline, API, test, dataset, or workflow.
- Each bullet must include a believable outcome.
- At most 1 bullet may contain a metric.
- Metrics must be approximate and intern-scale: about, roughly, ~, under, around.
- Do not use exact suspicious metrics like 9%, 17.3%, 98.7%, 5,000+, millions, billions, or sub-10ms latency.
- Non-metric bullets should use qualitative outcomes like "which the team adopted as the default" or "that replaced the previous manual workflow."
- Keep each bullet 18-30 words.
- Write like a real early-career engineer: clear, specific, slightly understated.
- Do not use hype words: spearheaded, leveraged, utilized, orchestrated, pioneered, championed, revolutionized, robust, scalable, comprehensive, seamless, cutting-edge, mission-critical, enterprise-grade.
- Do not use malformed artifacts: T3, T4, XYZ, ABC, TODO, Dell T.
- NEVER include the experience company name in the bullet text. No "at {experience_company}" or any variation. The company name is already shown in the resume subheading — repeating it wastes words and looks robotic.
- NEVER include the target company name "{target_company}" in the bullet text. Bullets describe past work, not the company being applied to.
- Avoid vague phrases: actionable insights, data-driven experiences, stakeholder visibility, business impact, customer-facing analytics experiences.
- Do not claim ownership of production systems unless clearly supported.
- Do not make every bullet mention the same tool.

Good style:
"Rebuilt dashboard filter components in React, making weekly Power BI views easier to update without changing each page by hand."

Bad style:
"Spearheaded robust dashboard modernization by leveraging React and Power BI to deliver scalable, data-driven stakeholder visibility."

Return STRICT JSON:
{{"bullets": ["...", "...", "..."]}}
"""
def _build_bullet_prompt_v2(
    experience_company: str, target_company: str, target_role: str,
    block_index: int, total_blocks: int, suggested_verbs: List[str],
    bullet_plans: List[Dict], role_archetype: Dict, exp_context: Dict,
    progression: Dict, jd_text: str, all_keywords: List[str],
    already_used_kws: List[str],
    candidate_inventory: Dict[str, Any] = None,
) -> str:
    ak = role_archetype.get("key", "general_tech")
    bf = role_archetype.get("bullet_focus", "")
    comp = progression.get("complexity", "intermediate")
    auton = progression.get("autonomy", "with mentorship")
    domain = exp_context.get("domain", "Technology")
    unreal_tech = exp_context.get("unrealistic_technologies", [])

    verb1 = suggested_verbs[0] if len(suggested_verbs) > 0 else "Built"
    verb2 = suggested_verbs[1] if len(suggested_verbs) > 1 else "Wrote"
    verb3 = suggested_verbs[2] if len(suggested_verbs) > 2 else "Tested"

    seeds = []
    for i, bp in enumerate(bullet_plans[:3]):
        task = bp.get("assigned_jd_task", "Technical contribution")
        seed = bp.get("bullet_seed", "")
        tech = bp.get("primary_technology", "")
        has_m = bp.get("has_metric", False)
        m_hint = bp.get("metric_hint", "")
        skws = bp.get("supporting_keywords", [])
        tc = bp.get("task_category", "build_system")
        verb = suggested_verbs[i] if i < len(suggested_verbs) else "Built"
        s = (f"Bullet {i + 1} (verb: {verb}):\n"
             f"   JD TASK: {task}\n   SEED: {seed}\n   TECHNOLOGY: {tech}\n")
        if skws:
            s += f"   RELATED KEYWORDS: {', '.join(skws[:3])}\n"
        if has_m:
            hint = m_hint if m_hint else pick_metric_hint(tc)
            s += f"   METRIC (approximate, intern-scale): {hint}\n"
        else:
            s += f"   RESULT: Qualitative outcome — what happened, who used it, what it replaced\n"
        seeds.append(s)

    dedup_note = ""
    if already_used_kws:
        dedup_note = f"\nAlready used (avoid as PRIMARY): {', '.join(already_used_kws[:15])}\n"

    company_constraint = (
        f"COMPANY CONTEXT: {experience_company} ({domain}).\n"
        f"   The candidate interned here. Write bullets for work relevant to the target role.\n")
    if unreal_tech:
        company_constraint += f"   DO NOT MENTION: {', '.join(unreal_tech[:4])}\n"

    inventory_section = ""
    if candidate_inventory:
        inventory_section = f"""
═══ CANDIDATE CAPABILITY INVENTORY ═══
{json.dumps(candidate_inventory, indent=2)[:3000]}
"""

    return f"""{GENUINE_JD_OPTIMIZED_STYLE}

Write exactly 3 resume bullets for an intern who worked at "{experience_company}" and is applying to {target_role} at {target_company}.

═══ JOB DESCRIPTION ═══
{jd_text[:3000]}

═══ CONTEXT ═══
Company: {experience_company} ({domain})
Block: {block_index}/{total_blocks} ({comp} work, {auton})
Archetype: {ak} — {bf}
{company_constraint}
{inventory_section}

═══ BULLET SEEDS ═══
{chr(10).join(seeds)}

═══ KEYWORDS TO WEAVE IN NATURALLY ═══
{', '.join(all_keywords[:15])}
{dedup_note}

═══ ASSIGNED STARTING VERBS (use exactly these, in order) ═══
Bullet 1: {verb1}
Bullet 2: {verb2}
Bullet 3: {verb3}

═══ BANNED WORDS ═══
Spearheaded, Leveraged, Utilized, Orchestrated, Pioneered, Championed, Harnessed,
Galvanized, Operationalized, Productionized, Architected, Assembled, Revolutionized,
Crafted, Navigated, Steered, Helmed, Forged, Comprehensive, Robust, Scalable,
Seamless, Cutting-edge, Mission-critical, Enterprise-grade

═══ BANNED PATTERNS ═══
- "resulting in", "thereby", "thus enabling"
- "robust and scalable", "end-to-end solution", "holistic approach"
- "actionable insights", "data-driven experiences", "stakeholder visibility"
- Any malformed artifacts: T3, T4, Dell T, XYZ, ABC, TODO
- NEVER mention the experience company name in the bullet text. No "at {experience_company}", no "at IIT Indore", no "at National Institute of Technology". The bullet must stand alone without naming where the work happened — the company name already appears in the resume subheading above the bullets.
- NEVER mention the target company name "{target_company}" in the bullet text. Bullets describe YOUR past work, not the company you are applying to. No "for {target_company}", no "aligned with {target_company}'s needs", no "{target_company}-style".

═══ METRIC RULES ═══
{METRIC_RULES}

═══ LANGUAGE RULES ═══
- Use the JD's technology names naturally, not as keyword stuffing.
- Use human connectors: which, so, after, because, that — not "resulting in" or "thereby."
- Vary sentence length. Some bullets 20 words, some 28.
- Each bullet must mention a concrete artifact: dashboard, script, component, notebook, model, pipeline, API, test, dataset, or workflow.
- Non-metric bullets end with qualitative outcomes.

═══ REQUIRED SENTENCE STRUCTURES — use a DIFFERENT structure for each bullet ═══
Structure A — MOTIVATION-FIRST: "[Verb] X because Y was broken/slow/missing — [result]."
Structure B — NARRATIVE: "[Verb] the X system and [second action], which [result]."
Structure C — CASUAL-SPECIFIC: "[Verb] a [specific thing] that [what it did] — [outcome]."
Structure D — OBSERVATION-ACTION: "[Verb] that X was [problem], [action], [what happened]."
Structure E — PLAIN BUILDER: "[Verb] [specific thing] for [purpose] using [tool]."

Return STRICT JSON:
{{"bullets":["{verb1}...", "{verb2}...", "{verb3}..."],
"structures_used":["A","D","C"],
"keywords_used":["kw1","kw2"],
"technologies_used":["t1","t2"]}}"""

async def _post_process(
    bullets: List[str], kws: List[str], techs: List[str],
    n: int, start_pos: int, verbs: List[str], all_kws: List[str],
) -> Tuple[List[str], Set[str]]:
    cleaned, used = [], set()
    for i, b in enumerate(bullets[:n]):
        b = str(b).strip()
        b = await fix_capitalization_gpt(b)
        if i < len(verbs):
            fw = b.split()[0] if b.split() else ""
            if fw.lower() != verbs[i].lower():
                if verbs[i].lower() in b.lower():
                    b = re.sub(rf'\b{re.escape(verbs[i])}\b', '', b, count=1, flags=re.I).strip()
                if b:
                    b = f"{verbs[i]} {b.lstrip()}"
                else:
                    b = verbs[i]
        dm = re.search(
            r',?\s+\b(by|of|to|from|through|via|using|across|with|achieving|improving|'
            r'enhancing|boosting|increasing|reducing|raising|lifting)\s*[.,]?\s*$', b, re.I)
        if dm:
            b = b[:dm.start()].rstrip(".,;: ") + "."
        b = adjust_bullet_length(b)
        if not b.endswith("."):
            b = b.rstrip(".,;: ") + "."
        b = deterministic_metric_placeholder_cleanup(b)
        b = latex_escape_text(b)
        if b:
            cleaned.append(b)
            for kw in all_kws:
                if isinstance(kw, str) and kw.lower() in b.lower():
                    used.add(kw.lower())
            if i < len(kws) and isinstance(kws[i], str):
                pk = kws[i].lower().strip()
                if pk and pk not in _global_kw_assignments:
                    _global_kw_assignments[pk] = start_pos + i
    return cleaned, used


async def generate_block_bullets(
    jd_text: str, exp_company: str, target_company: str, target_role: str,
    block_index: int, total_blocks: int, start_pos: int,
    plans: List[Dict], role_archetype: Dict, all_keywords: List[str],
    used_keywords: Set[str], n: int = 3,
    candidate_inventory: Dict[str, Any] = None,
) -> Tuple[List[str], Set[str]]:
    exp_ctx = await get_company_context_gpt(exp_company, jd_text=jd_text, target_role=target_role)
    prog = get_progression_context(block_index, total_blocks)
    verbs = []
    for bp in plans[:n]:
        tc = bp.get("task_category", "build_system")
        vc = TASK_CAT_TO_VERB_CAT.get(tc, "build")
        verbs.append(get_diverse_verb(vc))
    while len(verbs) < n:
        verbs.append(get_diverse_verb("build"))
    prompt = _build_bullet_prompt_v2(
        exp_company, target_company, target_role, block_index, total_blocks,
        verbs, plans, role_archetype, exp_ctx, prog, jd_text, all_keywords,
        list(_global_kw_assignments.keys()),
        candidate_inventory=candidate_inventory)
    cleaned, used = [], set()
    for attempt in range(3):
        try:
            temp = 0.35 + (attempt * 0.12)
            data = await gpt_json(prompt, temperature=temp)
            bullets = data.get("bullets", []) or []
            if not bullets or len(bullets) < n:
                _log(f"⚠️ [BLOCK {block_index}] Attempt {attempt + 1}: {len(bullets)} bullets")
                continue

            # Validate structure diversity
            structures_used = data.get("structures_used", [])
            if structures_used and len(structures_used) >= n:
                structures_used = [s.upper().strip() for s in structures_used[:n]]
                if not validate_structures(structures_used, block_index):
                    _log(f"⚠️ [BLOCK {block_index}] Attempt {attempt + 1}: structure diversity violated, retrying")
                    # Add explicit instruction to use different structures on retry
                    if attempt < 2:
                        prompt = prompt.replace(
                            "Never use the same structure twice in the same block.",
                            f"Never use the same structure twice in the same block.\n"
                            f"ALREADY USED STRUCTURES (avoid reusing more than twice): "
                            f"{', '.join(_used_structures)}")
                    continue

            cleaned, used = await _post_process(
                bullets, data.get("keywords_used", []),
                data.get("technologies_used", []), n, start_pos, verbs, all_keywords)
            if len(cleaned) >= n:
                # Record structures only after successful generation
                if structures_used and len(structures_used) >= n:
                    record_structures(structures_used[:n])
                break
        except Exception as e:
            _log(f"⚠️ [BLOCK {block_index}] Attempt {attempt + 1} error: {e}")
    while len(cleaned) < n:
        idx = len(cleaned)
        bp = plans[idx] if idx < len(plans) else {}
        verb = verbs[idx] if idx < len(verbs) else "Built"
        task = bp.get("assigned_jd_task", "technical contribution")
        tech = bp.get("primary_technology", "Python")
        tc = bp.get("task_category", "build_system")

        for micro_attempt in range(2):
            try:
                fallback_prompt = (
                    f'Write ONE resume bullet for an intern applying to {target_role} at {target_company}. '
                    f'The intern previously worked at {exp_company}. '
                    f'This bullet must demonstrate: "{task}" '
                    f'Start with "{verb}". Mention {tech} naturally. '
                    f'Use a qualitative outcome, not a metric. '
                    f'20-28 words, past tense. No placeholder words. No hype words. '
                    f'No malformed artifacts like T3, T4, Dell T. '
                    f'Return STRICT JSON: {{"bullet":"{verb} ..."}}'
                )
                mb = await gpt_json(fallback_prompt, temperature=0.4 + micro_attempt * 0.15)
                bullet = mb.get("bullet", "")
                if bullet and len(bullet.split()) >= 10:
                    bullet = await fix_capitalization_gpt(bullet)
                    bullet = adjust_bullet_length(bullet)
                    if not bullet.endswith("."):
                        bullet = bullet.rstrip(".,;: ") + "."
                    cleaned.append(latex_escape_text(bullet))
                    break
            except Exception:
                pass
    _log(f"✅ [BLOCK {block_index}] {len(cleaned)} bullets for {exp_company}")
    return cleaned[:n], used


# ═══════════════════════════════════════════════════════════════
# POST-GENERATION VALIDATION, DEDUP, RUBRIC
# ═══════════════════════════════════════════════════════════════

async def validate_and_fix_task_alignment(
    all_bullets: List[List[str]], bullet_plan: List[Dict],
    jd_text: str, target_role: str, role_archetype: Dict,
) -> List[List[str]]:
    flat_bullets = [b for block in all_bullets for b in block]
    if len(flat_bullets) < 6:
        return all_bullets
    checks = [
        {"idx": i, "bullet": flat_bullets[i][:180],
         "assigned_task": (bullet_plan[i].get("assigned_jd_task", "")[:100]
                           if i < len(bullet_plan) else "")}
        for i in range(min(len(flat_bullets), len(bullet_plan)))]
    prompt = f"""Rate each resume bullet on JD-SPECIFICITY (0.0-1.0).
Target role: {target_role}

JD TEXT (first 2000 chars):
{jd_text[:2000]}

BULLETS:
{json.dumps(checks)}

Score each bullet on JD-SPECIFICITY:
- 1.0 = Bullet uses the JD's exact technologies, exact terminology, and directly
        demonstrates the assigned task. A recruiter would highlight this bullet.
- 0.7 = Bullet demonstrates the right type of work but uses generic language
        instead of JD-specific terms (e.g., "database" instead of "PostgreSQL")
- 0.4 = Bullet is tangentially related to the task but doesn't mirror JD language
- 0.0 = Bullet is irrelevant to the assigned JD task

ALSO FLAG: Any bullet that contains a technology NOT mentioned in the JD
(unless it's a standard tool like Git, Linux, or Python). Flag these for
replacement with JD-mentioned technologies.

Return STRICT JSON: {{"results": [{{"idx": 0, "score": 0.9, "reason": "brief", "non_jd_tech": ["tech not in JD"]}}]}}
"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        results = {}
        for r in (data.get("results") or []):
            if not isinstance(r, dict): continue
            try: idx = int(r.get("idx", -1))
            except (TypeError, ValueError): continue
            results[idx] = r
    except Exception:
        _log("⚠️ [VALIDATION] Scoring failed")
        return all_bullets

    low_scoring = sorted(
        [r for r in results.values() if r.get("score", 1.0) < 0.45],
        key=lambda x: x.get("score", 1.0))
    _log(f"🔍 [VALIDATION] {len(low_scoring)} bullets scored < 0.45 out of {len(flat_bullets)}")

    for r in low_scoring[:3]:
        try: idx = int(r.get("idx", -1))
        except (TypeError, ValueError): continue
        if idx < 0 or idx >= len(flat_bullets): continue
        plan = bullet_plan[idx] if idx < len(bullet_plan) else {}
        task = plan.get("assigned_jd_task", "")
        tech = plan.get("primary_technology", "Python")
        tc = plan.get("task_category", "build_system")
        verb = re.sub(r"\\[#$%&_{}]", "",
                      flat_bullets[idx].split()[0]) if flat_bullets[idx].split() else "Built"
        try:
            fix = await gpt_json(
                f'Rewrite this bullet to better demonstrate: "{task}"\n'
                f'CURRENT: "{flat_bullets[idx][:200]}"\n'
                f'Start with "{verb}". Mention {tech} naturally. 18-30 words.\n'
                f'Use plain professional language. No hype words. No placeholder words.\n'
                f'No malformed artifacts like T3, T4, Dell T.\n'
                f'Return STRICT JSON: {{"bullet": "..."}}',
                temperature=0.35)
            new_b = fix.get("bullet", "")
            if new_b and len(new_b.split()) >= 15:
                new_b = await fix_capitalization_gpt(new_b)
                new_b = adjust_bullet_length(new_b)
                if not new_b.endswith("."): new_b = new_b.rstrip(".,;: ") + "."
                flat_bullets[idx] = latex_escape_text(new_b)
                _log(f"✅ [REALIGN] idx={idx} → rewritten")
        except Exception: pass
    result, i = [], 0
    for block in all_bullets:
        result.append(flat_bullets[i:i + len(block)])
        i += len(block)
    return result


async def deduplicate_across_blocks(
    all_bullets: List[List[str]], bullet_plan: List[Dict],
    jd_text: str, role_archetype: Dict,
) -> List[List[str]]:
    flat = [b for block in all_bullets for b in block]
    if len(flat) < 6:
        return all_bullets
    prompt = f"""Check these resume bullets for semantic redundancy.

{json.dumps([{"idx": i, "bullet": b[:150] if isinstance(b, str) else str(b)[:150]}
             for i, b in enumerate(flat[:12])])}

Return STRICT JSON:
{{"duplicate_pairs": [{{"idx_a": 0, "idx_b": 5, "reason": "..."}}], "all_unique": true}}
"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        pairs = data.get("duplicate_pairs", [])
    except Exception:
        return all_bullets
    if not pairs:
        _log("✅ [DEDUP] All bullets are unique")
        return all_bullets
    _log(f"🔄 [DEDUP] Found {len(pairs)} redundant pairs")
    rewritten_indices: Set[int] = set()
    for pair in pairs[:3]:
        if not isinstance(pair, dict): continue
        try:
            idx_a = int(pair.get("idx_a", -1))
            idx_b = int(pair.get("idx_b", -1))
        except (TypeError, ValueError): continue
        if idx_a < 0 or idx_b < 0: continue
        rewrite_idx = max(idx_a, idx_b)
        if rewrite_idx >= len(flat) or rewrite_idx in rewritten_indices: continue
        plan = bullet_plan[rewrite_idx] if rewrite_idx < len(bullet_plan) else {}
        task = plan.get("assigned_jd_task", "different technical work")
        tech = plan.get("primary_technology", "Python")
        tc = plan.get("task_category", "build_system")
        verb = re.sub(r"\\[#$%&_{}]", "",
                      flat[rewrite_idx].split()[0]) if flat[rewrite_idx].split() else "Built"
        try:
            fix = await gpt_json(
                f'Rewrite to focus on: "{task}"\n'
                f'Mention {tech} naturally. Start with "{verb}". 18-30 words. '
                f'Use a qualitative outcome, not a metric. No placeholder words.\n'
                f'No malformed artifacts like T3, T4, Dell T. Plain professional language.\n'
                f'Return STRICT JSON: {{"bullet": "..."}}',
                temperature=0.4)
            new_b = fix.get("bullet", "")
            if new_b and len(new_b.split()) >= 15:
                new_b = await fix_capitalization_gpt(new_b)
                new_b = adjust_bullet_length(new_b)
                if not new_b.endswith("."): new_b = new_b.rstrip(".,;: ") + "."
                flat[rewrite_idx] = latex_escape_text(new_b)
                rewritten_indices.add(rewrite_idx)
                _log(f"✅ [DEDUP] Bullet {rewrite_idx} rewritten")
        except Exception: pass
    result, i = [], 0
    for block in all_bullets:
        result.append(flat[i:i + len(block)])
        i += len(block)
    return result

async def score_bullet_quality_rubric(
    all_bullets: List[List[str]], bullet_plan: List[Dict],
    jd_text: str, target_role: str,
) -> List[List[str]]:
    flat = [b for block in all_bullets for b in block]
    if len(flat) < 3:
        return all_bullets
    checks = []
    for i, b in enumerate(flat):
        plan = bullet_plan[i] if i < len(bullet_plan) else {}
        checks.append({"idx": i, "bullet": b[:180],
                        "assigned_task": plan.get("assigned_jd_task", "")[:80]})
    prompt = f"""Score each resume bullet on 6 axes (0-3). Target role: {target_role}

JD TEXT (first 2000 chars):
{jd_text[:2000]}

AXES (score 0-3 each):
  task_specificity, jd_phrase_usage, metric_quality, verb_strength, jd_vocabulary_overlap, human_voice

task_specificity scoring:
  3 = bullet names a specific artifact, dataset, or system component
  2 = bullet describes a clear type of work but lacks specifics
  1 = bullet is vaguely technical
  0 = bullet is generic filler

jd_phrase_usage scoring:
  3 = bullet mirrors 2+ exact multi-word phrases from the JD
  2 = bullet uses 1 exact JD phrase
  1 = bullet uses JD vocabulary but not exact phrases
  0 = no JD language detected

metric_quality scoring:
  3 = if bullet has a metric: approximate, intern-scale with hedging ("about", "~", "roughly"). If no metric: qualitative outcome describing what happened ("which the team adopted", "that replaced the manual workflow") — this also scores 3.
  2 = metric present but generic ("improved by X%"), or qualitative outcome is vague
  1 = vague quantification ("significantly improved") or no outcome at all
  0 = metric looks fake (exact odd digits, unrealistic scale like "13K requests" for an intern, sub-10ms latency claims) OR bullet was supposed to have a metric and doesn't

verb_strength scoring:
  3 = strong, specific action verb that fits the work described
  2 = acceptable verb but could be more precise
  1 = weak or generic verb (e.g., "Worked on", "Helped with")
  0 = banned/AI-sounding verb (Spearheaded, Leveraged, etc.)

jd_vocabulary_overlap scoring:
  3 = bullet contains 4+ words/phrases found verbatim in the JD
  2 = bullet contains 2-3 words/phrases found verbatim in the JD
  1 = bullet contains 1 word/phrase found verbatim in the JD
  0 = bullet contains no JD-specific vocabulary

human_voice scoring:
  3 = plain professional language — natural sentence flow, concrete artifacts, no template patterns, does NOT mention any experience company/institution name in the bullet
  2 = mostly natural but has one formulaic phrase ("resulting in", "utilizing", one buzzword)
  1 = reads like a resume template — predictable verb-noun-metric structure, buzzwords present
  0 = obvious AI/ChatGPT resume bullet — multiple banned words, "resulting in a X% improvement" pattern, OR bullet mentions the experience company name (e.g., "at IIT Indore", "at National Institute of Technology, Jaipur") — company names in bullets are always score 0

BULLETS:
{json.dumps(checks[:12])}

Return STRICT JSON:
{{"scores": [{{"idx": 0, "task_specificity": 2, "jd_phrase_usage": 1, "metric_quality": 3,
              "verb_strength": 2, "jd_vocabulary_overlap": 2, "human_voice": 2,
              "total": 12, "weakest_axis": "jd_phrase_usage"}}]}}
"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        scores = {}
        for s in (data.get("scores") or []):
            if not isinstance(s, dict): continue
            try: idx = int(s.get("idx", -1))
            except (TypeError, ValueError): continue
            scores[idx] = s
    except Exception:
        _log("⚠️ [RUBRIC] Scoring failed")
        return all_bullets

    # Flag bullets scoring <= 6/18 overall OR 0 on human_voice — only severe issues
    weak = []
    for s in scores.values():
        total = s.get("total", 18)
        hv = s.get("human_voice", 3)
        if total <= 6 or hv <= 0:
            weak.append(s)
    weak.sort(key=lambda x: x.get("total", 18))
    _log(f"📐 [RUBRIC] {len(weak)} bullets scored ≤ 6/18 or human_voice = 0 out of {len(flat)}")

    axis_instructions = {
        "task_specificity": "Name the exact artifact, problem, dataset.",
        "jd_phrase_usage": "Mirror the JD's vocabulary.",
        "metric_quality": "Use a CONCRETE NUMBER (odd digit). Counts, latency, time — not just %.",
        "verb_strength": "Use strong verb: Built, Shipped, Trained, Automated — not Spearheaded/Leveraged.",
        "jd_vocabulary_overlap": "Replace generic terms with the JD's exact vocabulary. Read the JD and find the specific words they use.",
        "human_voice": "Rewrite this bullet in plain professional language. Keep the technical content. Remove hype words and template patterns.",
    }
    for s in weak[:6]:
        try: idx = int(s.get("idx", -1))
        except (TypeError, ValueError): continue
        if idx < 0 or idx >= len(flat): continue
        plan = bullet_plan[idx] if idx < len(bullet_plan) else {}
        task = plan.get("assigned_jd_task", "")
        tech = plan.get("primary_technology", "Python")
        tc = plan.get("task_category", "build_system")
        verb = re.sub(r"\\[#$%&_{}]", "",
                      flat[idx].split()[0]) if flat[idx].split() else "Built"
        weakest = s.get("weakest_axis", "task_specificity")
        hv_score = s.get("human_voice", 3)

        # If human_voice is the problem (0-1), override the improvement instruction
        if hv_score <= 1:
            improvement = axis_instructions["human_voice"]
            weakest = "human_voice"
        else:
            improvement = axis_instructions.get(weakest, "Improve specificity.")

        try:
            fix = await gpt_json(
                f'Improve this bullet. Score={s.get("total", "?")}/18, weakest: {weakest}.\n'
                f'CURRENT: "{flat[idx][:200]}"\nTASK: "{task}"\nIMPROVEMENT: {improvement}\n'
                f'Start with "{verb}". Mention {tech}. Metric: {pick_metric_hint(tc)}\n'
                f'Use human connectors (which, so, after, because) not resume connectors '
                f'(resulting in, thereby, thus enabling).\n'
                f'24-34 words. No placeholder words.\n'
                f'Return STRICT JSON: {{"bullet": "..."}}',
                temperature=0.4)
            new_b = fix.get("bullet", "")
            if new_b and len(new_b.split()) >= 15:
                new_b = await fix_capitalization_gpt(new_b)
                new_b = adjust_bullet_length(new_b)
                if not new_b.endswith("."): new_b = new_b.rstrip(".,;: ") + "."
                flat[idx] = latex_escape_text(new_b)
                _log(f"✅ [RUBRIC] idx={idx} total={s.get('total')} hv={hv_score} → improved ({weakest})")
        except Exception: pass
    result, i = [], 0
    for block in all_bullets:
        result.append(flat[i:i + len(block)])
        i += len(block)
    return result

# ═══════════════════════════════════════════════════════════════
# COVERAGE REMEDIATION
# ═══════════════════════════════════════════════════════════════

async def remediate_coverage_gaps(
    all_bullets: List[List[str]], must_have_keywords: List[str],
    bullet_plan: List[Dict], jd_text: str,
) -> List[List[str]]:
    flat = [b for block in all_bullets for b in block]
    plain = " ".join(flat).lower()
    missing_must = [k for k in must_have_keywords if k.lower() not in plain]
    if not missing_must:
        _log("✅ [COVERAGE] All must-have keywords present")
        return all_bullets
    _log(f"⚠️ [COVERAGE] {len(missing_must)} must-have keywords missing: {missing_must[:7]}")

    # Score each bullet for JD-specificity to find weakest candidates
    def _jd_specificity_score(bullet: str, jd_lower: str) -> float:
        bl = bullet.lower()
        words = bl.split()
        if not words:
            return 0.0
        jd_word_hits = sum(1 for w in words if len(w) > 3 and w in jd_lower)
        return jd_word_hits / max(1, len(words))

    jd_lower = jd_text.lower()
    bullet_scores = [_jd_specificity_score(b, jd_lower) for b in flat]

    rewritten_indices: Set[int] = set()
    for kw in missing_must[:5]:
        if not flat:
            break
        # Find bullet with LOWEST JD-specificity score that hasn't been rewritten
        candidates = [(score, idx) for idx, score in enumerate(bullet_scores)
                       if idx not in rewritten_indices]
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0])
        weakest_idx = candidates[0][1]

        plan = bullet_plan[weakest_idx] if weakest_idx < len(bullet_plan) else {}
        task = plan.get("assigned_jd_task", "technical contribution")
        verb = re.sub(r"\\[#$%&_{}]", "",
                      flat[weakest_idx].split()[0]) if flat[weakest_idx].split() else "Built"
        try:
            fix = await gpt_json(
                f'Rewrite to naturally incorporate "{kw}" (must-have keyword).\n'
                f'CURRENT: "{flat[weakest_idx][:200]}"\nTASK: "{task}"\n'
                f'Keep verb "{verb}". 18-30 words. The keyword must fit naturally.\n'
                f'Do not force it in. Do not add hype words. Do not add metrics.\n'
                f'Plain professional language.\n'
                f'Return STRICT JSON: {{"bullet": "..."}}',
                temperature=0.3)
            new_b = fix.get("bullet", "")
            if new_b and kw.lower() in new_b.lower() and len(new_b.split()) >= 15:
                new_b = await fix_capitalization_gpt(new_b)
                new_b = adjust_bullet_length(new_b)
                if not new_b.endswith("."):
                    new_b = new_b.rstrip(".,;: ") + "."
                flat[weakest_idx] = latex_escape_text(new_b)
                rewritten_indices.add(weakest_idx)
                _log(f"✅ [REMEDIATE] Injected '{kw}' into bullet {weakest_idx} (score={candidates[0][0]:.2f})")
        except Exception:
            pass
    result, i = [], 0
    for block in all_bullets:
        result.append(flat[i:i + len(block)])
        i += len(block)
    return result

async def remediate_uncovered_responsibilities(
    all_bullets: List[List[str]], uncovered_responsibilities: List[str],
    bullet_plan: List[Dict], jd_text: str, all_keywords: List[str],
) -> List[List[str]]:
    """Rewrite the weakest bullets to cover JD responsibilities that no bullet addresses."""
    if not uncovered_responsibilities:
        _log("✅ [ATS RESP] All JD responsibilities covered")
        return all_bullets

    flat = [b for block in all_bullets for b in block]
    if len(flat) < 6:
        return all_bullets

    _log(f"⚠️ [ATS RESP] {len(uncovered_responsibilities)} uncovered responsibilities: "
         f"{uncovered_responsibilities[:3]}")

    jd_lower = jd_text.lower()

    def _jd_specificity_score(bullet: str) -> float:
        bl = bullet.lower()
        words = bl.split()
        if not words:
            return 0.0
        hits = sum(1 for w in words if len(w) > 3 and w in jd_lower)
        return hits / max(1, len(words))

    bullet_scores = [_jd_specificity_score(b) for b in flat]
    rewritten: Set[int] = set()

    for resp in uncovered_responsibilities[:3]:
        candidates = [(score, idx) for idx, score in enumerate(bullet_scores)
                       if idx not in rewritten]
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0])
        weakest_idx = candidates[0][1]

        plan = bullet_plan[weakest_idx] if weakest_idx < len(bullet_plan) else {}
        verb = re.sub(r"\\[#$%&_{}]", "",
                      flat[weakest_idx].split()[0]) if flat[weakest_idx].split() else "Built"

        try:
            fix = await gpt_json(
                f'Rewrite this resume bullet to demonstrate this JD responsibility:\n'
                f'RESPONSIBILITY: "{resp}"\n'
                f'CURRENT: "{flat[weakest_idx][:200]}"\n'
                f'Start with "{verb}". 18-30 words. Use plain professional language.\n'
                f'Do not copy JD wording verbatim — demonstrate the capability naturally.\n'
                f'Do not include any company name in the bullet — not the experience company, not the target company.\n'
                f'Return STRICT JSON: {{"bullet": "..."}}',
                temperature=0.35)
            new_b = fix.get("bullet", "")
            if new_b and len(new_b.split()) >= 15:
                new_b = await fix_capitalization_gpt(new_b)
                new_b = adjust_bullet_length(new_b)
                if not new_b.endswith("."):
                    new_b = new_b.rstrip(".,;: ") + "."
                flat[weakest_idx] = latex_escape_text(new_b)
                rewritten.add(weakest_idx)
                _log(f"✅ [ATS RESP] Bullet {weakest_idx} rewritten to cover: {resp[:60]}")
        except Exception as e:
            _log(f"⚠️ [ATS RESP] Failed for '{resp[:40]}': {e}")

    result, i = [], 0
    for block in all_bullets:
        result.append(flat[i:i + len(block)])
        i += len(block)
    return result

# ═══════════════════════════════════════════════════════════════
# JD ECHO VERIFICATION — v2.9.0 NEW
# ═══════════════════════════════════════════════════════════════

async def verify_jd_echo(
    all_bullets: List[List[str]], jd_text: str, jd_tasks: List[Dict],
    target_role: str, target_company: str,
) -> List[List[str]]:
    """Post-generation verification: natural JD alignment check.
    Only rewrites bullets that are both misaligned AND vague. Does not require exact JD phrases."""
    flat = [b for block in all_bullets for b in block]
    if len(flat) < 6:
        return all_bullets

    numbered = "\n".join(f"{i}. {b[:180]}" for i, b in enumerate(flat))
    prompt = f"""Check whether these resume bullets are naturally aligned with the target JD.

Do NOT require exact JD phrase copying.
Reward natural overlap in responsibilities, tools, and work themes.
Penalize keyword stuffing or copied JD wording.

JD:
{jd_text[:2500]}

BULLETS:
{numbered}

For each bullet, return:
- aligned: true/false
- alignment_score: 0-100
- keyword_stuffing_risk: low/medium/high
- copied_jd_language_risk: low/medium/high
- reason: short explanation

Return STRICT JSON:
{{"assessments": [
    {{"idx": 0, "aligned": true, "alignment_score": 75,
      "keyword_stuffing_risk": "low", "copied_jd_language_risk": "low",
      "reason": "short explanation"}}
]}}
"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        assessments = data.get("assessments", [])
        if not assessments:
            _log("⚠️ [JD ALIGN] No assessments returned")
            return all_bullets

        good_count = sum(1 for a in assessments if isinstance(a, dict) and a.get("alignment_score", 100) >= 45)
        weak_count = sum(1 for a in assessments if isinstance(a, dict) and a.get("alignment_score", 100) < 45)
        _log(f"🔍 [JD ALIGN] {good_count} aligned, {weak_count} weak (< 45)")

        # Only rewrite bullets with alignment_score < 45
        for a in assessments:
            if not isinstance(a, dict):
                continue
            try:
                idx = int(a.get("idx", -1))
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= len(flat):
                continue

            score = a.get("alignment_score", 100)
            if score >= 45:
                continue

            if not should_auto_rewrite("weak_alignment"):
                _log(f"ℹ️ [JD ALIGN] Bullet {idx} score={score} — flagged but not auto-rewriting")
                continue

            verb = re.sub(r"\\[#$%&_{}]", "",
                          flat[idx].split()[0]) if flat[idx].split() else "Built"
            reason = a.get("reason", "")
            try:
                fix = await gpt_json(
                    f'This resume bullet has weak JD alignment (score {score}/100).\n'
                    f'Reason: {reason}\n'
                    f'Rewrite to better reflect the target role themes, but do NOT copy JD wording.\n'
                    f'CURRENT: "{flat[idx][:200]}"\n'
                    f'Start with "{verb}". 18-30 words. Plain professional language.\n'
                    f'Return STRICT JSON: {{"bullet": "..."}}',
                    temperature=0.35)
                new_b = fix.get("bullet", "")
                if new_b and len(new_b.split()) >= 15:
                    new_b = await fix_capitalization_gpt(new_b)
                    new_b = adjust_bullet_length(new_b)
                    if not new_b.endswith("."):
                        new_b = new_b.rstrip(".,;: ") + "."
                    flat[idx] = latex_escape_text(new_b)
                    _log(f"✅ [JD ALIGN] Bullet {idx} rewritten (score was {score})")
            except Exception:
                pass

        _log(f"✅ [JD ALIGN] Verification complete")

    except Exception as e:
        _log(f"⚠️ [JD ALIGN] Verification failed: {e}")

    result, i = [], 0
    for block in all_bullets:
        result.append(flat[i:i + len(block)])
        i += len(block)
    return result

async def validate_genuine_resume_bullets(
    all_bullets: List[List[str]],
    jd_text: str,
    candidate_inventory: Dict[str, Any],
) -> Dict[str, Any]:
    """Final quality gate: check all bullets for genuineness and interview-defensibility."""
    flat = [b for block in all_bullets for b in block]
    prompt = f"""
Evaluate these generated resume bullets for genuine, JD-optimized, interview-defensible quality.

JD:
{jd_text[:3000]}

Candidate capability inventory:
{json.dumps(candidate_inventory, indent=2)[:4000]}

Bullets:
{json.dumps(flat, indent=2)}

Check for:
- AI-sounding language
- copied JD wording
- copied old resume wording
- unsupported tools or claims
- too many metrics
- suspicious exact metrics
- senior-level exaggeration
- malformed artifacts like T3, T4, Dell T, XYZ, ABC
- vague phrases like actionable insights or data-driven experiences
- repeated keywords across bullets

Return STRICT JSON:
{{
  "pass": true,
  "overall_score": 0-100,
  "issues": [
    {{
      "bullet_index": 0,
      "issue_type": "ai_tone|unsupported_claim|too_many_metrics|artifact|vague|keyword_stuffing|copied_language",
      "severity": "low|medium|high",
      "reason": "...",
      "suggested_fix": "..."
    }}
  ]
}}
"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        return data if isinstance(data, dict) else {"pass": True, "overall_score": 70, "issues": []}
    except Exception as e:
        _log(f"⚠️ [GENUINE VALIDATOR] Failed: {e}")
        return {"pass": True, "overall_score": 70, "issues": []}


async def apply_genuine_validation_fixes(
    all_bullets: List[List[str]],
    validation_result: Dict[str, Any],
    target_role: str,
) -> List[List[str]]:
    """Apply fixes from genuine validator — only for high-severity artifacts, impossible metrics, or broken grammar."""
    issues = validation_result.get("issues", [])
    if not issues:
        return all_bullets

    flat = [b for block in all_bullets for b in block]
    high_severity = [i for i in issues if isinstance(i, dict) and i.get("severity") == "high"]

    # Only auto-fix artifact, impossible_metric, broken_grammar issues
    fixable_types = {"artifact", "impossible_metric", "broken_grammar"}
    to_fix = [i for i in high_severity if i.get("issue_type", "") in fixable_types
              or (i.get("issue_type") == "ai_tone" and i.get("severity") == "high")]

    _log(f"🔍 [GENUINE] {len(issues)} total issues, {len(high_severity)} high severity, {len(to_fix)} auto-fixable")

    for issue in to_fix[:4]:
        idx = issue.get("bullet_index", -1)
        if idx < 0 or idx >= len(flat):
            continue

        suggested = issue.get("suggested_fix", "")
        verb = re.sub(r"\\[#$%&_{}]", "", flat[idx].split()[0]) if flat[idx].split() else "Built"

        try:
            # Apply deterministic cleanup first
            flat[idx] = deterministic_artifact_cleanup(flat[idx])

            if _ARTIFACT_PATTERNS.search(flat[idx]) or _PLACEHOLDER_PATTERNS.search(flat[idx]):
                fix = await gpt_json(
                    f'Fix this resume bullet. Issue: {issue.get("reason", "artifact/quality problem")}\n'
                    f'CURRENT: "{flat[idx][:200]}"\n'
                    f'Start with "{verb}". 18-30 words. Plain professional language.\n'
                    f'Do not add metrics. Do not copy JD language.\n'
                    f'Return STRICT JSON: {{"bullet": "..."}}',
                    temperature=0.3)
                new_b = fix.get("bullet", "")
                if new_b and len(new_b.split()) >= 12:
                    new_b = await fix_capitalization_gpt(new_b)
                    new_b = adjust_bullet_length(new_b)
                    if not new_b.endswith("."):
                        new_b = new_b.rstrip(".,;: ") + "."
                    flat[idx] = latex_escape_text(new_b)
                    _log(f"✅ [GENUINE] Fixed bullet {idx}: {issue.get('issue_type')}")
        except Exception:
            pass

    result, i = [], 0
    for block in all_bullets:
        result.append(flat[i:i + len(block)])
        i += len(block)
    return result

# ═══════════════════════════════════════════════════════════════
# EXPERIENCE REWRITER
# ═══════════════════════════════════════════════════════════════

async def rewrite_experience_section(
    tex: str, jd_text: str, jd_info: Dict, target_company: str, target_role: str,
    core_keywords: List[str], master_plan: Dict, role_archetype: Dict,
    all_keywords: List[str], jd_tasks: Optional[List[Dict]] = None,
    candidate_inventory: Dict[str, Any] = None,
) -> Tuple[str, Set[str]]:
    reset_verb_tracking()
    reset_keyword_assignment_tracking()
    reset_metric_type_tracking()
    reset_structure_tracking()
    set_current_archetype(role_archetype.get("key", "general_tech"))

    bullet_plan = master_plan.get("bullet_plan", [])
    exp_companies = await _extract_experience_companies(tex)
    exp_used: Set[str] = set()
    all_blocks: List[List[str]] = []
    exp_pat = section_rx("Experience")
    block_index = 0
    abs_pos = 0

    for m in exp_pat.finditer(tex):
        section = m.group(1)
        s_tag, e_tag = r"\resumeItemListStart", r"\resumeItemListEnd"
        i = 0
        while True:
            a = section.find(s_tag, i)
            if a < 0: break
            b = section.find(e_tag, a)
            if b < 0: break
            ec = (exp_companies[block_index]
                  if block_index < len(exp_companies) else f"Company {block_index + 1}")
            plans = [bp for bp in bullet_plan if bp.get("block_index") == block_index]
            while len(plans) < 3:
                tc = "build_system"
                plans.append({
                    "assigned_jd_task": "Technical contribution", "task_category": tc,
                    "bullet_seed": "", "primary_technology": "Python",
                    "supporting_keywords": [], "jd_phrases_to_mirror": [],
                    "has_metric": False, "metric_hint": pick_metric_hint(tc),
                    "result_type": "qualitative_insight",
                })
            bullets, used = await generate_block_bullets(
                jd_text, ec, target_company, target_role, block_index, 4, abs_pos,
                plans[:3], role_archetype, all_keywords, exp_used, 3,
                candidate_inventory=candidate_inventory)
            exp_used.update(used)
            all_blocks.append(bullets)
            block_index += 1
            abs_pos += 3
            i = b + len(e_tag)

    _log("🔍 [VALIDATE] Scoring bullet-task alignment...")
    all_blocks = await validate_and_fix_task_alignment(
        all_blocks, bullet_plan, jd_text, target_role, role_archetype)
    _log("🔄 [DEDUP] Checking cross-block redundancy...")
    all_blocks = await deduplicate_across_blocks(
        all_blocks, bullet_plan, jd_text, role_archetype)
    _log("📐 [RUBRIC] Running 6-axis quality rubric (includes human_voice)...")
    all_blocks = await score_bullet_quality_rubric(
        all_blocks, bullet_plan, jd_text, target_role)
    _log("📊 [METRIC DIV] Enforcing metric type diversity...")
    all_blocks = await enforce_metric_diversity(
        all_blocks, bullet_plan, target_role, jd_text)
    _log("🔍 [JD ECHO] Running recruiter-perspective JD echo verification...")
    all_blocks = await verify_jd_echo(
        all_blocks, jd_text, jd_tasks or [], target_role, target_company)
    must_have = jd_info.get("must_have", [])
    if must_have:
        _log("📊 [REMEDIATE] Checking must-have keyword coverage...")
        all_blocks = await remediate_coverage_gaps(all_blocks, must_have, bullet_plan, jd_text)
    _log("🤖 [AI-TELL] Running AI-tell post-filter...")
    all_blocks = await ai_tell_post_filter(all_blocks, target_role, target_company)
    _log("🔍 [METRIC BELIEVE] Running metric believability verification...")
    all_blocks = await verify_metric_believability(
        all_blocks, target_role, target_company, jd_text)
    _log("🔧 [SANITIZE] Checking for placeholder words (XYZ, ABC, etc.)...")
    all_blocks = await sanitize_all_bullets(
        all_blocks, target_company, target_role, exp_companies)

    # Final company-name scrub across all bullets
    _log("🔧 [COMPANY SCRUB] Removing company names and fixing metric placeholders...")
    for block_idx, block in enumerate(all_blocks):
        for bullet_idx, bullet in enumerate(block):
            cleaned = deterministic_metric_placeholder_cleanup(bullet)
            cleaned = deterministic_company_name_cleanup(cleaned, exp_companies, target_company)
            if cleaned != bullet:
                all_blocks[block_idx][bullet_idx] = cleaned
                _log(f"   ✅ Removed company name from block {block_idx}, bullet {bullet_idx}")

    _log("🔍 [GENUINE] Running final genuine-quality validator...")
    if candidate_inventory:
        genuine_result = await validate_genuine_resume_bullets(
            all_blocks, jd_text, candidate_inventory or {})
        overall_score = genuine_result.get("overall_score", 100)
        issue_count = len(genuine_result.get("issues", []))
        _log(f"🔍 [GENUINE] Score={overall_score}/100, {issue_count} issues")
        if not genuine_result.get("pass", True) or overall_score < 60:
            all_blocks = await apply_genuine_validation_fixes(
                all_blocks, genuine_result, target_role)

    out, pos, block_idx = [], 0, 0
    for m in exp_pat.finditer(tex):
        out.append(tex[pos:m.start()])
        section = m.group(1)
        s_tag, e_tag = r"\resumeItemListStart", r"\resumeItemListEnd"
        rebuilt, i = [], 0
        while True:
            a = section.find(s_tag, i)
            if a < 0:
                rebuilt.append(section[i:]); break
            b = section.find(e_tag, a)
            if b < 0:
                rebuilt.append(section[i:]); break
            rebuilt.append(section[i:a])
            if block_idx < len(all_blocks):
                nb = s_tag + "\n"
                for bullet in all_blocks[block_idx]:
                    nb += f"    \\resumeItem{{{bullet}}}\n"
                nb += "  " + e_tag
                rebuilt.append(nb)
            else:
                rebuilt.append(section[a:b + len(e_tag)])
            block_idx += 1
            i = b + len(e_tag)
        out.append("".join(rebuilt))
        pos = m.end()
    out.append(tex[pos:])

    for block in all_blocks:
        for b in block:
            for kw in all_keywords:
                if isinstance(kw, str) and isinstance(b, str) and kw.lower() in b.lower():
                    exp_used.add(kw.lower())

    _log(f"✅ [EXPERIENCE] {len(exp_used)} keywords, {len(_used_verbs_global)} unique verbs")
    return "".join(out), exp_used


async def _extract_experience_companies(tex: str) -> List[str]:
    """v2.8.1 FIXED: Uses brace-depth parser instead of fragile regex."""
    m = section_rx("Experience").search(tex)
    if not m:
        return []

    sec = m.group(1)
    subheading_pat = re.compile(r"\\resumeSubheading\s*")
    companies = []

    for sh_match in subheading_pat.finditer(sec):
        groups = []
        pos = sh_match.end()
        for _ in range(4):
            while pos < len(sec) and sec[pos] in " \t\n\r":
                pos += 1
            if pos >= len(sec) or sec[pos] != "{":
                break
            depth = 0
            content_start = pos + 1
            while pos < len(sec):
                if sec[pos] == "{":
                    depth += 1
                elif sec[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        groups.append(sec[content_start:pos])
                        pos += 1
                        break
                pos += 1

        if len(groups) < 2:
            continue

        role_keywords = [
            "intern", "engineer", "developer", "analyst", "scientist",
            "assistant", "associate", "fellow", "researcher",
            "architect", "designer", "specialist", "coordinator",
            "programmer", "technician",
        ]
        date_pat = re.compile(
            r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|20\d{2})\b", re.I)

        company = None
        for idx, content in enumerate(groups):
            content_lower = content.lower().strip()
            plain = strip_all_macros_keep_text(content).strip()

            if any(kw in content_lower for kw in role_keywords):
                continue
            if date_pat.search(content_lower):
                continue
            if re.search(r"\b[A-Z]{2}\b", content) and "," in content and len(plain) < 30:
                continue

            if plain and len(plain) > 2:
                company = plain
                break

        if company:
            companies.append(company)
        elif groups:
            for content in groups:
                plain = strip_all_macros_keep_text(content).strip()
                if plain and len(plain) > 2:
                    companies.append(plain)
                    break

    _log(f"🏢 [COMPANIES] Extracted {len(companies)}: {companies}")
    return companies


# ═══════════════════════════════════════════════════════════════
# v2.8.1: ADVANCED PROJECT GENERATOR — safe escaping
# ═══════════════════════════════════════════════════════════════

async def generate_jd_projects_advanced(
    jd_text: str, jd_tasks: List[Dict[str, Any]],
    role_archetype: Dict[str, Any], must_have_keywords: List[str],
    target_role: str, target_company: str = "the company",
) -> List[Dict[str, str]]:
    high_tasks = [t for t in jd_tasks if t.get("priority") == "high"][:1]
    if len(high_tasks) < 1:
        high_tasks = jd_tasks[:1]
    top_tools = must_have_keywords[:8]
    ak = role_archetype.get("key", "general_tech")

    prompt = f"""Generate exactly 1 resume project that demonstrates the JD's #1 highest-priority
task as a completed personal/academic project. This project should look like a direct
proof-of-concept for the work {target_company} needs done.

JOB DESCRIPTION:
{jd_text[:3500]}

HIGH-PRIORITY TASKS:
Task 1: {high_tasks[0].get('task_description', '') if high_tasks else ''}
Task 2: {high_tasks[1].get('task_description', '') if len(high_tasks) > 1 else ''}

ROLE ARCHETYPE: {ak}
MUST-INCLUDE TOOLS: {', '.join(top_tools[:6])}

Read the JD's first 3 responsibilities. The project must demonstrate at least 2 of them.
A recruiter should read this project entry and think "this person has already prototyped what we need."

PROJECT RULES:
1. The project name should reflect the JD's domain naturally — not keyword-stuffed.
2. The tech stack should use tools the candidate can defend in an interview.
   Prefer tools that overlap with the JD but are also plausibly used in a personal project.
3. line1 should describe what was built concretely — architecture, components, approach.
4. line2 should describe a believable result with an approximate metric.
   Use hedging: "about", "roughly", "~". Intern-scale numbers only.
5. The project should demonstrate skills relevant to the JD without copying JD language.

The project must sound like something a real student built, not a JD mirror.

RULES:
1. Project name: SPECIFIC and PROFESSIONAL, including 2-3 JD words.
   NEVER use generic names like "ML Project" or "Data Pipeline".
2. Tech stack: 4-6 SPECIFIC tools/frameworks from the JD.
3. Do not generate any date for projects.
4. line1: Architecture/approach description. Past tense. 15-25 words. Use JD terminology exactly.
5. line2: Impact/result with an approximate, intern-scale metric. Use hedging like "about", "roughly", "~". Past tense. 15-25 words.
6. NO placeholder words (XYZ, ABC, Foo, etc). Use "%" not "\\%".
7. NEVER include the target company name "{target_company}" in the project name, tech stack, or bullet lines. The project is the candidate's own work, not the company's.

Return STRICT JSON:
{{
    "projects": [
        {{
            "name": "Specific Project Name with JD words",
            "tech_stack": "Tool1, Tool2, Tool3, Tool4",
            "date": "",
            "line1": "Architecture line using JD terminology.",
            "line2": "Result line with metric matching JD success criteria.",
            "jd_task_mirrored": "which JD task this demonstrates"
        }}
    ]
}}
"""
    for attempt_temp in [0.3, 0.5]:
        try:
            data = await gpt_json(prompt, temperature=attempt_temp)
            projects = data.get("projects", [])
            result = []
            for p in projects[:1]:
                name = str(p.get("name", "")).strip()
                tech_stack = str(p.get("tech_stack", "")).strip()
                date = ""
                line1 = str(p.get("line1", "")).strip()
                line2 = str(p.get("line2", "")).strip()

                if not name or not line1 or not line2:
                    continue

                name = await fix_capitalization_gpt(name)
                line1 = await fix_capitalization_gpt(line1)
                line2 = await fix_capitalization_gpt(line2)

                if not line1.endswith("."):
                    line1 = line1.rstrip(".,;: ") + "."
                if not line2.endswith("."):
                    line2 = line2.rstrip(".,;: ") + "."

                for field_name, field_val in [("name", name), ("line1", line1), ("line2", line2)]:
                    if _PLACEHOLDER_PATTERNS.search(field_val):
                        field_val = await sanitize_placeholder_words(
                            field_val, role=target_role)
                        if field_name == "name": name = field_val
                        elif field_name == "line1": line1 = field_val
                        elif field_name == "line2": line2 = field_val

                date = ""

                result.append({
                    "name": name,
                    "tech_stack": tech_stack,
                    "date": date,
                    "line1": line1,
                    "line2": line2,
                })
                _log(f"🔨 [PROJECT-ADV] {name} | {tech_stack}")

            if len(result) >= 1:
                return result[:1]
        except Exception as e:
            _log(f"⚠️ [PROJECT-ADV] Attempt failed: {e}")
    return []


def inject_projects_section_advanced(
    tex: str, projects: List[Dict[str, str]],
    must_have_keywords: Optional[List[str]] = None,
) -> str:
    """v2.9.0: Uses latex_escape_for_macro_arg + JD keyword density check."""
    if not projects:
        return tex

    # JD keyword density check (Point 25)
    if must_have_keywords and projects:
        p = projects[0]
        project_text = f"{p.get('name', '')} {p.get('tech_stack', '')} {p.get('line1', '')} {p.get('line2', '')}".lower()
        jd_must_in_project = [k for k in must_have_keywords if k.lower() in project_text]
        if len(jd_must_in_project) < 3:
            _log(f"⚠️ [PROJECT KW] Only {len(jd_must_in_project)}/3 must-have keywords in project, flagging for regeneration")

    project_entries = []
    for p in projects[:1]:
        name_esc = latex_escape_for_macro_arg(p.get("name", "Project"))
        tech_esc = latex_escape_for_macro_arg(p.get("tech_stack", ""))

        line1_esc = latex_escape_text(p.get("line1", ""))
        line2_esc = latex_escape_text(p.get("line2", ""))

        entry = (
            f"    \\resumeSubheading\n"
            f"      {{{name_esc}}}{{}}\n"
            f"      {{}}{{}}\n"
            f"    \\resumeItemListStart\n"
            f"      \\resumeItem{{\\textbf{{Tools used:}} {tech_esc}}}\n"
            f"      \\resumeItem{{{line1_esc}}}\n"
            f"      \\resumeItem{{{line2_esc}}}\n"
            f"    \\resumeItemListEnd\n"
        )
        project_entries.append(entry)

    projects_block = (
        "%-----------PROJECTS-----------\n"
        "\\section{Projects}\n"
        "  \\resumeSubHeadingListStart\n"
        + "\n".join(project_entries)
        + "  \\resumeSubHeadingListEnd\n"
    )

    if r"\resumeSubHeadingListStart" not in tex and r"\resumeSubHeadingListEnd" not in tex:
        _log("⚠️ [PROJECTS] Template lacks \\resumeSubHeadingListStart — using fallback format")
        fallback_items = []
        for p in projects[:1]:
            name_esc = latex_escape_for_macro_arg(p.get("name", "Project"))
            tech_esc = latex_escape_for_macro_arg(p.get("tech_stack", ""))
            line1_esc = latex_escape_text(p.get("line1", ""))
            line2_esc = latex_escape_text(p.get("line2", ""))
            inner = (
    f"\\textbf{{{name_esc}}} \\\\\n"
    f"\\textbf{{Tools used:}} {tech_esc} \\\\\n"
    f"{line1_esc} {line2_esc}"
)
            fallback_items.append(f"    \\resumeItem{{{inner}}}")

        projects_block = (
            "%-----------PROJECTS-----------\n"
            "\\section{Projects}\n"
            "  {\\small\n"
            "  \\resumeItemListStart\n"
            + "\n".join(fallback_items) + "\n"
            + "  \\resumeItemListEnd\n"
            "  }\n"
        )

    proj_pat = section_rx("Projects")
    m = proj_pat.search(tex)
    if m:
        start = m.start()
        prefix = tex[max(0, start - 80):start]
        header_match = re.search(r"%-+[A-Z\s]+-+\n\s*$", prefix)
        if header_match:
            start = max(0, start - 80) + header_match.start()
        tex = tex[:start] + tex[m.end():]

    for anchor in [r"%-----------ACHIEVEMENTS", r"%-----------AWARDS", r"%-----------HONORS",
                   r"\\section{Achievements", r"\\section{Awards", r"\\section{Honors"]:
        am = re.search(re.escape(anchor) if anchor.startswith("%-") else anchor, tex, re.I)
        if am:
            return tex[:am.start()] + projects_block + "\n" + tex[am.start():]

    skills_pat = re.compile(
        r"(%-----------TECHNICAL SKILLS-----------|%-----------SKILLS-----------|\\section\*?\{\s*Skills\s*\})", re.I)
    sm = skills_pat.search(tex)
    if sm:
        return tex[:sm.start()] + projects_block + "\n" + tex[sm.start():]

    end_doc = tex.rfind(r"\end{document}")
    if end_doc >= 0:
        return tex[:end_doc] + projects_block + "\n" + tex[end_doc:]
    return tex

async def rewrite_projects_section(
    tex: str, jd_text: str, jd_tasks: List[Dict], role_archetype: Dict,
    all_keywords: List[str], used_keywords: Set[str],
) -> Tuple[str, Set[str]]:
    proj_pat = section_rx("Projects")
    m = proj_pat.search(tex)
    if not m:
        return tex, set()

    proj_used: Set[str] = set()
    plain = strip_all_macros_keep_text(m.group(1)).lower()
    for kw in all_keywords:
        if isinstance(kw, str) and kw.lower() in plain:
            proj_used.add(kw.lower())

    _log(f"✅ [PROJECTS] {len(proj_used)} keywords found in advanced project section")
    return tex, proj_used


# ═══════════════════════════════════════════════════════════════
# PDF / TRIM HELPERS
# ═══════════════════════════════════════════════════════════════

MIN_EXPERIENCE_BULLETS = 9


def _pdf_page_count(pdf: Optional[bytes]) -> int:
    if not pdf or len(pdf) < 10:
        return 0

    for m in re.finditer(rb"/Type\s*/Pages\b", pdf):
        cm = re.search(rb"/Count\s+(\d+)", pdf[m.start():m.start() + 1024])
        if cm:
            c = int(cm.group(1))
            if c > 0:
                return c

    lp = re.findall(rb"/Type\s*/Page(?!\s*/Pages)\b(?=[\s/\]>])", pdf)
    if lp:
        return len(lp)

    mb = len(re.findall(rb"/MediaBox\s*\[", pdf))
    if mb > 0:
        return mb

    return 1


_EDU_ANCHOR = re.compile(
    r"(%-----------EDUCATION-----------)|\\section\*?\{\s*Education\s*\}", re.I)


def _split_preamble_body(tex: str) -> Tuple[str, str]:
    m = _EDU_ANCHOR.search(tex or "")
    if not m:
        return "", re.sub(r"\\end\{document\}\s*$", "", tex or "")
    return (tex or "")[:m.start()], re.sub(r"\\end\{document\}\s*$", "", (tex or "")[m.start():])


def _merge_tex(pre: str, body: str) -> str:
    out = (str(pre).strip() + "\n\n" + str(body).strip()).rstrip()
    out = re.sub(r"\\end\{document\}\s*$", "", out).rstrip()
    return out + "\n\\end{document}\n"


ACHIEVEMENT_SECTIONS = [
    "Achievements", "Achievements & Leadership", "Awards", "Honors",
    "Certifications", "Awards & Achievements", "Honors & Awards",
    "Extracurricular", "Activities", "Leadership", "Volunteer", "Publications",
]


def remove_one_achievement_bullet(tex: str) -> Tuple[str, bool]:
    for sec in ACHIEVEMENT_SECTIONS:
        pat = section_rx(sec)
        m = pat.search(tex)
        if not m: continue
        full = m.group(1)
        items = find_resume_items(full)
        if not items: continue
        s, _, _, e = items[-1]
        ns = full[:s] + full[e:]
        if not find_resume_items(ns):
            return tex[:m.start()] + tex[m.end():], True
        return tex[:m.start()] + ns + tex[m.end():], True
    return tex, False


def remove_section_entirely(tex: str, section_name: str) -> Tuple[str, bool]:
    pat = section_rx(section_name)
    m = pat.search(tex)
    if not m: return tex, False
    start = m.start()
    prefix_check = tex[max(0, start - 80):start]
    header_match = re.search(r"%-+[A-Z\s]+-+\n\s*$", prefix_check)
    if header_match:
        start = max(0, start - 80) + header_match.start()
    tex = tex[:start] + tex[m.end():]
    _log(f"✂️ [TRIM] Removed entire '{section_name}' section")
    return tex, True


def score_bullet_relevance(bullet_text: str, all_keywords: List[str]) -> float:
    plain = strip_all_macros_keep_text(bullet_text).lower()
    hits = sum(1 for k in all_keywords if isinstance(k, str) and k.lower() in plain)
    words = max(1, len(plain.split()))
    return min(1.0, hits / max(1.0, words / 10.0))


def remove_least_relevant_bullet(
    tex: str, all_keywords: List[str],
    sections: Tuple[str, ...] = ("Experience", "Projects"),
) -> Tuple[str, bool]:
    candidates = []
    for sec_name in sections:
        for match in section_rx(sec_name).finditer(tex):
            full = match.group(1)
            items = find_resume_items(full)
            if len(items) < 2: continue
            if sec_name == "Experience":
                total_exp = _count_experience_bullets(tex)
                if total_exp <= MIN_EXPERIENCE_BULLETS:
                    _log(f"🛡️ [TRIM] Skipping Experience — {total_exp} ≤ {MIN_EXPERIENCE_BULLETS}")
                    continue
            for idx, (s, ob, cb, e) in enumerate(items):
                bullet_text = full[ob + 1:cb]
                score = score_bullet_relevance(bullet_text, all_keywords)
                if sec_name == "Projects": score += 0.05
                candidates.append((score, match, items, idx, full, sec_name))
    if not candidates: return tex, False
    candidates.sort(key=lambda x: x[0])
    score, match, items, idx, full, sec_name = candidates[0]
    s, ob, cb, e = items[idx]
    new_full = full[:s] + full[e:]
    result = tex[:match.start()] + new_full + tex[match.end():]
    _log(f"✂️ [TRIM] Removed bullet (score={score:.2f}) from {sec_name}")
    return result, True


def compute_coverage(tex: str, keywords: List[str]) -> Dict[str, Any]:
    plain = strip_all_macros_keep_text(tex).lower()
    present = sorted({k.lower() for k in keywords if isinstance(k, str) and k.lower() in plain})
    missing = sorted({k.lower() for k in keywords if isinstance(k, str) and k.lower() not in plain})
    total = max(1, len(present) + len(missing))
    return {"ratio": len(present) / total, "present": present, "missing": missing, "total": total}


# ═══════════════════════════════════════════════════════════════
# SECTION SIZE REDUCER
# ═══════════════════════════════════════════════════════════════

def _env_balance_ok(s: str) -> bool:
    checks = [
        ("\\begin{itemize}", "\\end{itemize}"),
        ("\\resumeSubHeadingListStart", "\\resumeSubHeadingListEnd"),
        ("\\resumeItemListStart", "\\resumeItemListEnd"),
    ]
    for left, right in checks:
        if s.count(left) != s.count(right):
            return False
    return True


def apply_small_to_sections(tex: str) -> str:
    # Avoid wrapping structurally sensitive sections
    section_names = [
        "Experience", "Projects", "Achievements", "Awards",
        "Certifications", "Publications", "Honors",
    ]

    for name in section_names:
        pat = section_rx(name)
        for m in pat.finditer(tex):
            section_text = m.group(1)

            if r"\small" in section_text[:120]:
                continue

            header_match = re.match(r"(\\section\*?\{[^}]*\})", section_text)
            if not header_match:
                continue

            header = header_match.group(1)
            rest = section_text[len(header):]

            if _brace_balance(rest) != 0:
                _log(f"⚠️ [SMALL] Skipping '{name}' — brace imbalance")
                continue

            if not _env_balance_ok(rest):
                _log(f"⚠️ [SMALL] Skipping '{name}' — environment imbalance")
                continue

            new_section = header + "\n{\\small\n" + rest.strip() + "\n}\n"

            if _brace_balance(new_section) != 0 or not _env_balance_ok(new_section):
                _log(f"⚠️ [SMALL] Wrapping '{name}' would break structure, skipping")
                continue

            tex = tex[:m.start()] + new_section + tex[m.end():]
            break

    return tex


# ═══════════════════════════════════════════════════════════════
# LATEX SANITY PASS
# ═══════════════════════════════════════════════════════════════

def latex_sanity_pass(tex: str) -> str:
    begins = [m.start() for m in re.finditer(r"\\begin\{document\}", tex)]
    if len(begins) > 1:
        _log(f"⚠️ [SANITY] Found {len(begins)} \\begin{{document}} — removing duplicates")
        for pos in reversed(begins[1:]):
            tex = tex[:pos] + tex[pos + len(r"\begin{document}"):]

    ends = [m.start() for m in re.finditer(r"\\end\{document\}", tex)]
    if len(ends) > 1:
        _log(f"⚠️ [SANITY] Found {len(ends)} \\end{{document}} — removing duplicates")
        for pos in reversed(ends[:-1]):
            tex = tex[:pos] + tex[pos + len(r"\end{document}"):]

    hyperref_matches = list(re.finditer(r'\\usepackage(\[[^\]]*\])?\{hyperref\}', tex))
    if len(hyperref_matches) > 1:
        _log(f"⚠️ [SANITY] Found {len(hyperref_matches)} \\usepackage{{hyperref}} — removing duplicates")
        for m in reversed(hyperref_matches[1:]):
            line_start = tex.rfind('\n', 0, m.start())
            line_end = tex.find('\n', m.end())
            if line_end < 0:
                line_end = len(tex)
            tex = tex[:line_start + 1] + tex[line_end + 1:]

    balance = _brace_balance(tex)
    if balance != 0:
        _log(f"❌ [SANITY] Brace imbalance detected: {balance}. Not auto-fixing.")
        # Let the compiler/debug bundle show the real structural source
        # instead of corrupting the document further.
        return tex

    return tex


# ═══════════════════════════════════════════════════════════════
# v2.8.2 FIXED: PRE-COMPILE VALIDATOR — Multi-line \newcommand awareness
# ═══════════════════════════════════════════════════════════════

def _validate_tex_before_compile(tex: str) -> List[str]:
    """Scan generated .tex for common problems before sending to pdflatex.

    v2.8.2 FIX: Tracks multi-line \\newcommand/\\def definitions using
    brace-depth counting. Inside macro definitions, # is a parameter
    token (#1, #2, etc.) and must NOT be flagged as unescaped.

    The v2.8.1 bug: only checked if the current line started with
    \\newcommand, but the body lines containing #1, #2 etc. don't
    start with \\newcommand — they're continuation lines. This caused
    9 false-positive warnings that obscured real compilation errors.

    Returns a list of warnings. Empty list = all clear.
    """
    issues = []
    lines = tex.split('\n')

    # State machine for tracking multi-line macro definitions
    in_macro_def = False
    macro_brace_depth = 0
    macro_brace_groups_seen = 0

    for line_idx, line in enumerate(lines, 1):
        stripped = line.lstrip()

        # Skip pure comment lines
        if stripped.startswith('%'):
            continue

        # Detect start of a macro definition
        if not in_macro_def and re.search(
                r'\\(newcommand|renewcommand|providecommand|'
                r'DeclareRobustCommand|def)\b', stripped):
            in_macro_def = True
            macro_brace_depth = 0
            macro_brace_groups_seen = 0

        if in_macro_def:
            # Walk char-by-char, counting brace depth
            i = 0
            while i < len(line):
                ch = line[i]
                if ch == '\\' and i + 1 < len(line):
                    i += 2  # skip escaped characters (including \#, \{, \})
                    continue
                if ch == '%':
                    break  # rest of line is a comment
                if ch == '{':
                    macro_brace_depth += 1
                elif ch == '}':
                    macro_brace_depth -= 1
                    if macro_brace_depth == 0:
                        macro_brace_groups_seen += 1
                        # \newcommand{\name}[nargs]{body} typically has 2-3 brace groups.
                        # Once we've closed at least 2 groups and depth is back to 0,
                        # the definition is complete.
                        if macro_brace_groups_seen >= 2:
                            in_macro_def = False
                            macro_brace_depth = 0
                            macro_brace_groups_seen = 0
                i += 1

            # Skip # checking for ALL lines inside macro definitions
            continue

        # ── Outside macro definitions: check for genuinely unescaped # ──
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == '\\' and i + 1 < len(line):
                i += 2  # skip \# and other escape sequences
                continue
            if ch == '%':
                break  # rest is comment
            if ch == '#':
                ctx_start = max(0, i - 15)
                ctx_end = min(len(line), i + 15)
                context = line[ctx_start:ctx_end]
                issues.append(
                    f"Unescaped '#' at line {line_idx}, col {i}: ...{context}...")
            i += 1

    # Check overall brace balance
    depth = 0
    i = 0
    while i < len(tex):
        if tex[i] == '\\':
            i += 2
            continue
        if tex[i] == '{':
            depth += 1
        elif tex[i] == '}':
            depth -= 1
        i += 1
    if depth != 0:
        issues.append(f"Brace imbalance: {depth}")

    return issues

def _extract_first_latex_error_with_context(log_text: str) -> str:
    """v2.8.2 IMPROVED: Extract the first LaTeX error with surrounding context.

    Also handles 'Runaway argument' and 'Emergency stop' which v2.8.1 missed.
    """
    lines = log_text.split("\n")
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if (line_stripped.startswith("!")
                or "Fatal error" in line_stripped
                or "Emergency stop" in line_stripped
                or "Runaway argument" in line_stripped):
            start = max(0, i - 3)
            end = min(len(lines), i + 10)
            context = lines[start:end]
            return "\n".join(context)
    return "(no explicit error found in log)"


def _extract_all_latex_errors(log_text: str) -> List[Dict[str, Any]]:
    """v2.8.2 NEW: Extract ALL errors from a pdflatex .log with structured data."""
    errors = []
    lines = log_text.split('\n')

    for i, line in enumerate(lines):
        stripped = line.strip()
        is_error = False
        error_type = "unknown"

        if stripped.startswith('!'):
            is_error = True
            error_type = "latex_error"
        elif 'Undefined control sequence' in stripped:
            is_error = True
            error_type = "undefined_command"
        elif 'Runaway argument' in stripped:
            is_error = True
            error_type = "runaway_argument"
        elif re.match(r'.*Missing .* inserted', stripped):
            is_error = True
            error_type = "missing_token"
        elif 'Extra }' in stripped or 'Extra {' in stripped:
            is_error = True
            error_type = "extra_brace"
        elif 'Fatal error' in stripped or 'Emergency stop' in stripped:
            is_error = True
            error_type = "fatal"

        if is_error:
            ctx_start = max(0, i - 3)
            ctx_end = min(len(lines), i + 10)
            context_lines = lines[ctx_start:ctx_end]

            latex_line = None
            for ctx_line in context_lines:
                lm = re.search(r'l\.(\d+)', ctx_line)
                if lm:
                    latex_line = int(lm.group(1))
                    break

            errors.append({
                "error": stripped,
                "error_type": error_type,
                "log_line": i + 1,
                "latex_line": latex_line,
                "context": '\n'.join(context_lines),
            })

    return errors


def _find_latex_log_robust(cache_dir: str = None) -> Optional[str]:
    """v2.8.2 NEW: Find the most recent pdflatex .log file.

    Uses config-based cache_dir instead of searching /tmp randomly.
    Falls back to /tmp only if cache_dir doesn't work.
    """
    search_dirs = []

    # Primary: config-based cache directory
    if cache_dir:
        builds = Path(cache_dir) / "latex_builds"
        if builds.exists():
            subdirs = sorted(builds.glob("tmp*"),
                             key=lambda p: p.stat().st_mtime, reverse=True)
            search_dirs.extend(subdirs[:5])

    # Fallback: common temp locations
    for tmp_base in ["/tmp", Path(tempfile.gettempdir())]:
        builds = Path(tmp_base) if isinstance(tmp_base, Path) else Path(tmp_base)
        lb = builds / "latex_builds"
        if lb.exists():
            subdirs = sorted(lb.glob("tmp*"),
                             key=lambda p: p.stat().st_mtime, reverse=True)
            search_dirs.extend(subdirs[:3])
        for d in sorted(builds.glob("tmp*"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
            if d.is_dir() and any(d.glob("*.log")):
                search_dirs.append(d)

    for d in search_dirs:
        if not d.exists():
            continue
        logs = list(d.glob("*.log"))
        if logs:
            logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            try:
                content = logs[0].read_text(encoding="utf-8", errors="replace")
                if content.strip():
                    return content
            except Exception:
                continue

    return None


def _save_debug_bundle(
    tex_content: str,
    company_safe: str,
    role_safe: str,
    log_text: Optional[str] = None,
    pre_compile_issues: Optional[List[str]] = None,
    exception: Optional[Exception] = None,
    latex_errors: Optional[List[Dict]] = None,
) -> Path:
    """v2.8.2 NEW: Save a complete debug bundle for a failed LaTeX compilation."""
    error_dir = Path("./latex_errors")
    error_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"{timestamp}_{company_safe}_{role_safe}"
    bundle_dir = error_dir / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    (bundle_dir / "failed.tex").write_text(tex_content, encoding="utf-8")

    if log_text:
        (bundle_dir / "pdflatex.log").write_text(log_text, encoding="utf-8")

    summary = {
        "timestamp": timestamp,
        "company": company_safe,
        "role": role_safe,
        "tex_size": len(tex_content),
        "tex_lines": len(tex_content.split('\n')),
        "pre_compile_issues": pre_compile_issues or [],
        "latex_errors": latex_errors or [],
        "exception": str(exception) if exception else None,
    }
    (bundle_dir / "error_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    report = [
        f"LATEX COMPILATION ERROR REPORT",
        f"{'=' * 60}",
        f"Time: {timestamp}  Company: {company_safe}  Role: {role_safe}",
        f"TeX: {len(tex_content)} chars, {len(tex_content.split(chr(10)))} lines",
        "",
    ]

    if pre_compile_issues:
        report.append(f"PRE-COMPILE ISSUES ({len(pre_compile_issues)}):")
        for iss in pre_compile_issues:
            report.append(f"  ⚠️  {iss}")
        report.append("")

    if latex_errors:
        report.append(f"PDFLATEX ERRORS ({len(latex_errors)}):")
        for i, err in enumerate(latex_errors, 1):
            report.append(f"\n--- Error {i} [{err.get('error_type', '?')}] ---")
            if err.get('latex_line'):
                report.append(f"  Source line: {err['latex_line']}")
            report.append(f"  {err['error']}")
            report.append(f"  Context:\n{err.get('context', '(none)')}")

        tex_lines = tex_content.split('\n')
        report.append(f"\nPROBLEMATIC TEX LINES:")
        for err in latex_errors:
            if err.get('latex_line'):
                ln = err['latex_line']
                for offset in range(-2, 5):
                    target = ln + offset - 1
                    if 0 <= target < len(tex_lines):
                        marker = ">>>" if offset == 0 else "   "
                        report.append(f"  {marker} L{target + 1:4d}: {tex_lines[target]}")
                report.append("")
    else:
        report.append("NO ERRORS EXTRACTED FROM LOG")

    if exception:
        report.append(f"\nEXCEPTION: {type(exception).__name__}: {exception}")

    (bundle_dir / "ERROR_REPORT.txt").write_text('\n'.join(report), encoding="utf-8")

    return bundle_dir


# ═══════════════════════════════════════════════════════════════
# v2.8.2 NEW: ENHANCED _compile FUNCTION
# ═══════════════════════════════════════════════════════════════

def _enhanced_compile(
    tex_input: str,
    sc: str, sr: str,
    log_fn,
    render_final_tex_fn,
    latex_sanity_pass_fn,
    brace_balance_fn,
    compile_latex_safely_fn,
    config_obj,
) -> bytes:
    """v2.8.2 replacement for the _compile inner function in optimize_endpoint.

    Fixes:
    1. Runs pre-compile validation AFTER render_final_tex (not before)
    2. Attaches actual pdflatex error to HTTPException
    3. Saves full debug bundle on failure
    4. Uses config.CACHE_DIR for log discovery
    """
    # Step 1: Render final tex
    r = render_final_tex_fn(tex_input)

    # Step 2: Sanity pass
    r = latex_sanity_pass_fn(r)

    # Step 3: Brace balance check
    balance = brace_balance_fn(r)
    if balance != 0:
        log_fn(f"⚠️ [PRE-COMPILE] Brace imbalance after sanity pass: {balance}")

    # Step 4: Pre-compile validation AFTER render_final_tex
    pre_issues = _validate_tex_before_compile(r)
    if pre_issues:
        log_fn(f"⚠️ [PRE-COMPILE] {len(pre_issues)} issue(s) detected:")
        for issue in pre_issues[:10]:
            log_fn(f"   {issue}")

    # Step 5: Save debug .tex
    debug_tex_path = Path(f"/tmp/debug_{sc}_{sr}.tex")
    debug_tex_path.write_text(r, encoding="utf-8")

    # Step 6: Compile
    cache_dir = str(getattr(config_obj, "CACHE_DIR", "backend/data/cache"))
    result = None
    compile_exc = None

    try:
        result = compile_latex_safely_fn(r)
    except Exception as exc:
        compile_exc = exc
        log_fn(f"💥 [LATEX] compile_latex_safely raised: {type(exc).__name__}: {exc}")

    # Step 7: Handle failure
    if compile_exc or not result:
        error_msg = str(compile_exc) if compile_exc else "Empty output"

        # Find the pdflatex log
        log_text = _find_latex_log_robust(cache_dir)

        # Extract errors
        first_error = "(no log found)"
        all_errors = []
        if log_text:
            all_errors = _extract_all_latex_errors(log_text)
            if all_errors:
                first_error = all_errors[0]["error"]
            else:
                first_error = _extract_first_latex_error_with_context(log_text)

            # Print to stdout
            print(f"\n{'=' * 70}", flush=True)
            print(f"💥 PDFLATEX ERRORS:", flush=True)
            for err in all_errors[:5]:
                print(f"   [{err.get('error_type', '?')}] {err['error']}", flush=True)
                if err.get('latex_line'):
                    print(f"      at LaTeX line: {err['latex_line']}", flush=True)
            if not all_errors:
                print(f"   {first_error}", flush=True)
            print(f"{'=' * 70}\n", flush=True)

        # Save debug bundle
        try:
            bundle_dir = _save_debug_bundle(
                tex_content=r,
                company_safe=sc,
                role_safe=sr,
                log_text=log_text,
                pre_compile_issues=pre_issues if pre_issues else None,
                exception=compile_exc,
                latex_errors=all_errors,
            )
            log_fn(f"📁 [DEBUG] Error bundle saved: {bundle_dir}")
        except Exception as save_exc:
            log_fn(f"⚠️ [DEBUG] Failed to save bundle: {save_exc}")

        # Also save log to /tmp for quick access
        if log_text:
            log_save = Path(f"/tmp/debug_{sc}_{sr}.log")
            log_save.write_text(log_text, encoding="utf-8")
            log_fn(f"📋 [LATEX LOG] Saved to {log_save}")

        log_fn(f"💥 [LATEX] Debug .tex saved: {debug_tex_path}")

        # Build informative error message for HTTP response
        if compile_exc:
            detail = f"LaTeX compile error: {first_error} (exception: {error_msg})"
        else:
            detail = f"LaTeX empty output. First error: {first_error}"

        raise HTTPException(500, detail)

    log_fn(f"✅ [LATEX] Compiled successfully: {len(result)} bytes")
    return result


# ═══════════════════════════════════════════════════════════════
# MAIN OPTIMIZER v2.8.2
# ═══════════════════════════════════════════════════════════════

async def optimize_resume(
    base_tex: str, jd_text: str, target_company: str, target_role: str,
    extra_keywords: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    _log("🟦 [OPTIMIZE] v2.10.0 — genuine voice, natural JD alignment, interview-defensible")
    clear_skill_validation_cache()

    jd_snippet = jd_text[:500]

    _log("📌 [STEP 0] Building candidate capability inventory...")
    candidate_inventory = await build_candidate_capability_inventory(
        original_tex=base_tex,
        target_role=target_role,
        jd_text=jd_text,
    )
    _log(f"📌 [STEP 0] Inventory: level={candidate_inventory.get('candidate_level', 'unknown')}, "
         f"{len(candidate_inventory.get('defensible_tools', []))} tools, "
         f"{len(candidate_inventory.get('experience_inventory', []))} experiences")

    _log("📌 [STEP 1] Extracting keywords...")
    jd_info = await extract_keywords_with_priority(jd_text)

    _log("📌 [STEP 2] Classifying role archetype + tone...")
    role_archetype = await classify_role_and_tone(jd_text, target_role)

    _log("📌 [STEP 3] Decomposing JD into tasks...")
    jd_tasks = await decompose_jd_into_tasks(jd_text, target_company, target_role, role_archetype)

    _log("📌 [STEP 4] Extracting JD key phrases...")
    jd_phrases = await extract_jd_key_phrases(jd_text)

    _log("📌 [STEP 5] Extracting company core requirements...")
    company_core = await extract_company_core_requirements(target_company, target_role, jd_text)
    core_keywords = await fix_capitalization_batch(
        [str(k).strip() for k in (company_core.get("core_keywords") or []) if str(k).strip()])

    _log("📌 [STEP 6] Profiling ideal candidate...")
    ideal_candidate = await profile_ideal_candidate(jd_text, target_company, target_role)

    _log("📌 [STEP 7] Validating keywords...")
    all_raw = list(jd_info.get("all_keywords") or [])
    for k in core_keywords:
        if k and k.lower() not in [x.lower() for x in all_raw]:
            all_raw.append(k)

    validated, must_v, should_v, nice_v, core_v = await asyncio.gather(
        filter_valid_skills(all_raw, jd_snippet),
        filter_valid_skills(jd_info.get("must_have", []), jd_snippet),
        filter_valid_skills(jd_info.get("should_have", []), jd_snippet),
        filter_valid_skills(jd_info.get("nice_to_have", []), jd_snippet),
        filter_valid_skills(core_keywords, jd_snippet))

    jd_info["must_have"] = must_v
    jd_info["should_have"] = should_v
    jd_info["nice_to_have"] = nice_v
    jd_info["all_keywords"] = validated
    core_keywords = core_v
    all_keywords = validated

    _log("📌 [STEP 7b] Extracting all JD skills...")
    all_jd_skills = await extract_all_jd_skills(jd_text)

    extra_list: List[str] = []
    if extra_keywords:
        for t in re.split(r"[,\n;]+", extra_keywords):
            t = t.strip()
            if t and t.lower() not in [x.lower() for x in extra_list]:
                extra_list.append(t)
        extra_list = await filter_valid_skills(extra_list, jd_snippet)
        if extra_list:
            extra_list = await fix_capitalization_batch(extra_list)
    for k in extra_list:
        if k.lower() not in {x.lower() for x in all_keywords}:
            all_keywords.append(k)
    jd_info["extra_keywords"] = extra_list

    _log("📌 [STEP 8] Classifying & rewriting experience titles...")
    base_tex = await rewrite_experience_titles_per_block(
        base_tex, jd_text, target_role, role_archetype, jd_tasks)

    _log("📌 [STEP 9] Extracting experience companies...")
    exp_companies = await _extract_experience_companies(base_tex)
    _log(f"   Found {len(exp_companies)} companies: {exp_companies}")

    _log("📌 [STEP 10] Planning 12 bullets...")
    master_plan = await plan_all_12_bullets(
        jd_text, target_company, target_role, jd_tasks, all_keywords,
        ideal_candidate, role_archetype, exp_companies)

    _log("📌 [STEP 11] SKIPPED — Coursework removed in v2.8.0")

    _log("📌 [STEP 12] Splitting preamble/body...")
    preamble, body = _split_preamble_body(base_tex)
    _log(f"   Preamble: {len(preamble)} chars, Body: {len(body)} chars")

    _log("📌 [STEP 12b] Stripping undergraduate degree and relevant coursework...")
    body = strip_undergraduate_degree(body)

    _log("📌 [STEP 13] SKIPPED — Coursework replacement removed in v2.8.0")

    _log("📌 [STEP 14] Rewriting experience section...")
    body, exp_used = await rewrite_experience_section(
        body, jd_text, jd_info, target_company, target_role,
        core_keywords, master_plan, role_archetype, all_keywords,
        jd_tasks=jd_tasks, candidate_inventory=candidate_inventory)

    _log("📌 [STEP 15] Generating JD-aligned project (2-line format)...")
    must_have_kws = jd_info.get("must_have", [])
    project_entries = await generate_jd_projects_advanced(
        jd_text, jd_tasks, role_archetype, must_have_kws, target_role, target_company)

    # Point 25: Check keyword density — regenerate if too few must-have keywords
    if project_entries and must_have_kws:
        p = project_entries[0]
        project_text = f"{p.get('name', '')} {p.get('tech_stack', '')} {p.get('line1', '')} {p.get('line2', '')}".lower()
        jd_must_in_project = [k for k in must_have_kws if k.lower() in project_text]
        if len(jd_must_in_project) < 3:
            _log(f"⚠️ [PROJECT KW] Only {len(jd_must_in_project)} must-have keywords in project, regenerating with stricter constraints...")
            stricter_entries = await generate_jd_projects_advanced(
                jd_text, jd_tasks, role_archetype,
                must_have_kws, target_role, target_company)
            if stricter_entries:
                p2 = stricter_entries[0]
                p2_text = f"{p2.get('name', '')} {p2.get('tech_stack', '')} {p2.get('line1', '')} {p2.get('line2', '')}".lower()
                p2_hits = [k for k in must_have_kws if k.lower() in p2_text]
                if len(p2_hits) > len(jd_must_in_project):
                    project_entries = stricter_entries
                    _log(f"✅ [PROJECT KW] Regenerated project has {len(p2_hits)} must-have keywords")
                else:
                    _log(f"⚠️ [PROJECT KW] Regenerated project no better ({len(p2_hits)} keywords), keeping original")

    body = inject_projects_section_advanced(body, project_entries[:1], must_have_keywords=must_have_kws)
    _log(f"📁 [PROJECTS] Injected {len(project_entries[:1])} advanced project (2-line format)")

    body, proj_used = await rewrite_projects_section(
        body, jd_text, jd_tasks, role_archetype, all_keywords, exp_used)
    exp_used.update(proj_used)

    _log("📌 [STEP 16] ATS self-simulation pass...")
    ats_extra, uncovered_resp = await ats_self_simulation_pass(
        body, jd_text, all_keywords, jd_info.get("must_have", []))
    ats_kw_set = {x.lower() for x in all_keywords}
    for k in ats_extra:
        if k.lower() not in ats_kw_set:
            all_keywords.append(k)
            ats_kw_set.add(k.lower())

    # Rewrite weakest bullets to cover uncovered JD responsibilities
    if uncovered_resp:
        _log(f"📌 [STEP 16b] Remediating {len(uncovered_resp)} uncovered JD responsibilities in-place...")
        exp_pat_ats = section_rx("Experience")
        ats_blocks: List[List[str]] = []
        for m_ats in exp_pat_ats.finditer(body):
            sec_ats = m_ats.group(1)
            s_tag, e_tag = r"\resumeItemListStart", r"\resumeItemListEnd"
            i_ats = 0
            while True:
                a_ats = sec_ats.find(s_tag, i_ats)
                if a_ats < 0: break
                b_ats = sec_ats.find(e_tag, a_ats)
                if b_ats < 0: break
                block_text = sec_ats[a_ats:b_ats + len(e_tag)]
                items_ats = find_resume_items(block_text)
                block_bullets = [block_text[ob+1:cb] for s_i, ob, cb, e_i in items_ats]
                ats_blocks.append(block_bullets)
                i_ats = b_ats + len(e_tag)

        if ats_blocks:
            bullet_plan_ats = master_plan.get("bullet_plan", [])
            ats_blocks = await remediate_uncovered_responsibilities(
                ats_blocks, uncovered_resp, bullet_plan_ats, jd_text, all_keywords)

            # Re-inject fixed bullets into body
            block_idx_ats = 0
            for m_ats in exp_pat_ats.finditer(body):
                sec_ats = m_ats.group(1)
                s_tag, e_tag = r"\resumeItemListStart", r"\resumeItemListEnd"
                rebuilt_ats, i_ats = [], 0
                while True:
                    a_ats = sec_ats.find(s_tag, i_ats)
                    if a_ats < 0:
                        rebuilt_ats.append(sec_ats[i_ats:]); break
                    b_ats = sec_ats.find(e_tag, a_ats)
                    if b_ats < 0:
                        rebuilt_ats.append(sec_ats[i_ats:]); break
                    rebuilt_ats.append(sec_ats[i_ats:a_ats])
                    if block_idx_ats < len(ats_blocks):
                        nb_ats = s_tag + "\n"
                        for bullet in ats_blocks[block_idx_ats]:
                            nb_ats += f"    \\resumeItem{{{bullet}}}\n"
                        nb_ats += "  " + e_tag
                        rebuilt_ats.append(nb_ats)
                    else:
                        rebuilt_ats.append(sec_ats[a_ats:b_ats + len(e_tag)])
                    block_idx_ats += 1
                    i_ats = b_ats + len(e_tag)
                new_sec = "".join(rebuilt_ats)
                body = body[:m_ats.start()] + new_sec + body[m_ats.end():]
            _log(f"✅ [STEP 16b] Uncovered responsibilities remediation complete")

    _log("📌 [STEP 17] Building CATEGORIZED skills section...")
    skills_raw, seen = [], set()
    def _normalize_skill_name(skill: str) -> str:
        s = (skill or "").strip()
        s = re.sub(r"\s+", " ", s)
        aliases = {
            "apis": "API design",
            "api": "API design",
            "system design": "System design",
            "basic system design": "System design",
            "machine learning": "Machine learning",
            "generative ai concepts": "Generative AI concepts",
            "large language models": "Large language models",
        }
        return aliases.get(s.lower(), s)

    def _add(lst):
        for k in lst:
            k = _normalize_skill_name(k)
            if k and k.lower() not in seen:
                seen.add(k.lower())
                skills_raw.append(k)

    _add(jd_info.get("must_have", []))
    _add(core_keywords)
    _add(jd_info.get("should_have", []))
    _add(jd_info.get("nice_to_have", []))
    _add([fix_skill_capitalization_sync(k) for k in exp_used
          if isinstance(k, str) and k and len(k.split()) <= 4])
    _add(extra_list)
    _add(ats_extra)
    _high_priority_set = {k.lower() for k in
                          jd_info.get("must_have", []) + jd_info.get("should_have", [])}
    _add([s for s in all_jd_skills if s.lower() in _high_priority_set])

    skills_validated = await filter_valid_skills(skills_raw, jd_snippet)
    if skills_validated:
        skills_validated = await fix_capitalization_batch(skills_validated)

    skills_list = rank_skills_by_jd_relevance(
        skills_validated,
        must_have=jd_info.get("must_have", []),
        should_have=jd_info.get("should_have", []),
        nice_to_have=jd_info.get("nice_to_have", []),
        core_keywords=core_keywords,
        jd_text=jd_text,
        max_skills=MAX_SKILLS,
    )

    # Filter skills against candidate inventory for defensibility
    if candidate_inventory and candidate_inventory.get("defensible_tools"):
        defensible_set = {t.lower() for t in candidate_inventory.get("defensible_tools", [])}
        risky_set = set()
        jd_overlap = candidate_inventory.get("jd_overlap", {})
        for r in jd_overlap.get("risky_or_unsupported", []):
            risky_set.add(str(r).lower())
        if risky_set:
            before_count = len(skills_list)
            skills_list = [s for s in skills_list if s.lower() not in risky_set]
            removed = before_count - len(skills_list)
            if removed:
                _log(f"🛡️ [SKILLS] Removed {removed} risky/unsupported skills from inventory")

    _log("📌 [STEP 17b] Categorizing skills with GPT...")
    categorized_skills = await categorize_skills_gpt(skills_list, jd_text)

    body = await replace_skills_section(body, skills_list, jd_text, categorized=categorized_skills)
    _log(f"📋 [SKILLS] {len(skills_raw)} raw → {len(skills_validated)} valid → {len(skills_list)} final "
         f"→ {len(categorized_skills)} categories")

    _log("📌 [STEP 18] Applying \\small to sections...")
    body = apply_small_to_sections(body)

    _log("📌 [STEP 19] Merging preamble + body...")
    final_tex = _merge_tex(preamble, body)

    merge_balance = _brace_balance(final_tex)
    _log(f"📌 [BRACE CHECK] After merge: balance={merge_balance}")

    _log("📌 [STEP 20] Injecting PDF metadata...")
    final_tex = inject_pdf_metadata(
        final_tex, target_company, target_role, skills_list, [])

    meta_balance = _brace_balance(final_tex)
    _log(f"📌 [BRACE CHECK] After metadata: balance={meta_balance}")

    _log("📌 [STEP 21] Running LaTeX sanity pass...")
    final_tex = latex_sanity_pass(final_tex)

    _log("📌 [STEP 22] Computing coverage...")
    coverage = compute_coverage(final_tex, all_keywords)

    phrases_present, phrases_missing = check_phrase_coverage(final_tex, jd_phrases)
    _log(f"🔤 [PHRASE MIRROR] {len(phrases_present)}/{len(jd_phrases)} JD phrases present")
    _log(f"📊 [COVERAGE] {coverage['ratio']:.1%}")

    exp_bullet_count = _count_experience_bullets(final_tex)
    _log(f"📝 [EXP BULLETS] {exp_bullet_count} experience bullets (min={MIN_EXPERIENCE_BULLETS})")

    project_bullet_strs = []
    for p in project_entries:
        project_bullet_strs.append(
            f"{p.get('name', '')} [{p.get('tech_stack', '')}] — "
            f"{p.get('line1', '')} {p.get('line2', '')}")

    chosen_jd_title = await classify_best_intern_title(jd_text, target_role, role_archetype)

    return final_tex, {
        "jd_info": jd_info, "company_core": company_core, "ideal_candidate": ideal_candidate,
        "role_archetype": {
            "key": role_archetype.get("key"),
            "name": role_archetype.get("name"),
            "tone": role_archetype.get("tone_register"),
        },
        "jd_tasks": [{"id": t["task_id"], "desc": t["task_description"],
                       "priority": t["priority"], "category": t["task_category"]}
                     for t in jd_tasks],
        "master_plan": master_plan, "all_keywords": all_keywords, "coverage": coverage,
        "jd_phrase_coverage": {
            "present": phrases_present, "missing": phrases_missing,
            "ratio": len(phrases_present) / max(1, len(jd_phrases)),
        },
        "exp_used_keywords": list(exp_used), "skills_list": skills_list,
        "ats_extra_keywords": ats_extra,
        "global_keyword_assignments": dict(_global_kw_assignments),
        "specific_technologies_used": [],
        "project_bullets_generated": project_bullet_strs,
        "project_entries": project_entries,
        "experience_bullet_count": exp_bullet_count,
        "courses": [],
        "chosen_intern_title": chosen_jd_title,
        "title_assignments": {
            "all_blocks": chosen_jd_title,
        },
        "skills_breakdown": {
            "must": len(must_v), "core": len(core_keywords),
            "should": len(should_v), "nice": len(nice_v),
            "extra": len(extra_list), "ats_sim": len(ats_extra),
            "jd_all_skills": len(all_jd_skills),
            "total": len(skills_list),
        },
        "skills_categories": {k: len(v) for k, v in categorized_skills.items()},
        "pdf_metadata": {
            "author": "Sri Akash Kadali",
            "creator": "Sri Akash Kadali",
            "keywords_count": len(skills_list),
        },
        "education_changes": {
            "undergrad_removed": True,
            "coursework_removed": True,
            "masters_only": True,
        },
        "candidate_inventory_summary": {
            "level": candidate_inventory.get("candidate_level", "unknown"),
            "defensible_tools_count": len(candidate_inventory.get("defensible_tools", [])),
            "experience_count": len(candidate_inventory.get("experience_inventory", [])),
            "strong_jd_overlap": candidate_inventory.get("jd_overlap", {}).get("strong_overlap", []),
            "risky_claims": candidate_inventory.get("jd_overlap", {}).get("risky_or_unsupported", []),
        },
        "project_format": "advanced_2line_subheading",
        "version": "v2.10.0",
    }


# ═══════════════════════════════════════════════════════════════
# API ENDPOINT — v2.8.2
# ═══════════════════════════════════════════════════════════════

@router.post("/")
@router.post("/run")
@router.post("/submit")
async def optimize_endpoint(
    jd_text: str = Form(...),
    use_humanize: bool = Form(False),
    base_resume_tex: Optional[UploadFile] = File(None),
    extra_keywords: Optional[str] = Form(None),
):
    try:
        _ = use_humanize
        jd_text = (jd_text or "").strip()
        if not jd_text:
            raise HTTPException(400, "jd_text is required.")

        raw_tex = ""
        if base_resume_tex is not None:
            tb = await base_resume_tex.read()
            if tb:
                raw_tex = secure_tex_input(
                    base_resume_tex.filename or "upload.tex",
                    tb.decode("utf-8", errors="ignore"))
        if not raw_tex:
            dp = getattr(config, "DEFAULT_BASE_RESUME", None)
            if isinstance(dp, (str, bytes)):
                dp = Path(dp)
            if not dp or not dp.exists():
                raise HTTPException(500, "Default base resume not found")
            raw_tex = dp.read_text(encoding="utf-8")

        _log(f"📄 [INPUT] Base TeX: {len(raw_tex)} chars")

        target_company, target_role = await extract_company_role(jd_text)
        _log(f"🎯 [TARGET] {target_role} at {target_company}")
        sc, sr = safe_filename(target_company), safe_filename(target_role)

        optimized_tex, info = await optimize_resume(
            raw_tex, jd_text, target_company, target_role, extra_keywords)

        cur_tex = optimized_tex
        resume_keywords = info.get("all_keywords", [])

        # v2.8.2: Use enhanced_compile as the _compile function
        def _compile(t: str) -> bytes:
            return _enhanced_compile(
                t, sc, sr, _log, render_final_tex, latex_sanity_pass,
                _brace_balance, compile_latex_safely, config)

        _log("📌 [COMPILE] First compilation attempt...")
        cur_pdf = _compile(cur_tex)
        _log(f"📌 [COMPILE] Success! Pages: {_pdf_page_count(cur_pdf)}, Size: {len(cur_pdf)} bytes")

        trims, streak, prev = 0, 0, len(cur_pdf)

        skills_list = info.get("skills_list", [])
        _skills_trimmed_tight = False
        _skills_trimmed_emergency = False
        _additional_info_removed = False

        while trims < 80 and _pdf_page_count(cur_pdf) > 1:
            nt, ok = None, False

            nt, ok = remove_one_achievement_bullet(cur_tex)

            if not ok and not _additional_info_removed:
                nt, ok = remove_section_entirely(cur_tex, "Additional Information")
                if ok: _additional_info_removed = True

            if not ok and not _skills_trimmed_tight and len(skills_list) > MAX_SKILLS_TIGHT:
                trimmed = rank_skills_by_jd_relevance(
                    skills_list,
                    must_have=info.get("jd_info", {}).get("must_have", []),
                    should_have=info.get("jd_info", {}).get("should_have", []),
                    nice_to_have=info.get("jd_info", {}).get("nice_to_have", []),
                    core_keywords=info.get("jd_info", {}).get("all_keywords", [])[:15],
                    jd_text=jd_text, max_skills=MAX_SKILLS_TIGHT)
                trimmed_categorized = await categorize_skills_gpt(trimmed, jd_text)
                new_skills_tex = render_skills_section_categorized(trimmed_categorized)
                skills_pat = re.compile(
                    r"(\\section\*?\{Skills\}[\s\S]*?)(?=%-----------|\\section\*?\{|\\end\{document\})", re.I)
                if re.search(skills_pat, cur_tex):
                    nt = re.sub(skills_pat, lambda _: new_skills_tex + "\n", cur_tex)
                    ok = True
                    _skills_trimmed_tight = True
                    skills_list = trimmed
                    _log(f"✂️ [TRIM-SKILLS] Reduced to {len(trimmed)} (cap={MAX_SKILLS_TIGHT})")

            if not ok:
                exp_count = _count_experience_bullets(cur_tex)
                if exp_count > MIN_EXPERIENCE_BULLETS:
                    nt, ok = remove_least_relevant_bullet(
                        cur_tex, resume_keywords, ("Experience", "Projects"))
                else:
                    nt, ok = remove_least_relevant_bullet(
                        cur_tex, resume_keywords, ("Projects",))

            if not ok and not _skills_trimmed_emergency and len(skills_list) > MAX_SKILLS_EMERGENCY:
                trimmed = rank_skills_by_jd_relevance(
                    skills_list,
                    must_have=info.get("jd_info", {}).get("must_have", []),
                    should_have=info.get("jd_info", {}).get("should_have", []),
                    nice_to_have=info.get("jd_info", {}).get("nice_to_have", []),
                    core_keywords=info.get("jd_info", {}).get("all_keywords", [])[:15],
                    jd_text=jd_text, max_skills=MAX_SKILLS_EMERGENCY)
                trimmed_categorized = await categorize_skills_gpt(trimmed, jd_text)
                new_skills_tex = render_skills_section_categorized(trimmed_categorized)
                skills_pat = re.compile(
                    r"(\\section\*?\{Skills\}[\s\S]*?)(?=%-----------|\\section\*?\{|\\end\{document\})", re.I)
                if re.search(skills_pat, cur_tex):
                    nt = re.sub(skills_pat, lambda _: new_skills_tex + "\n", cur_tex)
                    ok = True
                    _skills_trimmed_emergency = True
                    skills_list = trimmed
                    _log(f"✂️ [TRIM-SKILLS-EMRG] Reduced to {len(trimmed)} (cap={MAX_SKILLS_EMERGENCY})")

            if not ok:
                exp_count = _count_experience_bullets(cur_tex)
                if exp_count > 6:
                    nt, ok = remove_least_relevant_bullet(
                        cur_tex, resume_keywords, ("Experience",))
                    if ok:
                        _log(f"⚠️ [TRIM] Removed experience bullet below MIN ({exp_count - 1} remaining)")

            if not ok:
                _log(f"🛡️ [TRIM] No more removable content")
                break

            try:
                np = _compile(nt)
            except HTTPException:
                _log(f"⚠️ [TRIM] Compilation failed after trim #{trims + 1}, stopping trim loop")
                break

            trims += 1
            ns = len(np)
            if ns >= prev:
                streak += 1
                if streak >= 4:
                    cur_tex, cur_pdf = nt, np
                    break
            else:
                streak = 0
            cur_tex, cur_pdf, prev = nt, np, ns

        cov = info["coverage"]
        ratio = float(cov.get("ratio", 0))
        score = int(round(ratio * 100))
        matched = cov.get("present", [])
        missing = cov.get("missing", [])
        verdict = ("Excellent Match" if score >= 80 else "Strong Match" if score >= 65
                   else "Good Match" if score >= 50 else "Needs Improvement")

        paths = build_output_paths(target_company, target_role)
        op = paths["optimized"]
        if cur_pdf:
            op.parent.mkdir(parents=True, exist_ok=True)
            op.write_bytes(cur_pdf)
        op = paths["temp"]
        if cur_pdf:
            op.parent.mkdir(parents=True, exist_ok=True)
            op.write_bytes(cur_pdf)
        phrase_cov = info.get("jd_phrase_coverage", {})
        final_exp_count = _count_experience_bullets(cur_tex)
        final_pages = _pdf_page_count(cur_pdf)
        _log(f"📝 [FINAL] {final_exp_count} experience bullets, {final_pages} page(s), {trims} trims")
        _log(f"✅ [DONE] Score={score}%, Verdict={verdict}")

        return JSONResponse({
            "alignment_score": score, "alignment_percent": f"{score}%",
            "matched_keywords_count": len(matched), "missing_keywords_count": len(missing),
            "confidence_score": round(min(0.99, 0.5 + ratio * 0.5), 2), "verdict": verdict,
            "eligibility": {"score": ratio, "present": matched, "missing": missing,
                            "total": cov["total"], "verdict": verdict},
            "company_name": target_company, "role": target_role,
            "optimized": {"tex": render_final_tex(cur_tex),
                          "pdf_b64": base64.b64encode(cur_pdf).decode("ascii"),
                          "filename": str(op) if cur_pdf else ""},
            "temp": {"tex": render_final_tex(cur_tex),
                          "pdf_b64": base64.b64encode(cur_pdf).decode("ascii"),
                          "filename": str(op) if cur_pdf else ""},
            "tex_string": render_final_tex(cur_tex),
            "pdf_base64": base64.b64encode(cur_pdf).decode("ascii"),
            "coverage_ratio": ratio, "coverage_present": matched, "coverage_missing": missing,
            "trim_summary": {"items_removed": trims, "final_pages": final_pages,
                             "final_experience_bullets": final_exp_count},
            "role_archetype": info.get("role_archetype", {}),
            "jd_tasks_count": len(info.get("jd_tasks", [])),
            "jd_phrase_coverage": {
                "present_count": len(phrase_cov.get("present", [])),
                "missing_count": len(phrase_cov.get("missing", [])),
                "ratio": phrase_cov.get("ratio", 0),
            },
            "technology_specificity": {
                "specific_technologies_used": info.get("specific_technologies_used", [])},
            "skills_list": skills_list,
            "skills_breakdown": info.get("skills_breakdown", {}),
            "skills_categories": info.get("skills_categories", {}),
            "ats_extra_keywords": info.get("ats_extra_keywords", []),
            "project_bullets_generated": info.get("project_bullets_generated", []),
            "project_entries": info.get("project_entries", []),
            "experience_bullet_count": final_exp_count,
            "pdf_metadata": info.get("pdf_metadata", {}),
            "courses": [],
            "chosen_intern_title": info.get("chosen_intern_title", ""),
            "title_assignments": info.get("title_assignments", {}),
            "education_changes": info.get("education_changes", {}),
            "candidate_inventory_summary": info.get("candidate_inventory_summary", {}),
            "project_format": info.get("project_format", ""),
            "version": info.get("version", "v2.10.0"),
        })
    except HTTPException:
        raise
    except Exception as e:
        _log(f"💥 [PIPELINE] Failed: {e}")
        print(f"\n{'=' * 70}", flush=True)
        print(f"💥 FULL TRACEBACK:", flush=True)
        traceback.print_exc()
        print(f"{'=' * 70}\n", flush=True)
        raise HTTPException(500, str(e))