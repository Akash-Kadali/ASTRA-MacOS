"""
Resume optimizer API (FastAPI) — ENHANCED VERSION v3.0

IMPROVEMENTS OVER v2:
- ✨ GPT-based IDEAL CANDIDATE PROFILING (implicit JD requirements)
- ✨ TOP-3 IMPLICIT REQUIREMENTS the job DEFINITELY wants
- ✨ RANKED IMPORTANCE of all bullet points
- ✨ 8 bullets from resume keywords + 4 bullets from ideal candidate insights
- ✨ NO keyword used twice across any bullet
- ✨ GPT-based capitalization (no hardcoded map)
- ✨ GPT-based company context (no hardcoded company map)
- ✨ GPT-based skill validation (removes PhD, MS, random words)
- ✨ Selective quantification: 4 numbers across 12 bullets (positions 2,6,7,10)
- ✨ UNIQUE NUMBER TRACKING — never repeats same number
- ✨ Action verb diversity tracking (no repetition across all 12)
- ✨ Result phrases without numbers
- ✨ Technical depth indicators
- ✨ Skill progression across experience blocks
- ✨ Believability constraints for intern-level
- ✨ Bullet structure templates
- ✨ Cross-bullet coherence
- ✨ Industry-specific vocabulary via GPT
"""

import base64
import json
import re
import asyncio
import threading
import random
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, Optional, Set, Any

# --- third-party ---
from fastapi import APIRouter, UploadFile, Form, File, HTTPException
from fastapi.responses import JSONResponse

from backend.core import config
from backend.core.compiler import compile_latex_safely
from backend.core.security import secure_tex_input
from backend.core.utils import log_event, safe_filename, build_output_paths
from backend.api.render_tex import render_final_tex

router = APIRouter(prefix="/api/optimize", tags=["optimize"])

# --- OpenAI ---
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


# ============================================================
# 🧠 GPT Helper
# ============================================================

def _json_from_text(text: str, default: Any):
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return default
    try:
        return json.loads(m.group(0))
    except Exception:
        return default


async def gpt_json(prompt: str, temperature: float = 0.0, model: str = "gpt-4o-mini") -> dict:
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


# ============================================================
# 🎲 REALISTIC NUMBER GENERATION WITH UNIQUE TRACKING
# ============================================================

_used_numbers_by_category: Dict[str, Set[str]] = {
    "percent": set(), "count": set(), "metric": set(), "comparison": set()
}
_quantified_bullet_positions: Set[int] = set()


def reset_number_tracking():
    global _used_numbers_by_category, _quantified_bullet_positions
    _used_numbers_by_category = {
        "percent": set(), "count": set(), "metric": set(), "comparison": set()
    }
    _quantified_bullet_positions.clear()


def generate_messy_decimal(min_val: float, max_val: float, decimal_places: int = 2) -> float:
    for _ in range(50):
        num = random.uniform(min_val, max_val)
        rounded = round(num, decimal_places)
        if decimal_places == 2:
            last_digit = int((rounded * 100) % 10)
        elif decimal_places == 1:
            last_digit = int((rounded * 10) % 10)
        else:
            last_digit = int(rounded % 10)
        if last_digit % 2 != 0:
            return rounded
    return round(min_val, decimal_places)


def generate_messy_number(category: str, jd_context: str = "") -> str:
    global _used_numbers_by_category
    for attempt in range(100):
        formatted = ""
        if category == "percent":
            if "accuracy" in jd_context.lower() or "precision" in jd_context.lower():
                base = random.randint(11, 29)
            else:
                base = random.randint(7, 89)
            if random.random() < 0.6:
                decimal = generate_messy_decimal(base, base + 0.99, 2)
                formatted = f"{decimal}%"
            else:
                if base % 2 == 0:
                    base += 1
                formatted = f"{base}%"
        elif category == "count":
            base = random.choice([random.randint(567, 9999), random.randint(10000, 99999)])
            if base % 2 == 0:
                base += 1
            if base >= 10000:
                formatted = f"{base // 1000}K+"
            elif base >= 1000:
                formatted = f"{base:,}"
            else:
                formatted = str(base)
        elif category == "metric":
            metric_name = random.choice(["F1 score", "precision", "recall", "accuracy"])
            value = generate_messy_decimal(0.73, 0.97, 2)
            formatted = f"{metric_name} of {value}"
        elif category == "comparison":
            if "accuracy" in jd_context.lower():
                start = random.randint(51, 67)
                improvement = random.randint(17, 33)
            else:
                start = random.randint(41, 79)
                improvement = random.randint(11, 29)
            if start % 2 == 0:
                start += 1
            if improvement % 2 == 0:
                improvement += 1
            end = min(99, start + improvement)
            if end % 2 == 0:
                end -= 1
            formatted = f"from {start}% to {end}%"

        if formatted and formatted not in _used_numbers_by_category[category]:
            _used_numbers_by_category[category].add(formatted)
            log_event(f"🎲 [NUMBER-{category.upper()}] Generated: {formatted}")
            return formatted

    fallback = f"{random.randint(13, 87)}%"
    _used_numbers_by_category[category].add(fallback)
    return fallback


QUANTIFICATION_TEMPLATES = {
    "percent_improvement": [
        "improving {metric} by {value}",
        "achieving {value} enhancement in {metric}",
        "boosting {metric} performance by {value}",
        "elevating {metric} through {value} improvement",
    ],
    "count_scale": [
        "processing {value} data samples daily",
        "analyzing {value} records for pattern detection",
        "handling {value} concurrent model predictions",
        "evaluating {value} feature combinations systematically",
    ],
    "metric_achievement": [
        "attaining {value} on validation datasets",
        "reaching {value} across diverse test scenarios",
        "delivering {value} in production deployment",
        "securing {value} during rigorous evaluation",
    ],
    "comparison_hero": [
        "improving model accuracy {value} through systematic hyperparameter optimization",
        "enhancing prediction reliability {value} via ensemble methodology",
        "accelerating inference speed {value} using optimized architecture",
        "increasing system throughput {value} with distributed processing",
    ],
}


def generate_quantified_phrase(category: str, jd_context: str = "") -> str:
    templates = QUANTIFICATION_TEMPLATES.get(category, QUANTIFICATION_TEMPLATES["percent_improvement"])
    template = random.choice(templates)
    if category == "percent_improvement":
        metric = random.choice(["accuracy", "precision", "throughput", "efficiency"])
        value = generate_messy_number("percent", jd_context)
        return template.format(metric=metric, value=value)
    elif category == "count_scale":
        value = generate_messy_number("count", jd_context)
        return template.format(value=value)
    elif category == "metric_achievement":
        value = generate_messy_number("metric", jd_context)
        return template.format(value=value)
    elif category == "comparison_hero":
        value = generate_messy_number("comparison", jd_context)
        return template.format(value=value)
    return template


# Quantified positions: absolute bullet index 0-11
QUANTIFIED_POSITIONS = [1, 5, 6, 9]  # Positions 2, 6, 7, 10 (0-indexed)
HERO_POSITIONS = [1, 5]


def reset_quantification_tracking():
    reset_number_tracking()


def get_quantification_category(bullet_position: int, jd_context: str = "") -> Optional[str]:
    if bullet_position not in QUANTIFIED_POSITIONS:
        return None
    category_map = {
        1: "comparison_hero",
        5: "comparison_hero",
        6: "count_scale",
        9: "metric_achievement",
    }
    if bullet_position == 1 and "accuracy" not in jd_context.lower():
        return "percent_improvement"
    return category_map.get(bullet_position, None)


def should_quantify_bullet(bullet_position: int) -> bool:
    return bullet_position in QUANTIFIED_POSITIONS


# ============================================================
# 💪 ACTION VERB MANAGEMENT — NO REPETITION ACROSS ALL 12 BULLETS
# ============================================================

ACTION_VERBS = {
    "development": [
        "Architected", "Engineered", "Developed", "Built", "Implemented",
        "Constructed", "Designed", "Created", "Established", "Formulated",
        "Programmed", "Prototyped", "Assembled",
    ],
    "research": [
        "Investigated", "Explored", "Analyzed", "Evaluated", "Validated",
        "Examined", "Studied", "Researched", "Assessed", "Characterized",
        "Scrutinized", "Probed",
    ],
    "optimization": [
        "Optimized", "Enhanced", "Streamlined", "Accelerated", "Refined",
        "Improved", "Strengthened", "Advanced", "Elevated", "Augmented",
        "Amplified", "Intensified",
    ],
    "data_work": [
        "Processed", "Transformed", "Aggregated", "Curated", "Cleaned",
        "Structured", "Organized", "Consolidated", "Standardized", "Normalized",
        "Synthesized", "Compiled",
    ],
    "ml_training": [
        "Trained", "Fine-tuned", "Calibrated", "Tuned", "Configured",
        "Parameterized", "Adapted", "Specialized", "Customized", "Fitted",
        "Conditioned", "Adjusted",
    ],
    "deployment": [
        "Deployed", "Launched", "Released", "Shipped", "Delivered",
        "Productionized", "Operationalized", "Integrated", "Provisioned", "Staged",
        "Rolled-out", "Instituted",
    ],
    "analysis": [
        "Analyzed", "Diagnosed", "Identified", "Discovered", "Uncovered",
        "Detected", "Recognized", "Profiled", "Mapped", "Quantified",
        "Interpreted", "Dissected",
    ],
    "collaboration": [
        "Collaborated", "Partnered", "Coordinated", "Facilitated", "Supported",
        "Contributed", "Assisted", "Engaged", "Interfaced", "Liaised",
        "Cooperated", "Unified",
    ],
    "automation": [
        "Automated", "Systematized", "Scripted", "Programmed", "Orchestrated",
        "Scheduled", "Templated", "Codified", "Mechanized", "Streamlined",
        "Roboticized", "Computerized",
    ],
    "documentation": [
        "Documented", "Recorded", "Cataloged", "Annotated", "Detailed",
        "Specified", "Outlined", "Summarized", "Reported", "Communicated",
        "Chronicled", "Transcribed",
    ],
}

_used_verbs_global: Set[str] = set()


def reset_verb_tracking():
    global _used_verbs_global
    _used_verbs_global.clear()


def get_diverse_verb(category: str, fallback: str = "Developed") -> str:
    global _used_verbs_global
    verbs = ACTION_VERBS.get(category, ACTION_VERBS["development"])
    available = [v for v in verbs if v.lower() not in _used_verbs_global]
    if not available:
        all_verbs = [v for cat in ACTION_VERBS.values() for v in cat]
        available = [v for v in all_verbs if v.lower() not in _used_verbs_global]
    if not available:
        chosen = fallback
    else:
        chosen = random.choice(available)
    _used_verbs_global.add(chosen.lower())
    log_event(f"✅ [VERB] Selected: {chosen} (Total used: {len(_used_verbs_global)}/12)")
    return chosen


def get_verb_categories_for_context(company_type: str) -> List[str]:
    if "research" in company_type.lower():
        return ["research", "analysis", "development", "documentation"]
    elif "industry" in company_type.lower():
        return ["development", "deployment", "optimization", "automation"]
    else:
        return ["development", "analysis", "collaboration", "data_work"]


# ============================================================
# 🎯 RESULT PHRASES (Impact without numbers)
# ============================================================

RESULT_PHRASES = {
    "performance": [
        "achieving enhanced model generalization across diverse datasets",
        "resulting in improved prediction accuracy on held-out test data",
        "enabling robust performance under varying input conditions",
        "delivering production-grade model reliability and consistency",
        "attaining competitive benchmark results against established baselines",
    ],
    "efficiency": [
        "enabling faster experimentation and iteration cycles",
        "streamlining the end-to-end development workflow significantly",
        "reducing computational overhead while maintaining output quality",
        "accelerating model training and evaluation throughput",
        "improving overall resource utilization and pipeline efficiency",
    ],
    "quality": [
        "ensuring high-quality and reproducible model outputs",
        "maintaining rigorous quality standards throughout development",
        "achieving consistent and reliable experimental results",
        "delivering enterprise-grade code quality and documentation",
        "meeting stringent production readiness requirements",
    ],
    "scalability": [
        "supporting seamless scaling to larger datasets",
        "enabling distributed processing capabilities for production workloads",
        "facilitating efficient handling of increased data volumes",
        "ensuring system robustness under production-scale demands",
        "accommodating future growth and extensibility requirements",
    ],
    "insight": [
        "uncovering actionable insights from complex data patterns",
        "revealing previously hidden correlations and trends",
        "generating valuable intelligence for downstream applications",
        "providing data-driven recommendations for model improvements",
        "enabling informed decision-making through rigorous analysis",
    ],
    "collaboration": [
        "facilitating cross-functional collaboration and knowledge sharing",
        "enabling seamless integration with existing team workflows",
        "supporting reproducibility and handoff to other team members",
        "improving documentation and codebase maintainability",
        "establishing reusable components for future projects",
    ],
}

_used_result_phrases: Set[str] = set()


def reset_result_phrase_tracking():
    global _used_result_phrases
    _used_result_phrases.clear()


def get_result_phrase(category: str) -> str:
    global _used_result_phrases
    phrases = RESULT_PHRASES.get(category, RESULT_PHRASES["performance"])
    available = [p for p in phrases if p not in _used_result_phrases]
    if not available:
        _used_result_phrases.clear()
        available = phrases
    chosen = random.choice(available)
    _used_result_phrases.add(chosen)
    return chosen


# ============================================================
# 🔬 TECHNICAL DEPTH INDICATORS
# ============================================================

TECHNICAL_DEPTH_PHRASES = {
    "ml_techniques": [
        "employing stratified Cross-Validation for robust evaluation",
        "utilizing Grid Search and Bayesian optimization for Hyperparameter Tuning",
        "applying advanced Feature Engineering with domain-specific transformations",
        "implementing custom Data Augmentation strategies for improved generalization",
        "leveraging Ensemble Methods to combine multiple model predictions",
        "conducting systematic Ablation Studies to validate design choices",
    ],
    "dl_techniques": [
        "incorporating Batch Normalization and Dropout for regularization",
        "implementing Learning Rate Scheduling with warm restarts",
        "utilizing Gradient Clipping to stabilize training dynamics",
        "applying Transfer Learning with frozen backbone and fine-tuned heads",
        "employing Attention Mechanisms for improved feature representation",
        "implementing residual connections for gradient flow optimization",
    ],
    "data_techniques": [
        "implementing comprehensive Data Preprocessing pipelines with validation",
        "applying Dimensionality Reduction for efficient feature representation",
        "utilizing robust Outlier Detection and handling strategies",
        "implementing Missing Value Imputation with multiple strategies",
        "applying class balancing techniques for imbalanced datasets",
        "conducting thorough Exploratory Data Analysis for insight generation",
    ],
    "mlops_techniques": [
        "implementing Model Versioning with comprehensive experiment tracking",
        "establishing CI/CD pipelines for automated model validation",
        "utilizing containerization with Docker for reproducible deployments",
        "implementing Feature Store patterns for consistent feature serving",
        "establishing Model Monitoring dashboards for production oversight",
        "applying infrastructure-as-code practices for environment management",
    ],
    "evaluation_techniques": [
        "conducting Precision-Recall analysis for classification performance",
        "implementing comprehensive error analysis and failure mode identification",
        "utilizing statistical significance testing for model comparisons",
        "applying Confusion Matrix analysis for multi-class evaluation",
        "implementing custom evaluation metrics aligned with business objectives",
        "conducting systematic bias and fairness audits",
    ],
}


def get_technical_depth_phrase(category: str) -> str:
    phrases = TECHNICAL_DEPTH_PHRASES.get(category, TECHNICAL_DEPTH_PHRASES["ml_techniques"])
    return random.choice(phrases)


# ============================================================
# 📈 SKILL PROGRESSION FRAMEWORK
# ============================================================

INTERN_PROGRESSION = {
    "early": {
        "scope": ["assisted", "supported", "contributed to", "participated in"],
        "tasks": ["data preprocessing", "baseline implementation", "literature review", "code documentation"],
        "autonomy": "under guidance of senior engineers",
        "complexity": "foundational components",
    },
    "mid": {
        "scope": ["developed", "implemented", "designed", "built"],
        "tasks": ["model development", "pipeline creation", "experiment execution", "performance analysis"],
        "autonomy": "with mentorship from team leads",
        "complexity": "core system components",
    },
    "late": {
        "scope": ["led", "architected", "spearheaded", "owned"],
        "tasks": ["end-to-end pipeline", "model optimization", "deployment preparation", "technical documentation"],
        "autonomy": "independently with periodic reviews",
        "complexity": "production-ready solutions",
    },
}


def get_progression_context(block_index: int, total_blocks: int = 4) -> Dict[str, Any]:
    if block_index == 0:
        return INTERN_PROGRESSION["late"]
    elif block_index == total_blocks - 1:
        return INTERN_PROGRESSION["early"]
    else:
        return INTERN_PROGRESSION["mid"]


# ============================================================
# 🏭 BELIEVABILITY CONSTRAINTS
# ============================================================

BELIEVABILITY_RULES = {
    "collaboration_phrases": [
        "in collaboration with senior engineers",
        "as part of a cross-functional team",
        "working closely with research mentors",
        "under guidance of technical leads",
        "contributing to team-wide initiatives",
    ]
}


def get_believability_phrase(scope: str = "medium") -> str:
    if random.random() < 0.3:
        return random.choice(BELIEVABILITY_RULES["collaboration_phrases"])
    return ""


# ============================================================
# 🔠 GPT-BASED CAPITALIZATION (replaces hardcoded map)
# ============================================================

_capitalization_cache: Dict[str, str] = {}


async def fix_capitalization_gpt(text: str) -> str:
    """Use GPT to fix capitalization of ALL technical terms in a text block."""
    if not text or len(text.strip()) < 3:
        return text

    # Check cache for short strings (skill names)
    text_lower = text.lower().strip()
    if text_lower in _capitalization_cache:
        return _capitalization_cache[text_lower]

    prompt = f"""Fix the capitalization of ALL technical terms in this text. 
Return STRICT JSON: {{"fixed": "the corrected text"}}

Rules:
- Programming languages: Python, Java, JavaScript, C++, SQL, R, MATLAB, Go, Rust
- Frameworks: PyTorch, TensorFlow, Keras, Scikit-learn, NumPy, Pandas, FastAPI, React
- Tools: Docker, Kubernetes, Git, GitHub, AWS, GCP, Azure, MLflow, Airflow, Spark
- Acronyms: ML, AI, NLP, CV, CNN, RNN, LSTM, BERT, GPT, LLM, API, REST, CI/CD, ETL
- Concepts: Machine Learning, Deep Learning, Natural Language Processing, Computer Vision
- Databases: PostgreSQL, MongoDB, Redis, MySQL, DynamoDB, Elasticsearch
- Keep sentence structure intact, only fix capitalization of tech terms
- If text is a single skill/keyword, just return it properly capitalized

Text: "{text}"
"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        fixed = data.get("fixed", text).strip()
        if len(text_lower) < 50:
            _capitalization_cache[text_lower] = fixed
        return fixed
    except Exception:
        return text


async def fix_capitalization_batch(items: List[str]) -> List[str]:
    """Fix capitalization for a batch of skill/keyword strings using one GPT call."""
    if not items:
        return []

    # Check cache first
    uncached = []
    cached_results = {}
    for item in items:
        key = item.lower().strip()
        if key in _capitalization_cache:
            cached_results[key] = _capitalization_cache[key]
        else:
            uncached.append(item)

    if not uncached:
        return [cached_results.get(i.lower().strip(), i) for i in items]

    prompt = f"""Fix capitalization of these technical keywords/skills for a resume.
Return STRICT JSON: {{"fixed": ["Python", "PyTorch", "Machine Learning", ...]}}

Rules:
- Programming languages: Python, Java, JavaScript, C++, SQL, R, MATLAB
- Frameworks: PyTorch, TensorFlow, Keras, Scikit-learn, NumPy, Pandas, FastAPI
- Tools: Docker, Kubernetes, Git, AWS, GCP, Azure, MLflow, Spark, Airflow
- Acronyms: ML, AI, NLP, CV, CNN, RNN, LSTM, BERT, GPT, LLM, API, REST, CI/CD
- Concepts: Machine Learning, Deep Learning, Natural Language Processing
- Databases: PostgreSQL, MongoDB, Redis, MySQL, DynamoDB
- Each keyword should have first letter capitalized if not an acronym
- Preserve multi-word terms as-is except for capitalization fixes

Keywords: {json.dumps(uncached)}
"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        fixed_list = data.get("fixed", uncached)
        if len(fixed_list) != len(uncached):
            fixed_list = uncached

        for orig, fixed in zip(uncached, fixed_list):
            key = orig.lower().strip()
            _capitalization_cache[key] = str(fixed).strip()

        result = []
        for item in items:
            key = item.lower().strip()
            if key in _capitalization_cache:
                result.append(_capitalization_cache[key])
            elif key in cached_results:
                result.append(cached_results[key])
            else:
                result.append(item)
        return result
    except Exception:
        return items


def _ensure_first_letter_capital(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    if s[0].isalpha() and s[0].islower():
        return s[0].upper() + s[1:]
    return s


async def fix_skill_capitalization(skill: str) -> str:
    """Fix capitalization for a single skill term via cache or GPT."""
    skill = (skill or "").strip()
    if not skill:
        return ""
    key = skill.lower().strip()
    if key in _capitalization_cache:
        return _capitalization_cache[key]
    fixed = await fix_capitalization_gpt(skill)
    fixed = _ensure_first_letter_capital(fixed)
    _capitalization_cache[key] = fixed
    return fixed


def fix_skill_capitalization_sync(skill: str) -> str:
    """Sync version — uses cache only, no GPT call."""
    skill = (skill or "").strip()
    if not skill:
        return ""
    key = skill.lower().strip()
    if key in _capitalization_cache:
        return _capitalization_cache[key]
    return _ensure_first_letter_capital(skill)


# ============================================================
# ✅ SKILL VALIDATION using GPT — STRICT FILTERING
# ============================================================

_validated_skills_cache: Dict[str, bool] = {}


async def is_valid_skill(keyword: str) -> bool:
    global _validated_skills_cache
    keyword_lower = keyword.lower().strip()
    if keyword_lower in _validated_skills_cache:
        return _validated_skills_cache[keyword_lower]

    non_skills = {
        "phd", "ph.d", "ms", "m.s", "msc", "m.sc", "bs", "b.s", "bsc", "b.sc",
        "bachelor", "master", "masters", "degree", "university", "college",
        "experience", "years", "year", "month", "months", "week", "weeks",
        "required", "preferred", "plus", "bonus", "nice to have",
        "strong", "excellent", "good", "proficient", "familiar", "advanced", "basic",
        "knowledge", "understanding", "ability", "skills", "skill",
        "iso", "nist", "gdpr", "hipaa", "sox", "pci", "cmmi", "itil",
        "compliance", "certified", "certification", "framework", "standard",
        "iso 42001", "nist ai rmf", "ai rmf", "rmf",
    }
    if keyword_lower in non_skills:
        _validated_skills_cache[keyword_lower] = False
        log_event(f"❌ [SKILL FAST-REJECT] '{keyword}' → Non-skill")
        return False

    if re.match(r"^(iso|nist|pci|gdpr)\s+\d+", keyword_lower):
        _validated_skills_cache[keyword_lower] = False
        log_event(f"❌ [SKILL PATTERN-REJECT] '{keyword}' → Standard pattern")
        return False

    prompt = f"""Is "{keyword}" a HARD TECHNICAL SKILL or ESSENTIAL SOFT SKILL for a resume?

**ACCEPT (true):** Programming languages, frameworks, libraries, tools, platforms, databases,
technical concepts (Machine Learning, System Design, etc.), essential soft skills ONLY
(Leadership, Communication, Problem-Solving, Teamwork).

**REJECT (false):** Standards/certifications (ISO, NIST, GDPR), degrees, generic qualifiers,
time periods, vague terms, company names, job requirements.

Return STRICT JSON: {{"is_skill": true}} or {{"is_skill": false}}
Keyword: "{keyword}"
"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        is_skill = data.get("is_skill", False)
        _validated_skills_cache[keyword_lower] = bool(is_skill)
        status = "✅" if is_skill else "❌"
        log_event(f"{status} [SKILL GPT] '{keyword}' → {is_skill}")
        return bool(is_skill)
    except Exception as e:
        log_event(f"⚠️ [SKILL VALIDATION] Failed for '{keyword}': {e}")
        _validated_skills_cache[keyword_lower] = False
        return False


async def filter_valid_skills(keywords: List[str]) -> List[str]:
    if not keywords:
        return []
    tasks = [is_valid_skill(kw) for kw in keywords]
    results = await asyncio.gather(*tasks)
    valid_skills = [kw for kw, is_valid in zip(keywords, results) if is_valid]
    removed = set(keywords) - set(valid_skills)
    if removed:
        log_event(f"🧹 [SKILL FILTER] Removed {len(removed)} non-skills: {', '.join(list(removed)[:5])}")
    return valid_skills


# ============================================================
# 🏢 GPT-BASED COMPANY CONTEXT (replaces hardcoded map)
# ============================================================

_company_context_cache: Dict[str, Dict[str, Any]] = {}


async def get_company_context_gpt(company_name: str) -> Dict[str, Any]:
    """Use GPT to generate company context dynamically instead of hardcoded map."""
    name_lower = (company_name or "").lower().strip()
    if name_lower in _company_context_cache:
        return _company_context_cache[name_lower]

    prompt = f"""Analyze this company/institution for resume bullet writing context.
Return STRICT JSON:
{{
    "type": "industry_internship or research_internship or internship",
    "domain": "2-4 word domain description",
    "context": "1-2 sentence description of what ML/AI work is done here",
    "technical_vocabulary": ["5-8 domain-specific technical terms used at this company"],
    "ml_projects": ["3-5 realistic ML project descriptions for an intern here"],
    "believable_tasks": ["8-12 tasks an ML intern would realistically do here"],
    "progression_tasks": {{
        "early": ["3-4 early-stage intern tasks"],
        "mid": ["3-4 mid-stage intern tasks"],
        "late": ["3-4 late-stage intern tasks"]
    }}
}}

Company/Institution: "{company_name}"

Rules:
- Be REALISTIC about what an intern would actually do
- For universities/research institutions, focus on research internship context
- For companies, focus on industry internship context
- Technical vocabulary should be domain-specific, not generic
"""
    try:
        data = await gpt_json(prompt, temperature=0.2)
        result = {
            "type": data.get("type", "internship"),
            "domain": data.get("domain", "ML/AI"),
            "context": data.get("context", "Technical internship applying Machine Learning."),
            "technical_vocabulary": data.get("technical_vocabulary", ["model development", "data analysis"]),
            "ml_projects": data.get("ml_projects", ["ML Model Development"]),
            "believable_tasks": data.get("believable_tasks", ["Model Development", "Data Analysis"]),
            "progression_tasks": data.get("progression_tasks", {
                "early": ["learning", "documentation"],
                "mid": ["implementation", "testing"],
                "late": ["optimization", "delivery"],
            }),
        }
        _company_context_cache[name_lower] = result
        log_event(f"🏢 [COMPANY CONTEXT] Generated for '{company_name}': type={result['type']}")
        return result
    except Exception as e:
        log_event(f"⚠️ [COMPANY CONTEXT] Failed for '{company_name}': {e}")
        fallback = {
            "type": "internship",
            "domain": "ML/AI",
            "context": "Technical internship applying Machine Learning and Data Science.",
            "technical_vocabulary": ["model development", "data analysis", "pipeline"],
            "ml_projects": ["ML Model Development", "Data Pipeline Creation"],
            "believable_tasks": ["Model Development", "Data Analysis", "Testing", "Documentation"],
            "progression_tasks": {
                "early": ["learning", "documentation"],
                "mid": ["implementation", "testing"],
                "late": ["optimization", "delivery"],
            },
        }
        _company_context_cache[name_lower] = fallback
        return fallback


# ============================================================
# 🏢 Company Core Expectations (target employer) — GPT-based
# ============================================================

_company_core_cache: Dict[str, Dict[str, Any]] = {}


async def extract_company_core_requirements(
    target_company: str, target_role: str, jd_text: str,
) -> Dict[str, Any]:
    ckey = (target_company or "").strip().lower()
    rkey = (target_role or "").strip().lower()
    cache_key = f"{ckey}__{rkey}"
    if cache_key in _company_core_cache:
        return _company_core_cache[cache_key]

    if not ckey or ckey in {"company", "unknown"}:
        out = {
            "core_areas": ["System Design", "Experimentation", "Distributed Systems"],
            "core_keywords": ["System Design", "Distributed Systems", "A/B Testing", "Data Pipelines", "Scalability"],
            "notes": "Generic big-tech expectations used.",
        }
        _company_core_cache[cache_key] = out
        return out

    prompt = (
        "You are building an ATS-focused resume optimizer.\n"
        "Infer KEY COMPANY EXPECTATIONS for the target employer often NOT stated in JD.\n\n"
        f"Target Company: {target_company}\nTarget Role: {target_role}\n\n"
        "Rules:\n"
        '- Return STRICT JSON: {"core_areas":["..."],"core_keywords":["..."],"notes":"..."}\n'
        "- core_areas: 3-6 high-level areas (2-4 words each)\n"
        "- core_keywords: 8-14 resume-friendly skills/topics/tools commonly expected\n"
        "- Do NOT include standards (ISO, NIST), certifications, or compliance terms\n"
        "- Do NOT invent proprietary internal tool names\n"
        "- Keep tokens short (1-4 words)\n\n"
        f"JD (context only):\n{jd_text[:2500]}"
    )
    try:
        data = await gpt_json(prompt, temperature=0.0)
        core_areas = data.get("core_areas", []) or []
        core_kw = data.get("core_keywords", []) or []
        notes = (data.get("notes", "") or "").strip()

        # Fix capitalization via GPT batch
        core_areas = await fix_capitalization_batch([str(x).strip() for x in core_areas if str(x).strip()])
        core_kw = await fix_capitalization_batch([str(x).strip() for x in core_kw if str(x).strip()])

        # Deduplicate
        seen: Set[str] = set()
        deduped_areas, deduped_kw = [], []
        for a in core_areas:
            if a.lower() not in seen:
                seen.add(a.lower())
                deduped_areas.append(a)
        for k in core_kw:
            if k.lower() not in seen:
                seen.add(k.lower())
                deduped_kw.append(k)

        out = {"core_areas": deduped_areas[:8], "core_keywords": deduped_kw[:18], "notes": notes}
        _company_core_cache[cache_key] = out
        log_event(f"🏢 [COMPANY CORE] {target_company} areas={len(deduped_areas)} keywords={len(deduped_kw)}")
        return out
    except Exception as e:
        log_event(f"⚠️ [COMPANY CORE] Failed: {e}")
        out = {
            "core_areas": ["System Design", "Experimentation", "Distributed Systems"],
            "core_keywords": ["System Design", "Distributed Systems", "A/B Testing", "Data Pipelines"],
            "notes": "Fallback profile used.",
        }
        _company_core_cache[cache_key] = out
        return out


# ============================================================
# ✨ NEW: IDEAL CANDIDATE PROFILING — Implicit JD Requirements
# ============================================================

_ideal_candidate_cache: Dict[str, Dict[str, Any]] = {}


async def profile_ideal_candidate(
    jd_text: str, target_company: str, target_role: str,
) -> Dict[str, Any]:
    """
    ✨ NEW FEATURE: Ask GPT who the IDEAL candidate is for this job.
    Extracts IMPLICIT requirements not explicitly in JD.
    Returns ranked points by importance with top-3 must-haves.
    """
    cache_key = f"{(target_company or '').lower()}__{(target_role or '').lower()}"
    if cache_key in _ideal_candidate_cache:
        return _ideal_candidate_cache[cache_key]

    prompt = f"""You are a senior technical recruiter at {target_company} hiring for {target_role}.

JOB DESCRIPTION:
{jd_text[:3000]}

Think deeply about what this job REALLY needs beyond what's written. What would the IDEAL candidate have done in their past experience?

Return STRICT JSON:
{{
    "ideal_profile_summary": "2-3 sentence description of the ideal candidate",
    "implicit_requirements": [
        {{
            "requirement": "What the job implicitly wants (1 short sentence)",
            "importance_rank": 1,
            "why_implicit": "Why this isn't stated but critical",
            "bullet_theme": "A concrete resume bullet theme showing this capability"
        }},
        ... (exactly 6 implicit requirements, ranked 1-6 by importance)
    ],
    "top_3_must_haves": [
        "Concrete thing #1 the job DEFINITELY wants candidates to have done",
        "Concrete thing #2 the job DEFINITELY wants candidates to have done",
        "Concrete thing #3 the job DEFINITELY wants candidates to have done"
    ],
    "ideal_candidate_bullet_themes": [
        "4 specific resume bullet THEMES that would make this candidate stand out",
        "Each theme is a concrete past-experience description (not a skill name)",
        "These are things NOT derivable from JD keywords alone",
        "They come from understanding what the company truly values"
    ],
    "differentiation_factors": [
        "3-4 factors that separate a good candidate from a great one for this role"
    ]
}}

CRITICAL RULES:
- Focus on IMPLICIT requirements — things not directly stated in JD
- Think about: company culture, team dynamics, hidden expectations
- Think about: what past projects would impress this specific team
- top_3_must_haves should be CONCRETE PAST EXPERIENCES, not skills
- ideal_candidate_bullet_themes should be ACTIONABLE resume bullet ideas
- Each bullet theme must be distinct and non-overlapping
"""
    try:
        data = await gpt_json(prompt, temperature=0.3, model="gpt-4o-mini")

        result = {
            "ideal_profile_summary": data.get("ideal_profile_summary", ""),
            "implicit_requirements": data.get("implicit_requirements", [])[:6],
            "top_3_must_haves": data.get("top_3_must_haves", [])[:3],
            "ideal_candidate_bullet_themes": data.get("ideal_candidate_bullet_themes", [])[:4],
            "differentiation_factors": data.get("differentiation_factors", [])[:4],
        }

        _ideal_candidate_cache[cache_key] = result
        log_event(f"🌟 [IDEAL CANDIDATE] Profiled for {target_company}/{target_role}")
        log_event(f"   Top 3 must-haves: {result['top_3_must_haves']}")
        log_event(f"   Implicit reqs: {len(result['implicit_requirements'])}")
        log_event(f"   Bullet themes: {len(result['ideal_candidate_bullet_themes'])}")
        return result
    except Exception as e:
        log_event(f"⚠️ [IDEAL CANDIDATE] Failed: {e}")
        fallback = {
            "ideal_profile_summary": "A strong ML engineer with hands-on experience.",
            "implicit_requirements": [],
            "top_3_must_haves": [
                "Built end-to-end ML pipelines from data to deployment",
                "Worked with large-scale datasets in production environments",
                "Demonstrated ability to iterate quickly on model experiments",
            ],
            "ideal_candidate_bullet_themes": [
                "End-to-end ML pipeline development with production deployment",
                "Large-scale data processing and feature engineering",
                "Model experimentation with systematic evaluation",
                "Cross-functional collaboration on ML-driven products",
            ],
            "differentiation_factors": [
                "Production ML experience", "Scale of data handled",
                "Speed of experimentation", "Business impact awareness",
            ],
        }
        _ideal_candidate_cache[cache_key] = fallback
        return fallback


async def rank_all_bullet_points(
    jd_text: str,
    target_company: str,
    target_role: str,
    jd_keywords: List[str],
    ideal_candidate: Dict[str, Any],
) -> Dict[str, Any]:
    """
    ✨ NEW: Rank ALL bullet point themes by importance.
    Determines which 8 come from keywords and which 4 from ideal candidate.
    Ensures NO keyword is used twice.
    """
    top_3 = ideal_candidate.get("top_3_must_haves", [])
    bullet_themes = ideal_candidate.get("ideal_candidate_bullet_themes", [])
    implicit_reqs = ideal_candidate.get("implicit_requirements", [])

    prompt = f"""You are optimizing a resume for {target_role} at {target_company}.

JD KEYWORDS AVAILABLE (each can only be used ONCE across all 12 bullets):
{json.dumps(jd_keywords[:30])}

IDEAL CANDIDATE INSIGHTS:
- Top 3 must-haves: {json.dumps(top_3)}
- Bullet themes from ideal candidate analysis: {json.dumps(bullet_themes)}
- Implicit requirements: {json.dumps([r.get('requirement', '') for r in implicit_reqs[:6]])}

TASK: Create a plan for 12 resume bullets across 4 experience blocks (3 bullets each).

Return STRICT JSON:
{{
    "keyword_bullets": [
        {{
            "bullet_index": 0,
            "block_index": 0,
            "primary_keyword": "one keyword from JD (UNIQUE, never repeated)",
            "secondary_keywords": ["1-2 supporting keywords"],
            "theme": "What this bullet should demonstrate",
            "source": "keyword"
        }},
        ... (exactly 8 bullets sourced from JD keywords)
    ],
    "ideal_candidate_bullets": [
        {{
            "bullet_index": 8,
            "block_index": 2,
            "theme": "What this bullet should demonstrate (from ideal candidate analysis)",
            "implicit_requirement": "Which implicit requirement this addresses",
            "supporting_keywords": ["1-2 JD keywords to weave in naturally"],
            "source": "ideal_candidate"
        }},
        ... (exactly 4 bullets sourced from ideal candidate insights)
    ],
    "keyword_usage_map": {{
        "keyword1": 3,
        "keyword2": 7
    }},
    "importance_ranking": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}}

CRITICAL RULES:
1. EXACTLY 8 keyword_bullets + 4 ideal_candidate_bullets = 12 total
2. Each JD keyword can appear as primary_keyword in ONLY ONE bullet
3. secondary_keywords should also be unique where possible
4. Spread keyword_bullets across all 4 blocks (2 per block)
5. Spread ideal_candidate_bullets: 1 per block
6. importance_ranking: order all 12 bullet indices from most to least important
7. The 4 ideal_candidate bullets should cover the top_3_must_haves
8. Block 0 = most recent experience, Block 3 = oldest
"""
    try:
        data = await gpt_json(prompt, temperature=0.2)
        keyword_bullets = data.get("keyword_bullets", [])[:8]
        ideal_bullets = data.get("ideal_candidate_bullets", [])[:4]
        importance = data.get("importance_ranking", list(range(12)))
        usage_map = data.get("keyword_usage_map", {})

        log_event(f"📋 [BULLET PLAN] keyword_bullets={len(keyword_bullets)}, ideal_bullets={len(ideal_bullets)}")
        log_event(f"📋 [IMPORTANCE] Top 3 most important: {importance[:3]}")

        return {
            "keyword_bullets": keyword_bullets,
            "ideal_candidate_bullets": ideal_bullets,
            "importance_ranking": importance,
            "keyword_usage_map": usage_map,
        }
    except Exception as e:
        log_event(f"⚠️ [BULLET RANKING] Failed: {e}")
        # Fallback: simple distribution
        keyword_bullets = []
        for i in range(8):
            block = i // 2
            kw = jd_keywords[i] if i < len(jd_keywords) else "Machine Learning"
            keyword_bullets.append({
                "bullet_index": i if i < 6 else i + 1,
                "block_index": block,
                "primary_keyword": kw,
                "secondary_keywords": [],
                "theme": f"Demonstrate {kw} expertise",
                "source": "keyword",
            })
        ideal_bullets = []
        for i, theme in enumerate(bullet_themes[:4]):
            ideal_bullets.append({
                "bullet_index": [2, 5, 8, 11][i] if i < 4 else 11,
                "block_index": i,
                "theme": theme,
                "implicit_requirement": top_3[i] if i < len(top_3) else "",
                "supporting_keywords": [],
                "source": "ideal_candidate",
            })
        return {
            "keyword_bullets": keyword_bullets,
            "ideal_candidate_bullets": ideal_bullets,
            "importance_ranking": list(range(12)),
            "keyword_usage_map": {},
        }


# ============================================================
# 🔒 LaTeX-safe utils
# ============================================================

LATEX_ESC = {
    "#": r"\#", "%": r"\%", "$": r"\$", "&": r"\&",
    "_": r"\_", "{": r"\{", "}": r"\}",
}

UNICODE_NORM = {
    "–": "-", "—": "-", "−": "-", "•": "-", "·": "-", "●": "-",
    "→": "->", "⇒": "=>", "↔": "<->", "×": "x", "°": " degrees ",
    "\u00A0": " ", "\uf0b7": "-", "\x95": "-",
}


def latex_escape_text(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    for a, b in UNICODE_NORM.items():
        s = s.replace(a, b)
    specials = ["%", "$", "&", "_", "#", "{", "}"]
    for ch in specials:
        s = re.sub(rf"(?<!\\){re.escape(ch)}", LATEX_ESC[ch], s)
    s = re.sub(r"(?<!\\)\^", r"\^{}", s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    s = re.sub(r"\\(?![a-zA-Z#$%&_{}^])", "", s)
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
# 📏 BULLET LENGTH VALIDATION
# ============================================================

MIN_BULLET_WORDS = 18
MAX_BULLET_WORDS = 24
IDEAL_BULLET_WORDS = 21


def get_word_count(text: str) -> int:
    return len((text or "").split())


def is_valid_bullet_length(text: str) -> bool:
    count = get_word_count(text)
    return MIN_BULLET_WORDS <= count <= MAX_BULLET_WORDS


def adjust_bullet_length(text: str) -> str:
    words = (text or "").split()
    if len(words) > MAX_BULLET_WORDS:
        truncated = words[:MAX_BULLET_WORDS]
        result = " ".join(truncated).rstrip(".,;:") + "."
        return result
    return text


# ============================================================
# 🧰 LaTeX Parsing Utils
# ============================================================

def find_resume_items(block: str) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    i = 0
    macro = r"\resumeItem"
    n = len(macro)
    while True:
        i = block.find(macro, i)
        if i < 0:
            break
        j = i + n
        while j < len(block) and block[j].isspace():
            j += 1
        if j >= len(block) or block[j] != "{":
            i = j
            continue
        open_b = j
        depth, k = 0, open_b
        while k < len(block):
            if block[k] == "{":
                depth += 1
            elif block[k] == "}":
                depth -= 1
                if depth == 0:
                    out.append((i, open_b, k, k + 1))
                    i = k + 1
                    break
            k += 1
        else:
            break
    return out


def replace_resume_items(block: str, replacements: List[str]) -> str:
    items = find_resume_items(block)
    if not items:
        return block
    if len(replacements) < len(items):
        replacements = replacements + [None] * (len(items) - len(replacements))
    out: List[str] = []
    last = 0
    for (start, open_b, close_b, end), newtxt in zip(items, replacements):
        out.append(block[last:open_b + 1])
        out.append(newtxt if newtxt is not None else block[open_b + 1:close_b])
        out.append(block[close_b:end])
        last = end
    out.append(block[last:])
    return "".join(out)


def section_rx(name: str) -> re.Pattern:
    return re.compile(
        rf"(\\section\*?\{{\s*{re.escape(name)}\s*\}}[\s\S]*?)(?=\\section\*?\{{|\\end\{{document\}}|$)",
        re.IGNORECASE,
    )


# ============================================================
# 🧠 JD Analysis — UPGRADED
# ============================================================

async def extract_company_role(jd_text: str) -> Tuple[str, str]:
    prompt = (
        'Return STRICT JSON: {"company":"…","role":"…"}\n'
        "Use the official company short name and the exact job title.\n"
        f"JD:\n{jd_text}"
    )
    try:
        data = await gpt_json(prompt, temperature=0.0)
        return data.get("company", "Company"), data.get("role", "Role")
    except Exception as e:
        log_event(f"⚠️ [JD PARSE] Failed: {e}")
        return "Company", "Role"


async def extract_keywords_with_priority(jd_text: str) -> Dict[str, Any]:
    prompt = f"""Analyze this job description and extract ALL technical keywords with correct capitalization.

JOB DESCRIPTION:
{jd_text}

Return STRICT JSON:
{{
    "must_have": ["Python","PyTorch","SQL","Machine Learning"],
    "should_have": ["Docker","AWS","Kubernetes","MLOps"],
    "nice_to_have": ["Git","Linux","Agile"],
    "key_responsibilities": ["5-7 main job duties"],
    "domain_context": "brief domain description"
}}

IMPORTANT:
- Extract ONLY hard technical skills (programming languages, frameworks, tools, platforms)
- DO NOT include: ISO standards, NIST frameworks, certifications, compliance terms
- DO NOT include: degrees (PhD, MS), time periods, generic qualifiers
"""
    try:
        data = await gpt_json(prompt, temperature=0.0)

        # Fix capitalization via GPT batch
        must_raw = [str(k).strip() for k in data.get("must_have", []) if str(k).strip()]
        should_raw = [str(k).strip() for k in data.get("should_have", []) if str(k).strip()]
        nice_raw = [str(k).strip() for k in data.get("nice_to_have", []) if str(k).strip()]

        all_raw = must_raw + should_raw + nice_raw
        if all_raw:
            all_fixed = await fix_capitalization_batch(all_raw)
            idx = 0
            must_have = all_fixed[idx:idx + len(must_raw)]; idx += len(must_raw)
            should_have = all_fixed[idx:idx + len(should_raw)]; idx += len(should_raw)
            nice_to_have = all_fixed[idx:idx + len(nice_raw)]
        else:
            must_have, should_have, nice_to_have = [], [], []

        responsibilities = list(data.get("key_responsibilities", []))
        domain = data.get("domain_context", "Technology")

        seen: Set[str] = set()

        def dedup(lst: List[str]) -> List[str]:
            out: List[str] = []
            for item in lst:
                item = str(item).strip()
                if item and item.lower() not in seen:
                    seen.add(item.lower())
                    out.append(item)
            return out

        must_have = dedup(must_have)
        should_have = dedup(should_have)
        nice_to_have = dedup(nice_to_have)
        all_keywords = must_have + should_have + nice_to_have

        log_event(f"💡 [JD KEYWORDS] must={len(must_have)}, should={len(should_have)}, nice={len(nice_to_have)}")
        return {
            "must_have": must_have, "should_have": should_have, "nice_to_have": nice_to_have,
            "all_keywords": all_keywords, "responsibilities": responsibilities, "domain": domain,
        }
    except Exception as e:
        log_event(f"⚠️ [JD KEYWORDS] Failed: {e}")
        return {
            "must_have": [], "should_have": [], "nice_to_have": [],
            "all_keywords": [], "responsibilities": [], "domain": "Technology",
        }


async def extract_coursework_gpt(jd_text: str, max_courses: int = 24) -> List[str]:
    prompt = (
        f"From the JD, choose up to {max_courses} highly relevant university courses. "
        'Return STRICT JSON: {"courses":["Machine Learning","Deep Learning","Data Structures"]}\n'
        f"JD:\n{jd_text}"
    )
    try:
        data = await gpt_json(prompt, temperature=0.0)
        courses = data.get("courses", []) or []
        if courses:
            courses = await fix_capitalization_batch([str(c).strip() for c in courses if str(c).strip()])
        out: List[str] = []
        seen: Set[str] = set()
        for c in courses:
            c = _ensure_first_letter_capital(str(c).strip())
            if c and c.lower() not in seen:
                seen.add(c.lower())
                out.append(c)
        return out[:max_courses]
    except Exception as e:
        log_event(f"⚠️ [JD COURSES] Failed: {e}")
        return []


# ============================================================
# 🎓 Replace Coursework
# ============================================================

def replace_relevant_coursework_distinct(body_tex: str, courses: List[str], max_per_line: int = 6) -> str:
    seen: Set[str] = set()
    uniq: List[str] = []
    for c in courses:
        c = _ensure_first_letter_capital(re.sub(r"\s+", " ", str(c)).strip())
        if c and c.lower() not in seen:
            seen.add(c.lower())
            uniq.append(c)

    line_pat = re.compile(r"(\\item\s*\\textbf\{Relevant Coursework:\})([^\n]*)")
    matches = list(line_pat.finditer(body_tex))
    if not matches:
        return body_tex

    chunks: List[List[str]] = []
    if len(matches) == 1:
        chunks.append(uniq[:max_per_line])
    else:
        split_idx = (len(uniq) + 1) // 2
        chunks = [uniq[:split_idx][:max_per_line], uniq[split_idx:split_idx + max_per_line]]

    out: List[str] = []
    last = 0
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
# 🧱 Skills Section
# ============================================================

def render_skills_section_flat(skills: List[str]) -> str:
    if not skills:
        return ""
    seen: Set[str] = set()
    unique_skills: List[str] = []
    for s in skills:
        s = str(s).strip()
        if not s:
            continue
        if s.lower() not in seen:
            seen.add(s.lower())
            unique_skills.append(s)

    skills_content = ", ".join(latex_escape_text(s) for s in unique_skills)
    return (
        r"\section{Skills}" + "\n"
        r"\begin{itemize}[leftmargin=0.15in, label={}]" + "\n"
        r"  \item \small{" + skills_content + r"}" + "\n"
        r"\end{itemize}"
    )


async def replace_skills_section(body_tex: str, skills: List[str]) -> str:
    new_block = render_skills_section_flat(skills)
    if not new_block:
        return body_tex
    pattern = re.compile(
        r"(\\section\*?\{Skills\}[\s\S]*?)(?=%-----------|\\section\*?\{|\\end\{document\})",
        re.IGNORECASE,
    )
    if re.search(pattern, body_tex):
        return re.sub(pattern, lambda _: new_block + "\n", body_tex)
    m = re.search(r"%-----------TECHNICAL SKILLS-----------", body_tex, re.IGNORECASE)
    if m:
        return body_tex[:m.end()] + "\n" + new_block + "\n" + body_tex[m.end():]
    return body_tex


# ============================================================
# ✨ ENHANCED BULLET GENERATION — 8 keyword + 4 ideal candidate
# ============================================================

# Global keyword dedup tracker: ensures NO keyword appears in more than 1 bullet
_global_keyword_assignments: Dict[str, int] = {}  # keyword_lower -> bullet_index


def reset_keyword_assignment_tracking():
    global _global_keyword_assignments
    _global_keyword_assignments.clear()


async def generate_credible_bullets(
    jd_text: str,
    experience_company: str,
    target_company: str,
    target_role: str,
    company_core_keywords: List[str],
    must_use_keywords: List[str],
    should_use_keywords: List[str],
    responsibilities: List[str],
    used_keywords: Set[str],
    block_index: int,
    bullet_start_position: int,
    total_blocks: int = 4,
    num_bullets: int = 3,
    bullet_plan: Optional[Dict[str, Any]] = None,
    ideal_candidate: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], Set[str]]:
    """
    Generate ENHANCED resume bullets with:
    - ✨ NEW: 8 keyword-sourced + 4 ideal-candidate-sourced across resume
    - ✨ NEW: NO keyword used twice (global dedup)
    - ✨ Unique action verbs (NO repetition across all 12 bullets)
    - ✨ Different sentence structures for each bullet
    - ✨ 4 numbers total (positions 2,6,7,10)
    - ✨ 4 different number categories
    """
    global _global_keyword_assignments

    exp_context = await get_company_context_gpt(experience_company)
    progression = get_progression_context(block_index, total_blocks)

    # Determine which bullets in this block get quantification
    quantified_bullets_in_block = []
    for i in range(num_bullets):
        bullet_pos = bullet_start_position + i
        if should_quantify_bullet(bullet_pos):
            category = get_quantification_category(bullet_pos, jd_text)
            quantified_bullets_in_block.append((i, category))

    # Get verb categories and pre-select unique verbs
    verb_categories = get_verb_categories_for_context(exp_context.get("type", "internship"))
    suggested_verbs = []
    for cat in (verb_categories * 3)[:num_bullets]:
        verb = get_diverse_verb(cat)
        suggested_verbs.append(verb)

    tech_depth = get_technical_depth_phrase("ml_techniques")
    result_phrase = get_result_phrase("performance")
    believability = get_believability_phrase()

    # ✨ NEW: Determine bullet sources from plan
    bullet_sources = []  # list of dicts per bullet in this block
    for local_idx in range(num_bullets):
        abs_idx = bullet_start_position + local_idx
        source_info = {"source": "keyword", "primary_keyword": "", "theme": "", "implicit_requirement": ""}

        if bullet_plan:
            # Check keyword bullets
            for kb in bullet_plan.get("keyword_bullets", []):
                if kb.get("bullet_index") == abs_idx or kb.get("block_index") == block_index:
                    # Match by block index and local position
                    block_kbs = [b for b in bullet_plan["keyword_bullets"] if b.get("block_index") == block_index]
                    kbs_in_block = sorted(block_kbs, key=lambda x: x.get("bullet_index", 99))
                    keyword_local_indices = []
                    for bk in kbs_in_block:
                        if bk.get("source") == "keyword":
                            keyword_local_indices.append(bk)
                    break

            # Check ideal candidate bullets
            for ib in bullet_plan.get("ideal_candidate_bullets", []):
                if ib.get("block_index") == block_index:
                    source_info = {
                        "source": "ideal_candidate",
                        "primary_keyword": "",
                        "theme": ib.get("theme", ""),
                        "implicit_requirement": ib.get("implicit_requirement", ""),
                        "supporting_keywords": ib.get("supporting_keywords", []),
                    }
                    break

        bullet_sources.append(source_info)

    # Build keyword pool — ONLY keywords not yet assigned to another bullet
    available_must = [k for k in must_use_keywords
                      if k.lower() not in _global_keyword_assignments and k.lower() not in used_keywords]
    available_should = [k for k in should_use_keywords
                        if k.lower() not in _global_keyword_assignments and k.lower() not in used_keywords]
    core_pool = [k for k in (company_core_keywords or [])
                 if k.lower() not in _global_keyword_assignments and k.lower() not in used_keywords]

    keywords_for_block = core_pool[:3] + available_must[:6] + available_should[:4]
    keywords_for_block = [k for k in keywords_for_block if k]

    # Fix capitalization
    if keywords_for_block:
        keywords_for_block = await fix_capitalization_batch(keywords_for_block)

    keywords_str = ", ".join(keywords_for_block[:10]) if keywords_for_block else "Python, Machine Learning"
    resp_str = "; ".join(responsibilities[:3]) if responsibilities else "Model Development; Evaluation; Deployment"

    core_focus_str = ", ".join(core_pool[:4]) if core_pool else ""
    core_rule = f"- Naturally include target-company core areas: {core_focus_str}\n" if core_focus_str else ""

    tech_vocab = exp_context.get("technical_vocabulary", [])
    vocab_str = ", ".join(tech_vocab[:5]) if tech_vocab else ""

    # Build quantification instructions
    quant_instructions = []
    for local_idx, category in quantified_bullets_in_block:
        is_hero = (bullet_start_position + local_idx) in HERO_POSITIONS
        if is_hero:
            quant_instructions.append(
                f"   • Bullet {local_idx + 1}: HERO POINT with comparison (from X% to Y%)")
        else:
            if category == "count_scale":
                quant_instructions.append(
                    f"   • Bullet {local_idx + 1}: Include COUNT metric (e.g., '10,347 samples')")
            elif category == "metric_achievement":
                quant_instructions.append(
                    f"   • Bullet {local_idx + 1}: Include ML METRIC (e.g., 'F1 score of 0.87')")
            elif category == "percent_improvement":
                quant_instructions.append(
                    f"   • Bullet {local_idx + 1}: Include PERCENTAGE improvement (e.g., '23.7%')")

    quant_instruction = ""
    if quant_instructions:
        quant_instruction = f"""
🎯 QUANTIFICATION REQUIREMENTS:
{chr(10).join(quant_instructions)}
   • Other bullets: NO numbers
   • Numbers must end in ODD digits (e.g., 23.7%, 10,347, 0.87)
"""
    else:
        quant_instruction = "🎯 NO quantification for this block. Focus on methodology and qualitative impact.\n"

    # ✨ NEW: Build ideal candidate bullet instructions
    ideal_bullet_instructions = ""
    ideal_themes_in_block = []
    if ideal_candidate and bullet_plan:
        for ib in bullet_plan.get("ideal_candidate_bullets", []):
            if ib.get("block_index") == block_index:
                ideal_themes_in_block.append(ib)

    if ideal_themes_in_block:
        themes_text = "\n".join([
            f"   • One bullet MUST address: \"{t.get('theme', '')}\""
            f"\n     (Implicit requirement: {t.get('implicit_requirement', 'N/A')})"
            for t in ideal_themes_in_block
        ])
        supporting_kws = []
        for t in ideal_themes_in_block:
            supporting_kws.extend(t.get("supporting_keywords", []))
        skw_str = ", ".join(supporting_kws[:4]) if supporting_kws else ""

        ideal_bullet_instructions = f"""
🌟 IDEAL CANDIDATE BULLET (1 of 3 bullets in this block):
{themes_text}
   • Weave in these supporting keywords naturally: {skw_str}
   • This bullet shows IMPLICIT value, not just keyword matching
"""

    # ✨ NEW: Keyword dedup instruction
    already_used = list(_global_keyword_assignments.keys())
    dedup_instruction = ""
    if already_used:
        dedup_instruction = f"""
🚫 KEYWORD DEDUP — These keywords are ALREADY USED in other bullets (DO NOT use as primary):
{', '.join(already_used[:20])}
"""

    prompt = f"""Write EXACTLY {num_bullets} HIGHLY CREDIBLE resume bullet points for an INTERN at "{experience_company}",
tailored for applying to "{target_company}" ({target_role}).

🎯 CRITICAL REQUIREMENTS:
0. USE: INDIAN ACCENT IN LANGUAGE and USA SPELLING STANDARDS THROUGHOUT.

1. LENGTH: Each bullet MUST be EXACTLY 18-22 words

2. SKILLS TO USE: Each bullet MUST naturally integrate 2-3 skills from: {keywords_str}
   ⚠️ Each keyword should be PRIMARY in only ONE bullet across the ENTIRE resume

3. ✨ ACTION VERBS (NO substitution, NO repetition):
   - Bullet 1: {suggested_verbs[0]}
   - Bullet 2: {suggested_verbs[1]}
   - Bullet 3: {suggested_verbs[2]}

4. ✨ SENTENCE STRUCTURES: Use DIFFERENT structures for each bullet

5. TECHNICAL DEPTH: Show HOW, not just WHAT:
   - Example technique: "{tech_depth}"

{quant_instruction}

{ideal_bullet_instructions}

{dedup_instruction}

6. BELIEVABILITY FOR INTERN LEVEL:
   - Scope: {progression['scope'][0]} / {progression['scope'][1]} level work
   - Autonomy: {progression['autonomy']}
   - Complexity: {progression['complexity']}
   {f'- Collaboration: {believability}' if believability else ''}

7. DOMAIN VOCABULARY: {vocab_str}
   Company domain: {exp_context['domain']}

{core_rule}

JOB RESPONSIBILITIES TO ALIGN WITH:
{resp_str}

Return STRICT JSON with EXACTLY {num_bullets} bullets:
{{"bullets": ["bullet1", "bullet2", "bullet3"], "primary_keywords_used": ["kw1", "kw2", "kw3"]}}
"""

    try:
        data = await gpt_json(prompt, temperature=0.3)
        bullets = data.get("bullets", []) or []
        primary_kws = data.get("primary_keywords_used", []) or []

        cleaned: List[str] = []
        newly_used: Set[str] = set()

        for local_idx, b in enumerate(bullets[:num_bullets]):
            b = str(b).strip()
            # Fix capitalization via GPT
            b = await fix_capitalization_gpt(b)

            # Check quantification
            should_have_number = any(idx == local_idx for idx, _ in quantified_bullets_in_block)
            if not should_have_number:
                b = re.sub(r'\d+\.?\d*%', '', b)
                b = re.sub(r'\d+x', '', b)
                b = re.sub(r'\d+,?\d*\s+(?:samples|records|minutes|hours)', '', b)
                b = re.sub(r'(?:F1|f1)\s+score\s+of\s+\d+\.\d+', '', b)
                b = re.sub(r'from\s+\d+%?\s+to\s+\d+%?', '', b)
                b = re.sub(r'\s+', ' ', b).strip()

            b = adjust_bullet_length(b)
            b = latex_escape_text(b)

            if b:
                cleaned.append(b)
                # Track keyword usage
                for kw in keywords_for_block:
                    if kw.lower() in b.lower():
                        newly_used.add(kw.lower())

                # ✨ NEW: Track primary keyword globally
                if local_idx < len(primary_kws):
                    pk = primary_kws[local_idx].lower().strip()
                    if pk and pk not in _global_keyword_assignments:
                        _global_keyword_assignments[pk] = bullet_start_position + local_idx

        # Fallback bullets
        while len(cleaned) < num_bullets:
            idx = len(cleaned)
            verb = suggested_verbs[idx]
            kw1 = fix_skill_capitalization_sync(keywords_for_block[idx % max(1, len(keywords_for_block))]) if keywords_for_block else "Python"
            kw2 = fix_skill_capitalization_sync(keywords_for_block[(idx + 1) % max(1, len(keywords_for_block))]) if len(keywords_for_block) > 1 else "Machine Learning"

            should_have_number = any(i == idx for i, _ in quantified_bullets_in_block)
            if should_have_number:
                category = next(cat for i, cat in quantified_bullets_in_block if i == idx)
                quant_phrase = generate_quantified_phrase(category, jd_text)
                fallback = f"{verb} {kw1}-based analytical workflow with {kw2} integration, {quant_phrase} through systematic optimization."
            else:
                fallback = f"{verb} {kw1}-based analytical workflow with {kw2} integration, enabling systematic evaluation and improved reproducibility."

            cleaned.append(latex_escape_text(fallback))
            newly_used.add(kw1.lower())
            newly_used.add(kw2.lower())

        return cleaned[:num_bullets], newly_used

    except Exception as e:
        log_event(f"⚠️ [BULLETS] Generation failed for {experience_company}: {e}")
        fallbacks = []
        for idx in range(num_bullets):
            verb = suggested_verbs[idx]
            fallback = f"{verb} Machine Learning model development and Data Pipeline implementation supporting research objectives with reliable engineering practices."
            fallbacks.append(latex_escape_text(fallback))
        return fallbacks, set()


async def rewrite_experience_with_skill_alignment(
    tex_content: str,
    jd_text: str,
    jd_info: Dict[str, Any],
    target_company: str,
    target_role: str,
    company_core_keywords: List[str],
    bullet_plan: Optional[Dict[str, Any]] = None,
    ideal_candidate: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Set[str]]:
    """Rewrite all experience bullets using the enhanced plan."""
    # Reset all tracking for new resume
    reset_verb_tracking()
    reset_result_phrase_tracking()
    reset_quantification_tracking()
    reset_keyword_assignment_tracking()

    must_have = jd_info.get("must_have", []) or []
    should_have = jd_info.get("should_have", []) or []
    responsibilities = jd_info.get("responsibilities", []) or []

    exp_used_keywords: Set[str] = set()
    num_blocks = 4
    must_per_block = max(3, len(must_have) // num_blocks + 1)
    should_per_block = max(2, len(should_have) // num_blocks + 1)

    exp_pat = section_rx("Experience")
    out: List[str] = []
    pos = 0
    block_index = 0
    absolute_bullet_position = 0

    # ✨ NEW: Extract experience company names from tex via GPT
    exp_companies = await _extract_experience_companies(tex_content)

    for m in exp_pat.finditer(tex_content):
        out.append(tex_content[pos:m.start()])
        section = m.group(1)

        s_tag, e_tag = r"\resumeItemListStart", r"\resumeItemListEnd"
        rebuilt: List[str] = []
        i = 0

        while True:
            a = section.find(s_tag, i)
            if a < 0:
                rebuilt.append(section[i:])
                break

            b = section.find(e_tag, a)
            if b < 0:
                rebuilt.append(section[i:])
                break

            rebuilt.append(section[i:a])

            # Use extracted company name or fallback
            if block_index < len(exp_companies):
                exp_company = exp_companies[block_index]
            else:
                exp_company = f"Company {block_index + 1}"

            start_must = block_index * must_per_block
            end_must = min(start_must + must_per_block, len(must_have))
            block_must = must_have[start_must:end_must]

            unused_must = [k for k in must_have if k.lower() not in exp_used_keywords]
            block_must = list(dict.fromkeys(block_must + unused_must[:2]))

            start_should = block_index * should_per_block
            end_should = min(start_should + should_per_block, len(should_have))
            block_should = should_have[start_should:end_should]

            core_slice = company_core_keywords[(block_index * 2):(block_index * 2 + 3)] if company_core_keywords else []
            block_should = list(dict.fromkeys(block_should + core_slice))

            new_bullets, newly_used = await generate_credible_bullets(
                jd_text=jd_text,
                experience_company=exp_company,
                target_company=target_company,
                target_role=target_role,
                company_core_keywords=company_core_keywords,
                must_use_keywords=block_must,
                should_use_keywords=block_should,
                responsibilities=responsibilities,
                used_keywords=exp_used_keywords,
                block_index=block_index,
                bullet_start_position=absolute_bullet_position,
                total_blocks=num_blocks,
                num_bullets=3,
                bullet_plan=bullet_plan,
                ideal_candidate=ideal_candidate,
            )

            exp_used_keywords.update(newly_used)

            new_block = s_tag + "\n"
            for bullet in new_bullets:
                new_block += f"    \\resumeItem{{{bullet}}}\n"
            new_block += "  " + e_tag

            rebuilt.append(new_block)
            block_index += 1
            absolute_bullet_position += 3
            i = b + len(e_tag)

        out.append("".join(rebuilt))
        pos = m.end()

    out.append(tex_content[pos:])

    must_covered = len([k for k in must_have if k.lower() in exp_used_keywords])
    log_event(f"📊 [EXP COVERAGE] Must-have: {must_covered}/{len(must_have)}")
    log_event(f"🎲 [QUANTIFICATION] Positions: {QUANTIFIED_POSITIONS}, Hero: {HERO_POSITIONS}")
    log_event(f"✅ [VERBS] Total unique: {len(_used_verbs_global)}/12")
    log_event(f"🔑 [KEYWORD DEDUP] Unique primary keywords: {len(_global_keyword_assignments)}")

    return "".join(out), exp_used_keywords


async def _extract_experience_companies(tex_content: str) -> List[str]:
    """Extract company names from the experience section of the TeX."""
    exp_pat = section_rx("Experience")
    m = exp_pat.search(tex_content)
    if not m:
        return []

    section = m.group(1)
    # Try to find company names from \resumeSubheading or similar
    # Pattern: \resumeSubheading{Title}{Dates}{Company}{Location}
    companies = re.findall(r"\\resumeSubheading\{[^}]*\}\{[^}]*\}\{([^}]*)\}", section)
    if not companies:
        # Try alternate pattern: company might be in different position
        companies = re.findall(r"\\resumeSubheading\{([^}]*)\}", section)

    if companies:
        # Clean up
        cleaned = []
        for c in companies:
            c = strip_all_macros_keep_text(c).strip()
            if c and len(c) > 2:
                cleaned.append(c)
        return cleaned

    return []


# ============================================================
# 📄 PDF Helpers
# ============================================================

def _pdf_page_count(pdf_bytes: Optional[bytes]) -> int:
    if not pdf_bytes:
        return 0
    return len(re.findall(rb"/Type\s*/Page\b", pdf_bytes))


_EDU_SPLIT_ANCHOR = re.compile(
    r"(%-----------EDUCATION-----------)|\\section\*?\{\s*Education\s*\}", re.IGNORECASE
)


def _split_preamble_body(tex: str) -> Tuple[str, str]:
    m = _EDU_SPLIT_ANCHOR.search(tex or "")
    if not m:
        return "", re.sub(r"\\end\{document\}\s*$", "", tex or "")
    start = m.start()
    preamble = (tex or "")[:start]
    body = re.sub(r"\\end\{document\}\s*$", "", (tex or "")[start:])
    return preamble, body


def _merge_tex(preamble: str, body: str) -> str:
    out = (str(preamble).strip() + "\n\n" + str(body).strip()).rstrip()
    out = re.sub(r"\\end\{document\}\s*$", "", out).rstrip()
    out += "\n\\end{document}\n"
    return out


# ============================================================
# ✂️ Page Trimming
# ============================================================

ACHIEVEMENT_SECTION_NAMES = [
    "Achievements", "Awards & Achievements", "Achievements & Awards",
    "Awards", "Honors & Awards", "Honors", "Certifications",
]


def remove_one_achievement_bullet(tex_content: str) -> Tuple[str, bool]:
    for sec in ACHIEVEMENT_SECTION_NAMES:
        pat = section_rx(sec)
        m = pat.search(tex_content)
        if not m:
            continue
        full = m.group(1)
        items = find_resume_items(full)
        if items:
            s, _, _, e = items[-1]
            new_sec = full[:s] + full[e:]
            log_event(f"✂️ [TRIM] Removed bullet from '{sec}'")
            return tex_content[:m.start()] + new_sec + tex_content[m.end():], True
    return tex_content, False


def remove_last_bullet_from_sections(
    tex_content: str, sections: Tuple[str, ...] = ("Projects", "Experience")
) -> Tuple[str, bool]:
    for sec in sections:
        pat = section_rx(sec)
        last_m = None
        for match in pat.finditer(tex_content):
            last_m = match
        if last_m:
            full = last_m.group(1)
            items = find_resume_items(full)
            if items:
                s, _, _, e = items[-1]
                new_sec = full[:s] + full[e:]
                log_event(f"✂️ [TRIM] Removed bullet from '{sec}'")
                return tex_content[:last_m.start()] + new_sec + tex_content[last_m.end():], True
    return tex_content, False


# ============================================================
# 📊 Coverage Calculation
# ============================================================

def compute_coverage(tex_content: str, keywords: List[str]) -> Dict[str, Any]:
    plain = strip_all_macros_keep_text(tex_content).lower()
    present: Set[str] = set()
    missing: Set[str] = set()
    for kw in keywords:
        kw_lower = str(kw).lower().strip()
        if kw_lower and kw_lower in plain:
            present.add(kw_lower)
        elif kw_lower:
            missing.add(kw_lower)
    total = max(1, len(present) + len(missing))
    return {
        "ratio": len(present) / total,
        "present": sorted(present),
        "missing": sorted(missing),
        "total": total,
    }


# ============================================================
# 🚀 Main Optimizer — v3.0 with Ideal Candidate Profiling
# ============================================================

async def optimize_resume(
    base_tex: str,
    jd_text: str,
    target_company: str,
    target_role: str,
    extra_keywords: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    log_event("🟨 [OPTIMIZE] Starting v3.0 with IDEAL CANDIDATE PROFILING & keyword dedup")

    # 1) JD keywords
    jd_info = await extract_keywords_with_priority(jd_text)

    # 2) Company-core expectations
    company_core = await extract_company_core_requirements(target_company, target_role, jd_text)
    core_keywords_raw = company_core.get("core_keywords", []) or []
    core_keywords = await fix_capitalization_batch([str(k).strip() for k in core_keywords_raw if str(k).strip()])

    # 3) ✨ NEW: IDEAL CANDIDATE PROFILING
    log_event("🌟 [IDEAL CANDIDATE] Profiling ideal candidate...")
    ideal_candidate = await profile_ideal_candidate(jd_text, target_company, target_role)
    log_event(f"🌟 [IDEAL CANDIDATE] Top 3 must-haves: {ideal_candidate.get('top_3_must_haves', [])}")

    # 4) VALIDATE SKILLS
    log_event("🔍 [SKILL VALIDATION] Starting STRICT validation...")
    all_keywords_raw = list(jd_info.get("all_keywords", []) or [])
    for k in core_keywords:
        if k and k.lower() not in [x.lower() for x in all_keywords_raw]:
            all_keywords_raw.append(k)

    validated_keywords = await filter_valid_skills(all_keywords_raw)
    jd_info["must_have"] = await filter_valid_skills(jd_info.get("must_have", []))
    jd_info["should_have"] = await filter_valid_skills(jd_info.get("should_have", []))
    jd_info["nice_to_have"] = await filter_valid_skills(jd_info.get("nice_to_have", []))
    jd_info["all_keywords"] = validated_keywords
    core_keywords = await filter_valid_skills(core_keywords)
    jd_info["company_core_keywords"] = core_keywords
    all_keywords = validated_keywords

    # 5) Extra keywords
    extra_list: List[str] = []
    if extra_keywords:
        for token in re.split(r"[,\n;]+", extra_keywords):
            t = token.strip()
            if t and t.lower() not in [x.lower() for x in extra_list]:
                extra_list.append(t)
        extra_list = await filter_valid_skills(extra_list)
        if extra_list:
            extra_list = await fix_capitalization_batch(extra_list)

    if extra_list:
        jd_info["extra_keywords"] = extra_list
        for k in extra_list:
            if k.lower() not in [x.lower() for x in all_keywords]:
                all_keywords.append(k)
    else:
        jd_info["extra_keywords"] = []

    log_event(f"📊 [KEYWORDS] JD={len(jd_info.get('all_keywords', []))} + CORE={len(core_keywords)} + EXTRA={len(extra_list)} → TOTAL={len(all_keywords)}")

    # 6) ✨ NEW: RANK ALL BULLET POINTS — 8 keyword + 4 ideal candidate
    log_event("📋 [BULLET RANKING] Planning 12 bullets (8 keyword + 4 ideal candidate)...")
    bullet_plan = await rank_all_bullet_points(
        jd_text=jd_text,
        target_company=target_company,
        target_role=target_role,
        jd_keywords=all_keywords,
        ideal_candidate=ideal_candidate,
    )
    log_event(f"📋 [BULLET PLAN] keyword_bullets={len(bullet_plan.get('keyword_bullets', []))}, "
              f"ideal_bullets={len(bullet_plan.get('ideal_candidate_bullets', []))}")

    # 7) Coursework
    courses = await extract_coursework_gpt(jd_text, max_courses=24)

    # 8) Split preamble/body
    preamble, body = _split_preamble_body(base_tex)

    # 9) Coursework replace
    body = replace_relevant_coursework_distinct(body, courses, max_per_line=8)
    log_event("✅ [COURSEWORK] Updated")

    # 10) ✨ ENHANCED: Rewrite experience with bullet plan + ideal candidate
    body, exp_used_keywords = await rewrite_experience_with_skill_alignment(
        body, jd_text, jd_info,
        target_company=target_company,
        target_role=target_role,
        company_core_keywords=core_keywords,
        bullet_plan=bullet_plan,
        ideal_candidate=ideal_candidate,
    )
    log_event(f"✅ [EXPERIENCE] {len(exp_used_keywords)} keywords used, "
              f"{len(_global_keyword_assignments)} unique primary keywords")

    # 11) Skills section — VALIDATED ONLY
    skills_list: List[str] = []
    # Add from experience
    for kw in exp_used_keywords:
        fixed = fix_skill_capitalization_sync(kw)
        if fixed and fixed.lower() not in [s.lower() for s in skills_list]:
            skills_list.append(fixed)

    for kw in jd_info.get("must_have", []) or []:
        if kw and kw.lower() not in [s.lower() for s in skills_list]:
            skills_list.append(kw)

    for kw in jd_info.get("nice_to_have", []) or []:
        if kw and kw.lower() not in [s.lower() for s in skills_list]:
            skills_list.append(kw)

    for kw in core_keywords:
        if kw and kw.lower() not in [s.lower() for s in skills_list]:
            skills_list.append(kw)

    for kw in extra_list:
        if kw and kw.lower() not in [s.lower() for s in skills_list]:
            skills_list.append(kw)

    skills_list = await filter_valid_skills(skills_list)
    if skills_list:
        skills_list = await fix_capitalization_batch(skills_list)

    body = await replace_skills_section(body, skills_list)
    log_event(f"✅ [SKILLS] {len(skills_list)} validated skills")

    # 12) Merge back
    final_tex = _merge_tex(preamble, body)

    # 13) Coverage
    coverage = compute_coverage(final_tex, all_keywords)
    log_event(f"📊 [COVERAGE] {coverage['ratio']:.1%}")

    all_numbers_used = (
        list(_used_numbers_by_category['percent']) +
        list(_used_numbers_by_category['count']) +
        list(_used_numbers_by_category['metric']) +
        list(_used_numbers_by_category['comparison'])
    )
    log_event(f"🎲 [UNIQUE NUMBERS] Used: {all_numbers_used}")

    return final_tex, {
        "jd_info": jd_info,
        "company_core": company_core,
        "ideal_candidate": ideal_candidate,
        "bullet_plan": bullet_plan,
        "all_keywords": all_keywords,
        "coverage": coverage,
        "exp_used_keywords": list(exp_used_keywords),
        "skills_list": skills_list,
        "unique_numbers_used": all_numbers_used,
        "global_keyword_assignments": dict(_global_keyword_assignments),
    }


# ============================================================
# 🚀 API Endpoint
# ============================================================

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
        _ = use_humanize  # Ignored

        jd_text = (jd_text or "").strip()
        if not jd_text:
            raise HTTPException(status_code=400, detail="jd_text is required.")

        # Load base resume
        raw_tex = ""
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
                raise HTTPException(status_code=500, detail="Default base resume not found")
            raw_tex = default_path.read_text(encoding="utf-8")
            log_event(f"📄 Using default base: {default_path}")

        target_company, target_role = await extract_company_role(jd_text)

        optimized_tex, info = await optimize_resume(
            raw_tex, jd_text,
            target_company=target_company,
            target_role=target_role,
            extra_keywords=extra_keywords,
        )

        safe_company = safe_filename(target_company)
        safe_role = safe_filename(target_role)

        # Compile OPTIMIZED PDF
        cur_tex = optimized_tex
        final_tex = render_final_tex(cur_tex)

        try:
            pdf_bytes_optimized = compile_latex_safely(final_tex)
            if not pdf_bytes_optimized:
                debug_path = Path(f"/tmp/debug_failed_{safe_company}_{safe_role}.tex")
                debug_path.write_text(final_tex, encoding="utf-8")
                raise HTTPException(status_code=500, detail="LaTeX compilation failed.")
        except HTTPException:
            raise
        except Exception as e:
            debug_path = Path(f"/tmp/debug_failed_{safe_company}_{safe_role}.tex")
            debug_path.write_text(final_tex, encoding="utf-8")
            raise HTTPException(status_code=500, detail=f"LaTeX compilation failed: {str(e)}")

        # Trim if needed
        cur_pdf_bytes = pdf_bytes_optimized
        pages = _pdf_page_count(cur_pdf_bytes)
        trim_count = 0
        MAX_TRIMS = 50

        while pages > 1 and trim_count < MAX_TRIMS:
            cur_tex, removed = remove_one_achievement_bullet(cur_tex)
            if not removed:
                cur_tex, removed = remove_last_bullet_from_sections(cur_tex, ("Projects", "Experience"))
            if not removed:
                break
            trim_count += 1
            final_tex = render_final_tex(cur_tex)
            cur_pdf_bytes = compile_latex_safely(final_tex)
            pages = _pdf_page_count(cur_pdf_bytes)

        optimized_tex_final = cur_tex
        pdf_bytes_optimized = cur_pdf_bytes
        coverage = info["coverage"]

        # Save files
        paths = build_output_paths(target_company, target_role)
        opt_path = paths["optimized"]
        saved_paths: List[str] = []

        if pdf_bytes_optimized:
            opt_path.parent.mkdir(parents=True, exist_ok=True)
            opt_path.write_bytes(pdf_bytes_optimized)
            saved_paths.append(str(opt_path))
            log_event(f"💾 [SAVE] Optimized → {opt_path}")

        # ✨ UPGRADED: Response includes ideal candidate + bullet plan
        ideal_candidate_info = info.get("ideal_candidate", {})
        bullet_plan_info = info.get("bullet_plan", {})

        return JSONResponse({
            "company_name": target_company,
            "role": target_role,
            "eligibility": {
                "score": coverage["ratio"],
                "present": coverage["present"],
                "missing": coverage["missing"],
                "total": coverage["total"],
                "verdict": (
                    "Strong fit" if coverage["ratio"] >= 0.7
                    else "Good fit" if coverage["ratio"] >= 0.5
                    else "Needs improvement"
                ),
            },
            "company_core": info.get("company_core", {}),
            # ✨ NEW: Ideal candidate analysis
            "ideal_candidate": {
                "profile_summary": ideal_candidate_info.get("ideal_profile_summary", ""),
                "top_3_must_haves": ideal_candidate_info.get("top_3_must_haves", []),
                "implicit_requirements": ideal_candidate_info.get("implicit_requirements", []),
                "differentiation_factors": ideal_candidate_info.get("differentiation_factors", []),
                "bullet_themes_used": ideal_candidate_info.get("ideal_candidate_bullet_themes", []),
            },
            # ✨ NEW: Bullet plan transparency
            "bullet_plan": {
                "keyword_bullets_count": len(bullet_plan_info.get("keyword_bullets", [])),
                "ideal_candidate_bullets_count": len(bullet_plan_info.get("ideal_candidate_bullets", [])),
                "importance_ranking": bullet_plan_info.get("importance_ranking", []),
                "keyword_dedup_map": info.get("global_keyword_assignments", {}),
            },
            "optimized": {
                "tex": render_final_tex(optimized_tex_final),
                "pdf_b64": base64.b64encode(pdf_bytes_optimized or b"").decode("ascii"),
                "filename": str(opt_path) if pdf_bytes_optimized else "",
            },
            "humanized": {"tex": "", "pdf_b64": "", "filename": ""},
            "tex_string": render_final_tex(optimized_tex_final),
            "pdf_base64": base64.b64encode(pdf_bytes_optimized or b"").decode("ascii"),
            "pdf_base64_humanized": None,
            "saved_paths": saved_paths,
            "coverage_ratio": coverage["ratio"],
            "coverage_present": coverage["present"],
            "coverage_missing": coverage["missing"],
            "coverage_history": [],
            "did_humanize": False,
            "extra_keywords": info.get("jd_info", {}).get("extra_keywords", []),
            "skills_list": info.get("skills_list", []),
            "unique_numbers_used": info.get("unique_numbers_used", []),
        })

    except Exception as e:
        log_event(f"💥 [PIPELINE] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))