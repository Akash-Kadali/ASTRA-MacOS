"""
Resume optimizer API (FastAPI) — v2.0.0

CHANGES vs v1.0.0:
 FEAT  Added Projects section in "\textbf{Title} -- one-liner" format, placed after Experience and before Achievements.
 FEAT  Added placeholder sanitizer to detect tokens like XYZ, ABC, Lorem, Foo, etc., and replace them with contextual terms.
 FEAT  Added PDF metadata injection: pdfauthor=Sri Akash Kadali, pdfkeywords=relevant skills, pdfsubject=coursework.
 FEAT  Added metric diversity enforcement so bullets use a mix of counts, x→y improvements, time, throughput, and not only percentages.

 FIX   Experience trimming now preserves at least 9 bullets (target remains 12 when possible).
 FIX   Projects section now renders 1-2 projects under a single subheading, each as a one-line entry.
 FIX   Skills section now uses a flat comma-separated format with no category sub-headings.
 FIX   Soft skills mentioned in the JD are now included in the Skills section.
 FIX   JD-required skills are now added more completely.
 FIX   Section font sizes reduced to \small to improve one-page fitting.

 BUG   Added isinstance guards in deduplicate_across_blocks to prevent invalid object access.
 BUG   Added isinstance guards in validate_and_fix_task_alignment to prevent invalid object access.
 BUG   Added isinstance guards in score_bullet_quality_rubric to prevent invalid object access.
"""

import base64
import json
import re
import asyncio
import threading
import random as _random
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
    prompt: str, temperature: float = 0.0, model: str = "gpt-5.4-mini",
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


# ═══════════════════════════════════════════════════════════════
# TECHNOLOGY SPECIFICITY MAP
# ═══════════════════════════════════════════════════════════════

TECHNOLOGY_SPECIFICITY_MAP = {
    "llm": ["Llama 3.1", "GPT-4", "Claude Sonnet", "Mistral", "Gemini Pro"],
    "large language model": ["Llama", "GPT-4", "Claude", "Mistral"],
    "language model": ["BERT", "RoBERTa", "T5", "DistilBERT"],
    "transformer": ["BERT", "GPT-2", "T5", "BART", "RoBERTa"],
    "nlp": ["BERT", "RoBERTa", "spaCy", "NLTK", "Sentence-BERT"],
    "natural language processing": ["BERT for token classification", "RoBERTa for sentiment analysis"],
    "text classification": ["BERT with CLS pooling", "RoBERTa", "DistilBERT"],
    "sentiment analysis": ["VADER", "RoBERTa fine-tuned", "TextBlob"],
    "named entity recognition": ["spaCy NER", "BERT-NER with CRF"],
    "agentic ai": ["LangChain with ReAct", "LlamaIndex", "CrewAI"],
    "ai agent": ["LangChain agent", "function-calling with GPT-4"],
    "pytorch": ["PyTorch with CUDA", "PyTorch Lightning", "torch.nn modules"],
    "tensorflow": ["TensorFlow with Keras", "tf.data pipelines"],
    "keras": ["Keras with mixed precision", "custom Keras layers"],
    "scikit-learn": ["scikit-learn Pipeline", "GridSearchCV"],
    "deep learning": ["CNN with ResNet backbone", "LSTM with attention", "Transformer encoder-decoder"],
    "neural network": ["MLP with dropout", "ResNet transfer learning"],
    "cnn": ["ResNet pretrained", "EfficientNet", "custom CNN"],
    "rnn": ["bidirectional LSTM", "GRU with attention"],
    "lstm": ["bidirectional LSTM", "stacked LSTM"],
    "aws": ["AWS SageMaker", "Lambda", "S3", "EC2"],
    "kubernetes": ["Kubernetes with Helm", "KServe"],
    "docker": ["Docker multi-stage builds", "Docker Compose"],
    "cloud": ["AWS SageMaker", "GCP Vertex AI", "Azure ML"],
    "data pipeline": ["Airflow DAGs", "Prefect workflows"],
    "etl": ["Apache Airflow", "dbt", "Spark ETL"],
    "data processing": ["Pandas with Dask", "PySpark"],
    "mlops": ["MLflow tracking", "DVC versioning", "Kubeflow Pipelines"],
    "model deployment": ["FastAPI with Docker", "TorchServe", "SageMaker endpoints"],
    "ci/cd": ["GitHub Actions", "Jenkins pipeline"],
    "monitoring": ["Prometheus with Grafana", "CloudWatch"],
    "database": ["PostgreSQL with pgvector", "MongoDB", "Redis"],
    "sql": ["PostgreSQL", "MySQL", "SQLite"],
    "nosql": ["MongoDB", "DynamoDB", "Cassandra"],
    "vector database": ["Pinecone", "Weaviate", "ChromaDB"],
    "git": ["Git with feature branching", "GitHub PR workflows"],
}

_used_specific_technologies: Set[str] = set()


def reset_technology_tracking():
    global _used_specific_technologies
    _used_specific_technologies.clear()


async def get_specific_technology(
    generic_term: str, context: str = "",
    already_used: Optional[Set[str]] = None, block_index: int = 0,
) -> str:
    global _used_specific_technologies
    if already_used is None:
        already_used = _used_specific_technologies
    generic_lower = generic_term.lower().strip()
    candidates = TECHNOLOGY_SPECIFICITY_MAP.get(generic_lower, [])
    if not candidates:
        return await fix_capitalization_gpt(generic_term)
    available = [c for c in candidates if c.lower() not in already_used] or candidates
    complexity_bias = 0.3 if block_index == 0 else (0.5 if block_index <= 1 else 0.7)

    def _score(tech):
        s = 0.5
        if re.search(r"\d+\.\d+", tech):
            s += 0.3
        if len(tech.split()) > 2:
            s += 0.2
        return s

    available.sort(key=_score)
    idx = max(0, min(int(len(available) * complexity_bias), len(available) - 1))
    chosen = available[idx]
    already_used.add(chosen.lower())
    return chosen


# ═══════════════════════════════════════════════════════════════
# ROLE ARCHETYPE + TONE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

ROLE_ARCHETYPES = {
    "software_engineer": {
        "name": "Software Engineer",
        "bullet_focus": "systems built, scale handled, reliability, performance",
        "phrasing_style": "Designed and built X that handles Y at Z scale",
        "result_types": "latency reduced, uptime improved, throughput increased",
        "typical_verbs": ["Designed", "Built", "Engineered", "Implemented", "Architected"],
        "avoid": "ML-specific jargon unless JD mentions it",
    },
    "data_scientist": {
        "name": "Data Scientist",
        "bullet_focus": "hypotheses tested, insights found, models validated, business impact",
        "phrasing_style": "Identified that X by analyzing Y, leading to Z decision",
        "result_types": "insight discovered, model accuracy, business metric moved",
        "typical_verbs": ["Analyzed", "Discovered", "Validated", "Investigated", "Modeled"],
        "avoid": "pure engineering language, deployment details unless relevant",
    },
    "ml_engineer": {
        "name": "ML Engineer",
        "bullet_focus": "pipelines built, models deployed, latency/throughput, training infra",
        "phrasing_style": "Deployed a real-time scoring service that serves X predictions/sec",
        "result_types": "inference latency, model serving throughput, pipeline reliability",
        "typical_verbs": ["Deployed", "Trained", "Optimized", "Automated", "Built"],
        "avoid": "pure research language, business analysis language",
    },
    "cloud_infrastructure": {
        "name": "Cloud / Infrastructure Engineer",
        "bullet_focus": "infra provisioned, automation, cost optimization, reliability",
        "phrasing_style": "Automated X provisioning with Terraform, reducing Y by Z%",
        "result_types": "cost reduced, provisioning time cut, reliability improved",
        "typical_verbs": ["Automated", "Provisioned", "Configured", "Orchestrated", "Migrated"],
        "avoid": "ML model details, data science language",
    },
    "research": {
        "name": "Research Scientist / Engineer",
        "bullet_focus": "novel methods, ablations, benchmarks, publications",
        "phrasing_style": "Proposed a contrastive objective that improved X on benchmark Y",
        "result_types": "benchmark improved, novel approach validated, ablation completed",
        "typical_verbs": ["Investigated", "Proposed", "Evaluated", "Designed", "Validated"],
        "avoid": "production/deployment language, business metrics",
    },
    "data_engineer": {
        "name": "Data Engineer",
        "bullet_focus": "pipelines built, data quality, throughput, schema design",
        "phrasing_style": "Built a streaming ETL pipeline ingesting X events/sec",
        "result_types": "data freshness, pipeline throughput, data quality",
        "typical_verbs": ["Built", "Designed", "Automated", "Optimized", "Migrated"],
        "avoid": "ML model training details, research language",
    },
    "general_tech": {
        "name": "General Technical Role",
        "bullet_focus": "problems solved, systems built, efficiency gained",
        "phrasing_style": "Built X that solved Y, resulting in Z improvement",
        "result_types": "efficiency gained, problem solved, process improved",
        "typical_verbs": ["Built", "Developed", "Implemented", "Designed", "Improved"],
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
        log_event(f"🎭 [ROLE+TONE] {target_role} → {key}, tone={result['tone_register']}/{result['tone_pace']}")
        return result
    except Exception as e:
        log_event(f"⚠️ [ROLE+TONE] Failed: {e}")
        return {"key": "general_tech", **ROLE_ARCHETYPES["general_tech"],
                "tone_register": "technical", "tone_pace": "measured",
                "tone_vocabulary": "precise", "tone_examples": []}


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

JOB DESCRIPTION:
{jd_text[:3500]}

ROLE ARCHETYPE: {archetype_key} — focuses on: {archetype_focus}

Return STRICT JSON:
{{
    "tasks": [
        {{
            "task_id": 1,
            "task_description": "Concrete task",
            "task_category": "build_system|analyze_data|train_model|deploy_service|build_pipeline|optimize_performance|write_tests|design_architecture|automate_process|collaborate|research|monitor",
            "implied_technologies": ["2-3 specific technologies"],
            "what_good_looks_like": "What a strong resume bullet would say (1 sentence, past tense)",
            "priority": "high|medium|low",
            "key_jd_phrases": ["2-3 exact phrases from the JD that describe this task"]
        }},
        ... (10-12 tasks)
    ],
    "role_summary": "2-3 sentence summary",
    "domain_context": "industry/domain context"
}}

RULES:
- Tasks are WORK ACTIVITIES, not skill names
- key_jd_phrases must be EXACT quotes from the JD text
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
                    "what_good_looks_like": t.get("what_good_looks_like", ""),
                    "priority": t.get("priority", "medium"),
                    "key_jd_phrases": t.get("key_jd_phrases", [])[:3],
                })
        log_event(f"📋 [JD TASKS] {len(tasks)} tasks extracted")
        return tasks
    except Exception as e:
        log_event(f"⚠️ [JD TASKS] Failed: {e}")
        return [{"task_id": i + 1, "task_description": f"Task {i + 1}",
                 "task_category": "build_system", "implied_technologies": ["Python"],
                 "what_good_looks_like": "", "priority": "medium", "key_jd_phrases": []}
                for i in range(6)]


# ═══════════════════════════════════════════════════════════════
# JD KEY PHRASE EXTRACTION
# ═══════════════════════════════════════════════════════════════

async def extract_jd_key_phrases(jd_text: str) -> List[str]:
    prompt = f"""Extract the KEY NOUN PHRASES from this job description that a recruiter
would look for when scanning a resume.

JOB DESCRIPTION:
{jd_text[:3000]}

Return STRICT JSON:
{{
    "key_phrases": ["real-time feature serving", "model training pipeline", ... (15-25 phrases, 2-5 words each, EXACT JD language)]
}}

Rules:
- Extract EXACT phrases as they appear in the JD
- Focus on technical activities and domain concepts, not generic words
- 2-5 words each — not single words, not full sentences
"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        phrases = [str(p).strip().lower() for p in (data.get("key_phrases", []) or []) if str(p).strip()]
        log_event(f"🔤 [JD PHRASES] Extracted {len(phrases)} key phrases")
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

ACTION_VERBS = {
    "development": ["Architected", "Engineered", "Developed", "Built", "Implemented",
                    "Constructed", "Designed", "Created", "Established", "Programmed", "Prototyped"],
    "research": ["Investigated", "Explored", "Analyzed", "Evaluated", "Validated",
                 "Examined", "Studied", "Researched", "Assessed", "Characterized"],
    "optimization": ["Optimized", "Enhanced", "Streamlined", "Accelerated", "Refined",
                     "Improved", "Strengthened", "Elevated", "Augmented"],
    "data_work": ["Processed", "Transformed", "Aggregated", "Curated", "Cleaned",
                  "Structured", "Consolidated", "Standardized", "Synthesized"],
    "ml_training": ["Trained", "Fine-tuned", "Calibrated", "Tuned", "Configured",
                    "Adapted", "Specialized", "Customized"],
    "deployment": ["Deployed", "Launched", "Released", "Shipped", "Delivered",
                   "Productionized", "Integrated", "Provisioned"],
    "analysis": ["Analyzed", "Diagnosed", "Identified", "Discovered", "Uncovered",
                 "Detected", "Profiled", "Mapped", "Quantified"],
    "collaboration": ["Collaborated", "Partnered", "Coordinated", "Facilitated",
                      "Contributed", "Supported", "Engaged"],
    "automation": ["Automated", "Systematized", "Scripted", "Orchestrated",
                   "Scheduled", "Codified"],
    "documentation": ["Documented", "Recorded", "Cataloged", "Annotated",
                      "Specified", "Summarized", "Reported"],
}

TASK_CAT_TO_VERB_CAT = {
    "build_system": "development", "analyze_data": "analysis",
    "train_model": "ml_training", "deploy_service": "deployment",
    "build_pipeline": "development", "optimize_performance": "optimization",
    "write_tests": "development", "design_architecture": "development",
    "automate_process": "automation", "collaborate": "collaboration",
    "research": "research", "monitor": "analysis",
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
        all_v = [v for cat in ACTION_VERBS.values() for v in cat]
        available = [v for v in all_v if v.lower() not in _used_verbs_global]
    chosen = _random.choice(available) if available else fallback
    _used_verbs_global.add(chosen.lower())
    return chosen


# ═══════════════════════════════════════════════════════════════
# ODD-NUMBER METRIC HELPERS
# ═══════════════════════════════════════════════════════════════

def _odd_int(lo: int, hi: int) -> int:
    pool = [x for x in range(lo, hi + 1) if x % 10 not in (0, 5)]
    return _random.choice(pool) if pool else _random.randint(lo, hi)


def _odd_dec(lo: float, hi: float, places: int = 2) -> str:
    for _ in range(40):
        v = round(_random.uniform(lo, hi), places)
        s = f"{v:.{places}f}"
        if s[-1].isdigit() and int(s[-1]) % 2 == 1:
            return s
    return f"{round(_random.uniform(lo, hi), places):.{places}f}"


# ═══════════════════════════════════════════════════════════════
# METRIC TEMPLATES + pick_metric_hint  (v2.5.0: diversified)
# ═══════════════════════════════════════════════════════════════

METRIC_TEMPLATES: Dict[str, List[str]] = {
    "build_system": [
        "reducing build time from {a} min to {b} min",
        "cutting deployment steps from {a} to {b}",
        "supporting {n}+ concurrent services",
        "processing {n}K requests per hour",
        "eliminating {n} hours of manual work per sprint",
        "reducing rollback frequency from {a} times/month to {b}",
        "compiling {n} modules in under {b} seconds",
        "handling {n}K concurrent connections with {b}ms avg response",
    ],
    "analyze_data": [
        "surfacing {n} previously untracked data gaps",
        "reducing analysis turnaround from {a} days to {b} hours",
        "covering {n}K+ records across {m} data sources",
        "cutting dashboard load time from {a}s to {b}s",
        "identifying {n} anomalous patterns flagged for engineering review",
        "reducing data quality errors from {a} per week to {b}",
        "scanning {n}M rows in {b} seconds with {m} validation rules",
        "correlating {n} features across {m} datasets to surface {b} insights",
    ],
    "train_model": [
        "improving F1 from 0.{a_d} to 0.{b_d} on held-out test set",
        "reducing training time from {a} hours to {b} hours on {n}K samples",
        "cutting false-positive rate from {a}% to {b}%",
        "achieving {n}% precision at {m}% recall on imbalanced dataset",
        "training on {n}K labeled examples with {m}-fold cross-validation",
        "reducing per-epoch time from {a} min to {b} min on {n}K-sample dataset",
        "lowering loss from {a}.{b_d} to 0.{a_d} over {n} epochs",
        "converging in {b} epochs vs {a} with previous architecture on {n}K samples",
    ],
    "deploy_service": [
        "serving {n}K+ predictions per day with p99 latency under {m}ms",
        "reducing cold-start latency from {a}ms to {b}ms",
        "handling traffic spikes of {n}x baseline without degradation",
        "cutting rollback time from {a} min to under {b} min",
        "reducing failed deployments from {a} per month to {b}",
        "dropping p95 inference latency from {a}ms to {b}ms",
        "scaling from {b} to {a} replicas in under {m} seconds",
        "maintaining {n}.{b_d}% uptime across {m} microservices",
    ],
    "build_pipeline": [
        "ingesting {n}K events per hour from {m} upstream sources",
        "reducing pipeline failure rate from {a}% to under {b}%",
        "cutting end-to-end data latency from {a} hours to {b} minutes",
        "processing {n}GB of raw logs daily with zero data loss",
        "reducing manual intervention from {a} times/week to {b}",
        "shrinking pipeline run time from {a} hours to {b} minutes",
        "orchestrating {n} DAG tasks across {m} worker nodes",
        "backfilling {n} months of data in {b} hours vs {a} days previously",
    ],
    "optimize_performance": [
        "reducing inference latency from {a}ms to {b}ms ({n}x speedup)",
        "cutting memory footprint by {n}MB",
        "improving throughput from {a}K to {b}K requests per second",
        "halving query execution time from {a}s to {b}s",
        "reducing CPU utilization from {a}% to {b}% under peak load",
        "dropping model load time from {a}s to {b}s",
        "compressing model from {a}MB to {b}MB with negligible accuracy loss",
        "reducing batch processing from {a} min to {b} min for {n}K records",
    ],
    "automate_process": [
        "saving {n} engineer-hours per week",
        "reducing manual steps from {a} to {b} per release cycle",
        "cutting mean time to deploy from {a} hours to {b} minutes",
        "eliminating {n}% of recurring on-call alerts",
        "automating {n} previously manual QA checks per sprint",
        "reducing ticket resolution time from {a} days to {b} hours",
        "replacing {a} manual workflows with {b} automated pipelines",
        "shrinking CI/CD cycle from {a} min to {b} min across {n} repos",
    ],
    "research": [
        "outperforming baseline by {n}.{m} points on {dataset} benchmark",
        "reducing ablation runtime from {a} hours to {b} hours",
        "validating across {n} experimental conditions",
        "testing {n} architectural variants in a single sweep",
        "replicating published results within {b}% on {n} benchmarks",
        "improving BLEU score from 0.{a_d} to 0.{b_d} on held-out test split",
        "evaluating {n} hyperparameter configs in {b} GPU-hours",
        "reducing parameter count from {a}M to {b}M while retaining 0.{b_d} accuracy",
    ],
    "monitor": [
        "reducing mean time to detection from {a} min to {b} min",
        "cutting false alert rate by {n}% over a {m}-week baseline",
        "catching {n} silent data drift incidents before production impact",
        "covering {n} service SLOs with unified alerting",
        "reducing P1 incident response time from {a} min to {b} min",
        "decreasing alert noise from {a} pages/week to {b}",
        "tracking {n} metrics across {m} dashboards with {b}s refresh interval",
    ],
    "collaborate": [
        "aligning {n} cross-functional stakeholders across {m} teams",
        "delivering {n} features across {m}-week sprint cycles",
        "unblocking {n} downstream teams by clarifying data contracts",
        "reducing cross-team integration bugs from {a} per sprint to {b}",
    ],
    "design_architecture": [
        "supporting {n}x projected peak load",
        "reducing inter-service latency from {a}ms to {b}ms",
        "enabling {n} independent teams to deploy without coordination",
        "cutting infrastructure provisioning time from {a} hours to {b} min",
    ],
    "write_tests": [
        "increasing test coverage from {a}% to {b}%",
        "catching {n} regressions before merge in a {m}-week window",
        "reducing post-deploy bug rate by {n} issues/week",
        "shrinking test suite run time from {a} min to {b} min",
    ],
}


# v2.5.0: Track which metric TYPES have been used globally
_used_metric_types: List[str] = []


def reset_metric_type_tracking():
    global _used_metric_types
    _used_metric_types.clear()


def pick_metric_hint(task_category: str) -> str:
    global _used_metric_types
    templates = METRIC_TEMPLATES.get(task_category, METRIC_TEMPLATES["build_system"])
    datasets = ["GLUE", "SQuAD", "BEIR", "MS-MARCO", "MIMIC-III", "CommonCrawl", "WMT-19"]

    # v2.5.0: Prefer templates we haven't used recently
    def _classify_metric_type(tpl: str) -> str:
        if "from {a}" in tpl and "to {b}" in tpl:
            return "x_to_y"
        if "{n}K" in tpl or "{n}M" in tpl or "{n}GB" in tpl:
            return "count"
        if "%" in tpl:
            return "percentage"
        if "min" in tpl or "hours" in tpl or "seconds" in tpl or "ms" in tpl:
            return "time"
        if "x " in tpl or "x baseline" in tpl:
            return "multiplier"
        return "other"

    # Score templates: prefer types not recently used
    scored = []
    for tpl in templates:
        mtype = _classify_metric_type(tpl)
        recent_count = _used_metric_types[-6:].count(mtype) if _used_metric_types else 0
        score = _random.random() - (recent_count * 0.4)
        scored.append((score, tpl, mtype))
    scored.sort(key=lambda x: x[0], reverse=True)
    tpl = scored[0][1]
    mtype = scored[0][2]
    _used_metric_types.append(mtype)

    if task_category in ("deploy_service", "optimize_performance"):
        a = _odd_int(43, 173)
        b = _odd_int(7, max(7, a - 17))
        n = _odd_int(3, 47)
        m = _odd_int(7, 29)
        k = _odd_int(2, 7)
        a_d = _odd_int(61, 77)
        b_d = _odd_int(max(a_d + 3, 78), 93)
    elif task_category in ("build_pipeline", "analyze_data"):
        a = _odd_int(3, 47)
        b = _odd_int(1, max(1, a - 2))
        n = _odd_int(11, 83)
        m = _odd_int(2, 7)
        k = _odd_int(2, 6)
        a_d = _odd_int(61, 77)
        b_d = _odd_int(max(a_d + 3, 78), 93)
    elif task_category == "train_model":
        a_d = _odd_int(61, 77)
        b_d = _odd_int(max(a_d + 3, 78), 93)
        a = a_d
        b = _odd_int(max(1, a_d - 21), max(1, a_d - 7))
        n = _odd_int(3, 47)
        m = _odd_int(3, 8)
        k = _odd_int(2, 6)
    elif task_category in ("automate_process", "build_system", "write_tests"):
        a = _odd_int(7, 43)
        b = _odd_int(2, max(2, a - 3))
        n = _odd_int(3, 23)
        m = _odd_int(2, 9)
        k = _odd_int(2, 6)
        a_d = _odd_int(61, 77)
        b_d = _odd_int(max(a_d + 3, 78), 93)
    elif task_category == "research":
        n = _odd_int(1, 9)
        m = _odd_int(1, 7)
        k = _odd_int(2, 5)
        a = _odd_int(3, 23)
        b = _odd_int(1, max(1, a - 1))
        a_d = _odd_int(61, 77)
        b_d = _odd_int(max(a_d + 3, 78), 93)
    elif task_category == "monitor":
        a = _odd_int(17, 73)
        b = _odd_int(3, max(3, a - 7))
        n = _odd_int(3, 23)
        m = _odd_int(3, 9)
        k = _odd_int(2, 7)
        a_d = _odd_int(61, 77)
        b_d = _odd_int(max(a_d + 3, 78), 93)
    else:
        a = _odd_int(11, 93)
        b = _odd_int(2, max(2, a - 3))
        n = _odd_int(3, 67)
        m = _odd_int(2, 9)
        k = _odd_int(2, 7)
        a_d = _odd_int(61, 77)
        b_d = _odd_int(max(a_d + 3, 78), 93)

    try:
        return tpl.format(n=n, m=m, k=k, a=a, b=b, a_d=a_d, b_d=b_d,
                          dataset=_random.choice(datasets))
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
# PLACEHOLDER WORD SANITIZER — NEW in v2.5.0
# ═══════════════════════════════════════════════════════════════

# Common placeholder words that GPT might generate
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

# Contextual placeholder phrases (more than single word)
_PLACEHOLDER_PHRASES = [
    "xyz company", "abc corporation", "xyz tool", "abc framework",
    "company xyz", "company abc", "tool xyz", "platform xyz",
    "the xyz", "an xyz", "a xyz", "this xyz",
]


async def sanitize_placeholder_words(
    text: str, context: str = "", company: str = "", role: str = "",
) -> str:
    """v2.5.0: Detect and replace placeholder words (XYZ, ABC, Foo, Lorem, etc.)
    with contextually appropriate replacements using GPT."""
    if not text:
        return text

    # Check for placeholder phrases first
    text_lower = text.lower()
    has_placeholder_phrase = any(p in text_lower for p in _PLACEHOLDER_PHRASES)

    # Check for single-word placeholders
    matches = list(_PLACEHOLDER_PATTERNS.finditer(text))

    if not matches and not has_placeholder_phrase:
        return text

    log_event(f"🔧 [SANITIZE] Found {len(matches)} placeholder matches in text")

    prompt = f"""The following text contains PLACEHOLDER WORDS that need to be replaced
with REAL, contextually appropriate words.

TEXT:
"{text[:500]}"

CONTEXT: Resume bullet for {role} role at {company}.
{('ADDITIONAL CONTEXT: ' + context[:300]) if context else ''}

COMMON PLACEHOLDERS to watch for: XYZ, ABC, Foo, Bar, Baz, Lorem, Ipsum,
Acme, Initech, widget, gadget, "some company", "Company A", "Tool X", etc.

Replace EVERY placeholder with a SPECIFIC, REAL term that fits the context:
- Company placeholders → use real context or remove
- Tool placeholders → use a real tool name relevant to the role
- Generic nouns (widget, gadget) → use the actual artifact (API endpoint, data pipeline, microservice, etc.)
- "the system/platform/tool" → name the specific system

Return STRICT JSON: {{"fixed": "the corrected text with all placeholders replaced"}}
"""
    try:
        data = await gpt_json(prompt, temperature=0.2)
        fixed = data.get("fixed", "").strip()
        if fixed and len(fixed) >= len(text) * 0.5:
            # Verify placeholders are actually gone
            remaining = list(_PLACEHOLDER_PATTERNS.finditer(fixed))
            if len(remaining) < len(matches):
                log_event(f"✅ [SANITIZE] Replaced {len(matches) - len(remaining)} placeholders")
                return fixed
        return text
    except Exception as e:
        log_event(f"⚠️ [SANITIZE] Failed: {e}")
        return text


async def sanitize_all_bullets(
    all_bullets: List[List[str]], target_company: str, target_role: str,
    experience_companies: List[str],
) -> List[List[str]]:
    """v2.5.0: Scan all generated bullets for placeholder words and fix them."""
    result = []
    for block_idx, block in enumerate(all_bullets):
        fixed_block = []
        ec = experience_companies[block_idx] if block_idx < len(experience_companies) else "Company"
        for bullet in block:
            if _PLACEHOLDER_PATTERNS.search(bullet):
                fixed = await sanitize_placeholder_words(
                    bullet,
                    context=f"Intern at {ec}",
                    company=target_company,
                    role=target_role,
                )
                fixed_block.append(fixed)
            else:
                fixed_block.append(bullet)
        result.append(fixed_block)
    return result


# ═══════════════════════════════════════════════════════════════
# METRIC DIVERSITY ENFORCER — NEW in v2.5.0
# ═══════════════════════════════════════════════════════════════

def _classify_bullet_metric(bullet: str) -> str:
    """Classify what type of metric a bullet uses."""
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
    """v2.5.0: Ensure bullets use diverse metric types, not just percentages."""
    flat = [b for block in all_bullets for b in block]
    if len(flat) < 6:
        return all_bullets

    # Classify all metrics
    metric_types = [_classify_bullet_metric(b) for b in flat]
    type_counts: Dict[str, int] = {}
    for mt in metric_types:
        type_counts[mt] = type_counts.get(mt, 0) + 1

    pct_count = type_counts.get("percentage", 0)
    total_with_metrics = sum(1 for mt in metric_types if mt != "none")

    # If more than half of metrics are percentages, rewrite some
    if pct_count <= 2 or total_with_metrics < 4:
        log_event(f"✅ [METRIC DIV] Metrics diverse enough: {type_counts}")
        return all_bullets

    log_event(f"⚠️ [METRIC DIV] {pct_count}/{total_with_metrics} metrics are %-based, diversifying...")

    # Find percentage-heavy bullets to rewrite (keep first 2 percentages, rewrite rest)
    pct_indices = [i for i, mt in enumerate(metric_types) if mt == "percentage"]
    to_rewrite = pct_indices[2:]  # Keep at most 2 percentage metrics

    desired_types = ["x_to_y", "time", "count", "multiplier"]
    for ri, idx in enumerate(to_rewrite[:4]):
        plan = bullet_plan[idx] if idx < len(bullet_plan) else {}
        tc = plan.get("task_category", "build_system")
        desired = desired_types[ri % len(desired_types)]

        type_instructions = {
            "x_to_y": "Use a 'from X to Y' format (e.g., 'from 47ms to 13ms', 'from 8 hours to 23 minutes')",
            "time": "Use a time-based metric (e.g., '13ms latency', 'under 7 seconds', 'saving 23 hours/week')",
            "count": "Use a count/volume metric (e.g., '13K requests/day', '47 microservices', '83GB daily')",
            "multiplier": "Use a multiplier metric (e.g., '3x faster', '7x throughput increase')",
        }

        verb = re.sub(r"\\[#$%&_{}]", "",
                      flat[idx].split()[0]) if flat[idx].split() else "Built"
        tech = plan.get("primary_technology", "Python")

        try:
            fix = await gpt_json(
                f'Rewrite this resume bullet replacing the PERCENTAGE metric with a different type.\n'
                f'CURRENT: "{flat[idx][:200]}"\n'
                f'NEW METRIC TYPE: {type_instructions.get(desired, "Use a non-percentage metric")}\n'
                f'Keep starting verb "{verb}". Mention {tech}. 24-34 words. Odd digits only.\n'
                f'Return STRICT JSON: {{"bullet": "..."}}',
                temperature=0.35)
            new_b = fix.get("bullet", "")
            if new_b and len(new_b.split()) >= 15:
                new_mt = _classify_bullet_metric(new_b)
                if new_mt != "percentage":
                    new_b = await fix_capitalization_gpt(new_b)
                    new_b = adjust_bullet_length(new_b)
                    if not new_b.endswith("."):
                        new_b = new_b.rstrip(".,;: ") + "."
                    flat[idx] = latex_escape_text(new_b)
                    log_event(f"✅ [METRIC DIV] idx={idx}: percentage → {new_mt}")
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

    jd_context = (f"\n\nJOB CONTEXT (use this to judge relevance):\n{jd_snippet}"
                  if jd_snippet else "")

    prompt = f"""You are a senior technical recruiter reviewing a resume Skills section.
Decide whether "{keyword}" is a legitimate skill worth listing on a resume.{jd_context}

ACCEPT — these belong in Skills:
  • Programming / scripting languages
  • ML / AI frameworks and libraries
  • Agentic AI tools and frameworks
  • Data engineering / analytics tools
  • Cloud platforms and managed services
  • DevOps and infrastructure tools
  • Databases and storage systems
  • ML concepts and architectures
  • Protocols and data formats
  • Technical methodologies
  • Domain-specific technical concepts
  • Soft skills / interpersonal skills if they appear in the JD
  • Any specific tool/technology named in the job context above

REJECT — these do NOT belong in Skills:
  • Phrases starting with "Experience with/in/of", "Understanding of", "Knowledge of"
  • Domain knowledge claims without a tool (unless exact term is in JD)
  • Sentences or outcome statements
  • Degree / experience requirements
  • Compliance standards without tech
  • Overly generic single words

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
# EXTRACT ALL JD SKILLS (including soft skills) — v2.4.0+
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
        log_event(f"📋 [ALL JD SKILLS] {len(tech)} technical + {len(soft)} soft = {len(all_skills)}")
        return all_skills
    except Exception as e:
        log_event(f"⚠️ [ALL JD SKILLS] Failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# COMPANY CONTEXT
# ═══════════════════════════════════════════════════════════════

_company_cache: Dict[str, Dict] = {}


async def get_company_context_gpt(name: str) -> Dict[str, Any]:
    nl = (name or "").lower().strip()
    if nl in _company_cache:
        return _company_cache[nl]
    prompt = f"""Analyze "{name}" for resume context. Return STRICT JSON:
{{"type":"industry_internship|research_internship|internship","domain":"2-4 words",
"context":"1-2 sentences","technical_vocabulary":["5-8 terms"],
"realistic_technologies":["6-10 tools"],"unrealistic_technologies":["3-5 tools this company would NOT use"]}}
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
        }
        _company_cache[nl] = result
        log_event(f"🏢 [COMPANY] {name}: {result['domain']}")
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
        data = await gpt_json(prompt, temperature=0.3, model="gpt-5.4-mini")
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
# LATEX UTILITIES
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
# PDF METADATA INJECTION — NEW in v2.5.0
# ═══════════════════════════════════════════════════════════════

def inject_pdf_metadata(
    tex: str,
    target_company: str,
    target_role: str,
    skills_list: List[str],
    courses: List[str],
) -> str:
    """v2.5.0: Inject PDF metadata into LaTeX preamble.
    Sets pdfauthor=Sri Akash Kadali, pdfkeywords=relevant skills,
    pdfsubject=coursework, pdftitle=resume info, pdfcreator."""

    # Build keyword string from skills (hashtag format for Additional Info)
    skill_tags = " ".join(f"#{s.replace(' ', '')}" for s in skills_list[:30] if s)
    # Plain skills for pdfkeywords
    skill_kw_str = ", ".join(_ensure_cap(s) for s in skills_list[:40] if s)
    # Coursework string
    course_str = ", ".join(c for c in courses[:12] if c)

    # Escape special LaTeX chars in metadata strings
    def _meta_escape(s: str) -> str:
        # For hyperref metadata, we need minimal escaping
        return (s or "").replace("\\", "").replace("{", "").replace("}", "")

    author = "Sri Akash Kadali"
    title = f"{author} - Resume - {_meta_escape(target_role)} at {_meta_escape(target_company)}"
    subject = f"Relevant Coursework: {_meta_escape(course_str)}" if course_str else f"Resume for {_meta_escape(target_role)}"
    creator = "Sri Akash Kadali"
    keywords = _meta_escape(skill_kw_str)

    hypersetup_block = (
        "\n% --- PDF Metadata (v2.5.0) ---\n"
        "\\usepackage{hyperref}\n"
        "\\hypersetup{\n"
        "  pdfauthor={" + author + "},\n"
        "  pdftitle={" + title + "},\n"
        "  pdfsubject={" + subject + "},\n"
        "  pdfkeywords={" + keywords + "},\n"
        "  pdfcreator={" + creator + "},\n"
        "  pdfproducer={Sri Akash Kadali},\n"
        "  hidelinks,\n"
        "  colorlinks=false,\n"
        "}\n"
        "% --- End PDF Metadata ---\n"
    )

    # Check if hyperref is already loaded
    if r"\usepackage{hyperref}" in tex or r"\usepackage[" in tex and "hyperref" in tex:
        # Just inject/replace the hypersetup block
        hs_pat = re.compile(r"\\hypersetup\{[^}]*\}", re.DOTALL)
        hs_match = hs_pat.search(tex)
        if hs_match:
            new_hs = (
                "\\hypersetup{\n"
                "  pdfauthor={" + author + "},\n"
                "  pdftitle={" + title + "},\n"
                "  pdfsubject={" + subject + "},\n"
                "  pdfkeywords={" + keywords + "},\n"
                "  pdfcreator={" + creator + "},\n"
                "  pdfproducer={Sri Akash Kadali},\n"
                "  hidelinks,\n"
                "  colorlinks=false,\n"
                "}"
            )
            tex = tex[:hs_match.start()] + new_hs + tex[hs_match.end():]
        else:
            # hyperref loaded but no hypersetup — inject before \begin{document}
            bd = tex.find(r"\begin{document}")
            if bd >= 0:
                inject = (
                    "\\hypersetup{\n"
                    "  pdfauthor={" + author + "},\n"
                    "  pdftitle={" + title + "},\n"
                    "  pdfsubject={" + subject + "},\n"
                    "  pdfkeywords={" + keywords + "},\n"
                    "  pdfcreator={" + creator + "},\n"
                    "  pdfproducer={Sri Akash Kadali},\n"
                    "  hidelinks,\n"
                    "  colorlinks=false,\n"
                    "}\n"
                )
                tex = tex[:bd] + inject + tex[bd:]
    else:
        # No hyperref at all — inject the full block before \begin{document}
        bd = tex.find(r"\begin{document}")
        if bd >= 0:
            tex = tex[:bd] + hypersetup_block + tex[bd:]

    log_event(f"📄 [METADATA] Injected PDF metadata: author={author}, {len(skills_list)} skill keywords, {len(courses)} courses")
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


async def extract_coursework_gpt(jd: str, max_courses: int = 24) -> List[str]:
    try:
        data = await gpt_json(
            f'Choose up to {max_courses} relevant university courses for this JD. '
            f'Return STRICT JSON: {{"courses":["..."]}}\nJD:\n{jd}', temperature=0.0)
        courses = await fix_capitalization_batch(
            [str(c).strip() for c in (data.get("courses") or []) if str(c).strip()])
        seen: Set[str] = set()
        out = []
        for c in courses:
            c = _ensure_cap(c)
            if c and c.lower() not in seen:
                seen.add(c.lower()); out.append(c)
        return out[:max_courses]
    except Exception:
        return []


def replace_relevant_coursework_distinct(body: str, courses: List[str], mpl: int = 6) -> str:
    seen: Set[str] = set()
    uniq = []
    for c in courses:
        c = _ensure_cap(re.sub(r"\s+", " ", str(c)).strip())
        if c and c.lower() not in seen:
            seen.add(c.lower()); uniq.append(c)
    pat = re.compile(r"(\\item\s*\\textbf\{Relevant Coursework:\})([^\n]*)")
    matches = list(pat.finditer(body))
    if not matches:
        return body
    chunks = ([uniq[:mpl]] if len(matches) == 1
              else [uniq[:(len(uniq) + 1) // 2][:mpl], uniq[(len(uniq) + 1) // 2:][:mpl]])
    out, last = [], 0
    for i, m in enumerate(matches):
        out.append(body[last:m.start()])
        out.append(m.group(1) + " " + ", ".join(latex_escape_text(x) for x in chunks[i])
                   if i < len(chunks) else m.group(0))
        last = m.end()
    out.append(body[last:])
    return "".join(out)


# ═══════════════════════════════════════════════════════════════
# SKILL RANKING BY JD RELEVANCE — v2.5.1
# ═══════════════════════════════════════════════════════════════

MAX_SKILLS = 30           # hard cap for initial render
MAX_SKILLS_TIGHT = 20     # tighter cap used during page-fit trimming


def rank_skills_by_jd_relevance(
    skills_raw: List[str],
    must_have: List[str],
    should_have: List[str],
    nice_to_have: List[str],
    core_keywords: List[str],
    jd_text: str,
    max_skills: int = MAX_SKILLS,
) -> List[str]:
    """
    v2.5.1: Score each skill by JD priority tier, then cap.
    must_have=100, core=80, should=60, nice=40, in JD=20, other=1.
    """
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

        if sl in must_set:
            score = 100
        elif sl in core_set:
            score = 80
        elif sl in should_set:
            score = 60
        elif sl in nice_set:
            score = 40
        elif sl in jd_lower:
            score = 20
        else:
            score = 1

        scored.append((score, -idx, s))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    result = [s for _, _, s in scored[:max_skills]]
    log_event(f"📊 [SKILLS RANK] {len(skills_raw)} raw → {len(result)} after ranking (cap={max_skills})")
    return result


# ═══════════════════════════════════════════════════════════════
# SKILLS SECTION RENDERING — FLAT ONLY (v2.4.0+)
# ═══════════════════════════════════════════════════════════════

def render_skills_section_flat(skills: List[str]) -> str:
    if not skills:
        return ""
    seen: Set[str] = set()
    us = []
    for s in skills:
            s = _ensure_cap(str(s).strip())
            if s and s.lower() not in seen:
                seen.add(s.lower()); us.append(s)
    return (
        "\\section{Skills}\n"
        "\\begin{itemize}[leftmargin=0.15in, label={}]\n"
        "  \\item \\small{" + ", ".join(latex_escape_text(s) for s in us) + "}\n"
        "\\end{itemize}"
    )


async def replace_skills_section(body: str, skills: List[str], jd_text: str = "") -> str:
    nb = render_skills_section_flat(skills)
    if not nb:
        return body
    pat = re.compile(
        r"(\\section\*?\{Skills\}[\s\S]*?)(?=%-----------|\\section\*?\{|\\end\{document\})",
        re.I)
    if re.search(pat, body):
        return re.sub(pat, lambda _: nb + "\n", body)
    m = re.search(r"%-----------TECHNICAL SKILLS-----------", body, re.I)
    if m:
        return body[:m.end()] + "\n" + nb + "\n" + body[m.end():]
    return body


# ═══════════════════════════════════════════════════════════════
# ATS SELF-SIMULATION PASS
# ═══════════════════════════════════════════════════════════════

async def ats_self_simulation_pass(
    body_tex: str,
    jd_text: str,
    all_keywords: List[str],
    must_have: List[str],
) -> List[str]:
    resume_plain = strip_all_macros_keep_text(body_tex).lower()
    present = [k for k in all_keywords if k.lower() in resume_plain]
    missing_must = [k for k in must_have if k.lower() not in resume_plain]

    prompt = f"""You are an ATS (Applicant Tracking System) scanner AND a senior recruiter
reviewing a candidate's resume against a job description.

JOB DESCRIPTION (first 3000 chars):
{jd_text[:3000]}

RESUME PLAIN TEXT (extracted):
{resume_plain[:3000]}

KEYWORDS ALREADY PRESENT: {json.dumps(present[:25])}
MUST-HAVE KEYWORDS CURRENTLY MISSING: {json.dumps(missing_must[:10])}

Identify 8-12 technical keywords from the JD that are:
1. Clearly relevant to this role
2. Either absent from the resume OR only in passing (not in Skills)
3. Real technical terms — languages, frameworks, tools, ML concepts, platforms
   (NOT degree requirements, NOT "experience with X" phrases)
   Soft skills ARE acceptable if they appear in the JD.

Return STRICT JSON:
{{
    "missing_keywords": ["keyword1", "keyword2", ...],
    "reasoning": "brief explanation"
}}
Max 12 items."""

    try:
        data = await gpt_json(prompt, temperature=0.0)
        raw = [str(k).strip() for k in (data.get("missing_keywords") or []) if str(k).strip()]
        if not raw:
            return []
        validated = await filter_valid_skills(raw, jd_text[:500])
        if validated:
            validated = await fix_capitalization_batch(validated)
        log_event(f"🤖 [ATS SIM] {len(validated)} under-represented keywords → adding to Skills")
        return validated[:12]
    except Exception as e:
        log_event(f"⚠️ [ATS SIM] Failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# MASTER 12-BULLET PLAN
# ═══════════════════════════════════════════════════════════════

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

JD TASKS:
{tasks_str}

JD KEYWORDS: {json.dumps(jd_keywords[:25])}

IDEAL CANDIDATE:
- Must-haves: {json.dumps(top_3)}
- Implicit: {json.dumps([r.get('requirement', '') for r in implicit_reqs[:4]])}

Return STRICT JSON:
{{
    "bullet_plan": [
        {{
            "bullet_index": 0,
            "block_index": 0,
            "experience_company": "company name",
            "assigned_jd_task": "the JD task this bullet mirrors",
            "task_id": 1,
            "task_category": "build_system|analyze_data|train_model|deploy_service|build_pipeline|optimize_performance|automate_process|collaborate|research|monitor",
            "bullet_seed": "1-sentence past-tense description of what this bullet says",
            "primary_technology": "specific technology",
            "supporting_keywords": ["2-3 JD keywords"],
            "jd_phrases_to_mirror": ["1-2 exact JD phrases"],
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
3. 5-6 bullets have metrics, spread across blocks
4. Block 0 = most recent (advanced), Block 3 = oldest (simpler)
5. task_category MUST match the task's actual category
6. jd_phrases_to_mirror must be EXACT phrases from the JD
7. metric_hint must NOT be a bare round percentage
8. METRIC DIVERSITY: Across 12 bullets, use a MIX of metric types:
   - At most 2 percentage-based metrics
   - At least 2 "from X to Y" improvements (e.g., "from 47ms to 13ms")
   - At least 1 time-based metric (e.g., "saving 23 hours/week")
   - At least 1 count/volume metric (e.g., "13K requests/day")
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
                "task_id": t.get("task_id", 1), "task_category": tc,
                "bullet_seed": t.get("what_good_looks_like", "Technical contribution"),
                "primary_technology": (t.get("implied_technologies", ["Python"]) or ["Python"])[0],
                "supporting_keywords": [], "jd_phrases_to_mirror": [],
                "result_type": "qualitative_insight",
                "has_metric": idx % 3 == 0, "metric_hint": pick_metric_hint(tc),
            })
        log_event("📋 [MASTER PLAN] 12 bullets planned")
        return {"bullet_plan": plan[:12], "task_coverage": data.get("task_coverage", {})}
    except Exception as e:
        log_event(f"⚠️ [PLAN] Failed: {e}")
        plan = []
        for i in range(12):
            block = i // 3
            t = jd_tasks[i % len(jd_tasks)] if jd_tasks else {}
            tc = t.get("task_category", "build_system")
            plan.append({
                "bullet_index": i, "block_index": block,
                "assigned_jd_task": t.get("task_description", "Technical work"),
                "task_id": t.get("task_id", 1), "task_category": tc,
                "bullet_seed": t.get("what_good_looks_like", "Technical contribution"),
                "primary_technology": (t.get("implied_technologies", ["Python"]) or ["Python"])[0],
                "supporting_keywords": [], "jd_phrases_to_mirror": [],
                "result_type": "qualitative_insight",
                "has_metric": i % 3 == 0, "metric_hint": pick_metric_hint(tc),
            })
        return {"bullet_plan": plan, "task_coverage": {}}


# ═══════════════════════════════════════════════════════════════
# BULLET GENERATION
# ═══════════════════════════════════════════════════════════════

_global_kw_assignments: Dict[str, int] = {}


def reset_keyword_assignment_tracking():
    global _global_kw_assignments
    _global_kw_assignments.clear()


def _build_bullet_prompt_v2(
    experience_company: str, target_company: str, target_role: str,
    block_index: int, total_blocks: int, suggested_verbs: List[str],
    bullet_plans: List[Dict], role_archetype: Dict, exp_context: Dict,
    progression: Dict, jd_text: str, all_keywords: List[str],
    already_used_kws: List[str],
) -> str:
    ak = role_archetype.get("key", "general_tech")
    bf = role_archetype.get("bullet_focus", "")
    ps = role_archetype.get("phrasing_style", "")
    rt = role_archetype.get("result_types", "")
    av = role_archetype.get("avoid", "")
    tone_reg = role_archetype.get("tone_register", "technical")
    tone_pace = role_archetype.get("tone_pace", "measured")
    tone_vocab = role_archetype.get("tone_vocabulary", "precise")
    tone_ex = role_archetype.get("tone_examples", [])
    comp = progression.get("complexity", "intermediate")
    auton = progression.get("autonomy", "with mentorship")
    real_tech = exp_context.get("realistic_technologies", [])
    unreal_tech = exp_context.get("unrealistic_technologies", [])
    domain = exp_context.get("domain", "Technology")

    seeds = []
    for i, bp in enumerate(bullet_plans[:3]):
        task = bp.get("assigned_jd_task", "Technical contribution")
        seed = bp.get("bullet_seed", "")
        tech = bp.get("primary_technology", "")
        has_m = bp.get("has_metric", False)
        m_hint = bp.get("metric_hint", "")
        skws = bp.get("supporting_keywords", [])
        jd_phrases = bp.get("jd_phrases_to_mirror", [])
        tc = bp.get("task_category", "build_system")
        verb = suggested_verbs[i] if i < len(suggested_verbs) else "Built"

        s = (f"Bullet {i + 1} (verb: {verb}):\n"
             f"   JD TASK: {task}\n"
             f"   SEED: {seed}\n"
             f"   TECHNOLOGY: {tech}\n")
        if jd_phrases:
            s += f"   MIRROR THESE EXACT JD PHRASES (verbatim): {', '.join(jd_phrases[:2])}\n"
        if skws:
            s += f"   WEAVE IN: {', '.join(skws[:3])}\n"
        if has_m:
            hint = m_hint if m_hint else pick_metric_hint(tc)
            s += f"   METRIC (MUST vary type — see rules): {hint}\n"
        else:
            s += f"   RESULT: Specific qualitative result connected to THIS task\n"
        seeds.append(s)

    dedup = ""
    if already_used_kws:
        dedup = (f"\n🚫 Already used in other blocks (avoid as PRIMARY): "
                 f"{', '.join(already_used_kws[:15])}\n")

    company_constraint = (
        f"\n⚠️ COMPANY REALISM: Work at {experience_company} ({domain}).\n"
        f"   Realistic technologies: {', '.join(real_tech[:6])}\n"
    )
    if unreal_tech:
        company_constraint += f"   DO NOT MENTION: {', '.join(unreal_tech[:4])}\n"

    tone_section = (
        f"\nTONE MATCHING: Register: {tone_reg} | Pace: {tone_pace} | Vocab: {tone_vocab}\n"
    )
    if tone_ex:
        tone_section += f"   JD tone phrases: {', '.join(tone_ex[:3])}\n"

    return f"""Write 3 resume bullets for an intern at "{experience_company}" applying for {target_role} at {target_company}.

═══ THE ACTUAL JOB DESCRIPTION (mirror its language) ═══
{jd_text[:3000]}
═══════════════════════════════════════════════════════

ROLE: {ak} | Focus: {bf} | Style: {ps} | Results: {rt} | Avoid: {av}
{tone_section}
CONTEXT: {experience_company} ({domain}) | Block {block_index}/{total_blocks} ({comp}) | {auton}
{company_constraint}

═══ BULLET SEEDS ═══
{chr(10).join(seeds)}
{dedup}
ATS KEYWORDS: {', '.join(all_keywords[:15])}

═══ RULES ═══
1. TASK IS THE STAR. Technology is context, not subject.
   BAD: "Fine-tuned BERT for classification, achieving 85% accuracy."
   GOOD: "Built the content moderation classifier, fine-tuning BERT on 8K examples to cut manual review time by half."

2. ONE FLOWING SENTENCE per bullet. No "Utilized X for Y, achieving Z" template.
3. RESULT MUST CONNECT TO THE TASK. Pipeline bullet → pipeline result.
4. USE EXACT JD PHRASES where provided. Verbatim. Non-negotiable.
5. 24-34 words. Start with exact verb given. Vary sentence structure.
6. Intern scope: thousands not millions. Use "%" not "\\%".
7. PHRASE MIRRORING: provided JD phrases must appear verbatim in your bullet.
8. NEVER start two bullets with the same verb or syntactic pattern.
9. METRICS — CRITICAL DIVERSITY RULES:
   - Numbers must end in odd digits (43ms not 40ms, 13K not 10K).
   - NEVER use ONLY percentages. Mix these metric TYPES across bullets:
     a) "from X to Y" improvements (e.g., "from 47ms to 13ms", "from 8 hours to 23 min")
     b) Absolute counts (e.g., "13K requests/day", "47 microservices", "83 test cases")
     c) Time savings (e.g., "saving 23 engineer-hours/week", "under 7 seconds")
     d) Multipliers (e.g., "3x faster inference", "7x throughput")
     e) Percentages ONLY if other types already used (max 1 per 3 bullets)
   - Each bullet in this block MUST use a DIFFERENT metric type from the list above.
10. NEVER use placeholder words like XYZ, ABC, Foo, Bar, Lorem, Acme, widget, gadget,
    "some tool", "the system", "Company A". Use SPECIFIC REAL names only.

Return STRICT JSON:
{{"bullets":["{suggested_verbs[0] if suggested_verbs else 'Built'}...","{suggested_verbs[1] if len(suggested_verbs) > 1 else 'Designed'}...","{suggested_verbs[2] if len(suggested_verbs) > 2 else 'Automated'}..."],
"keywords_used":["kw1","kw2"],"technologies_used":["t1","t2"],
"jd_tasks_mirrored":["task1","task2","task3"],"jd_phrases_used":["phrase1","phrase2"]}}"""


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
                b = verbs[i] + " " + b[0].lower() + b[1:] if b else verbs[i]
        dm = re.search(
            r',?\s+\b(by|of|to|from|through|via|using|across|with|achieving|improving|'
            r'enhancing|boosting|increasing|reducing|raising|lifting)\s*[.,]?\s*$', b, re.I)
        if dm:
            b = b[:dm.start()].rstrip(".,;: ") + "."
        b = adjust_bullet_length(b)
        if not b.endswith("."):
            b = b.rstrip(".,;: ") + "."
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
) -> Tuple[List[str], Set[str]]:
    exp_ctx = await get_company_context_gpt(exp_company)
    prog = get_progression_context(block_index, total_blocks)
    verbs = []
    for bp in plans[:n]:
        tc = bp.get("task_category", "build_system")
        vc = TASK_CAT_TO_VERB_CAT.get(tc, "development")
        verbs.append(get_diverse_verb(vc))
    while len(verbs) < n:
        verbs.append(get_diverse_verb("development"))

    prompt = _build_bullet_prompt_v2(
        exp_company, target_company, target_role, block_index, total_blocks,
        verbs, plans, role_archetype, exp_ctx, prog, jd_text, all_keywords,
        list(_global_kw_assignments.keys()),
    )

    cleaned, used = [], set()
    for attempt in range(3):
        try:
            temp = 0.35 + (attempt * 0.12)
            data = await gpt_json(prompt, temperature=temp)
            bullets = data.get("bullets", []) or []
            if not bullets:
                log_event(f"⚠️ [BLOCK {block_index}] Attempt {attempt + 1}: empty response")
                continue
            if len(bullets) < n:
                log_event(f"⚠️ [BLOCK {block_index}] Attempt {attempt + 1}: only {len(bullets)}")
                continue
            cleaned, used = await _post_process(
                bullets, data.get("keywords_used", []),
                data.get("technologies_used", []), n, start_pos, verbs, all_keywords)
            if len(cleaned) >= n:
                break
            log_event(f"⚠️ [BLOCK {block_index}] Attempt {attempt + 1}: {len(cleaned)} after cleanup")
        except Exception as e:
            log_event(f"⚠️ [BLOCK {block_index}] Attempt {attempt + 1} error: {e}")

    while len(cleaned) < n:
        idx = len(cleaned)
        bp = plans[idx] if idx < len(plans) else {}
        verb = verbs[idx] if idx < len(verbs) else "Built"
        task = bp.get("assigned_jd_task", "technical contribution")
        tech = bp.get("primary_technology", "Python")
        tc = bp.get("task_category", "build_system")
        for micro_attempt in range(2):
            try:
                mb = await gpt_json(
                    f'Write ONE resume bullet. Intern at {exp_company}, applying for {target_role}. '
                    f'Start with "{verb}". Mirror JD task: "{task}". Mention {tech}. '
                    f'Metric (odd digits, NOT a percentage — use counts or time or from-to): {pick_metric_hint(tc)}. '
                    f'25-30 words, past tense. No placeholder words (XYZ, ABC, widget, etc). '
                    f'Return STRICT JSON: {{"bullet":"{verb} ..."}}',
                    temperature=0.4 + micro_attempt * 0.15)
                bullet = mb.get("bullet", "")
                if bullet and len(bullet.split()) >= 10:
                    bullet = await fix_capitalization_gpt(bullet)
                    bullet = adjust_bullet_length(bullet)
                    if not bullet.endswith("."):
                        bullet = bullet.rstrip(".,;: ") + "."
                    cleaned.append(latex_escape_text(bullet))
                    log_event(f"🔄 [MICRO-RETRY] Generated for position {start_pos + idx}")
                    break
            except Exception:
                pass
        else:
            log_event(f"❌ [BLOCK {block_index}] Could not generate bullet {idx}")

    log_event(f"✅ [BLOCK {block_index}] {len(cleaned)} bullets for {exp_company}")
    return cleaned[:n], used


# ═══════════════════════════════════════════════════════════════
# POST-GENERATION TASK ALIGNMENT VALIDATION  [BUG FIX]
# ═══════════════════════════════════════════════════════════════

async def validate_and_fix_task_alignment(
    all_bullets: List[List[str]],
    bullet_plan: List[Dict],
    jd_text: str, target_role: str, role_archetype: Dict,
) -> List[List[str]]:
    flat_bullets = [b for block in all_bullets for b in block]
    if len(flat_bullets) < 6:
        return all_bullets

    checks = [
        {
            "idx": i,
            "bullet": flat_bullets[i][:180],
            "assigned_task": (bullet_plan[i].get("assigned_jd_task", "")[:100]
                              if i < len(bullet_plan) else ""),
        }
        for i in range(min(len(flat_bullets), len(bullet_plan)))
    ]

    prompt = f"""Rate how well each resume bullet demonstrates its assigned JD task.
Target role: {target_role}

{json.dumps(checks)}

Score each 0.0-1.0:
  1.0 = clearly demonstrates capability for that task
  0.75 = acceptable but could be stronger
  0.5 = loosely related, not direct evidence
  0.0 = completely misaligned

Return STRICT JSON: {{"results": [{{"idx": 0, "score": 0.9, "reason": "brief reason"}}]}}
"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        raw_results = data.get("results") or []
        results = {}
        for r in raw_results:
            if not isinstance(r, dict):
                continue
            raw_idx = r.get("idx")
            if not isinstance(raw_idx, int):
                try:
                    raw_idx = int(raw_idx)
                except (TypeError, ValueError):
                    continue
            results[raw_idx] = r
    except Exception:
        log_event("⚠️ [VALIDATION] Scoring failed, keeping bullets as-is")
        return all_bullets

    low_scoring = sorted(
        [r for r in results.values() if r.get("score", 1.0) < 0.75],
        key=lambda x: x.get("score", 1.0))
    log_event(f"🔍 [VALIDATION] {len(low_scoring)} bullets scored < 0.75 out of {len(flat_bullets)}")

    for r in low_scoring[:6]:
        raw_idx = r.get("idx", -1)
        if not isinstance(raw_idx, int):
            try:
                raw_idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
        idx = raw_idx
        if idx < 0 or idx >= len(flat_bullets):
            continue
        plan = bullet_plan[idx] if idx < len(bullet_plan) else {}
        task = plan.get("assigned_jd_task", "")
        tech = plan.get("primary_technology", "Python")
        jd_phrases = plan.get("jd_phrases_to_mirror", [])
        tc = plan.get("task_category", "build_system")
        verb = re.sub(r"\\[#$%&_{}]", "",
                      flat_bullets[idx].split()[0]) if flat_bullets[idx].split() else "Built"

        phrase_instruction = ""
        if jd_phrases:
            phrase_instruction = f'MIRROR THESE EXACT JD PHRASES: {", ".join(jd_phrases[:2])}\n'

        try:
            fix = await gpt_json(
                f'Rewrite this bullet to CLEARLY demonstrate: "{task}"\n'
                f'CURRENT (score {r.get("score", 0):.2f}): "{flat_bullets[idx][:200]}"\n'
                f'PROBLEM: {r.get("reason", "weak alignment")}\n'
                f'{phrase_instruction}'
                f'Start with "{verb}". Mention {tech}. Metric: {pick_metric_hint(tc)}\n'
                f'24-34 words. Past tense. ONE sentence. No placeholder words (XYZ, ABC, etc).\n'
                f'Return STRICT JSON: {{"bullet": "..."}}',
                temperature=0.35)
            new_b = fix.get("bullet", "")
            if new_b and len(new_b.split()) >= 15:
                new_b = await fix_capitalization_gpt(new_b)
                new_b = adjust_bullet_length(new_b)
                if not new_b.endswith("."):
                    new_b = new_b.rstrip(".,;: ") + "."
                flat_bullets[idx] = latex_escape_text(new_b)
                log_event(f"✅ [REALIGN] idx={idx} score={r.get('score', 0):.2f} → rewritten")
        except Exception:
            pass

    result, i = [], 0
    for block in all_bullets:
        result.append(flat_bullets[i:i + len(block)])
        i += len(block)
    return result


# ═══════════════════════════════════════════════════════════════
# CROSS-BLOCK SEMANTIC DEDUPLICATION  [BUG FIX]
# ═══════════════════════════════════════════════════════════════

async def deduplicate_across_blocks(
    all_bullets: List[List[str]], bullet_plan: List[Dict],
    jd_text: str, role_archetype: Dict,
) -> List[List[str]]:
    flat = [b for block in all_bullets for b in block]
    if len(flat) < 6:
        return all_bullets

    prompt = f"""Check these resume bullets for semantic redundancy.
Two bullets are redundant if they describe essentially the same work/achievement.

{json.dumps([{"idx": i, "bullet": b[:150] if isinstance(b, str) else str(b)[:150]}
             for i, b in enumerate(flat[:12])])}

Return STRICT JSON:
{{"duplicate_pairs": [{{"idx_a": 0, "idx_b": 5, "reason": "both describe data pipeline building"}}],
"all_unique": true}}
Only flag TRULY redundant pairs. idx_a and idx_b must be plain integers.
"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        pairs = data.get("duplicate_pairs", [])
    except Exception:
        return all_bullets

    if not pairs:
        log_event("✅ [DEDUP] All bullets are unique")
        return all_bullets

    log_event(f"🔄 [DEDUP] Found {len(pairs)} redundant pairs")
    rewritten_indices: Set[int] = set()

    for pair in pairs[:3]:
        if not isinstance(pair, dict):
            continue
        raw_a = pair.get("idx_a", -1)
        raw_b = pair.get("idx_b", -1)
        try:
            idx_a = int(raw_a) if not isinstance(raw_a, int) else raw_a
            idx_b = int(raw_b) if not isinstance(raw_b, int) else raw_b
        except (TypeError, ValueError):
            continue
        if idx_a < 0 or idx_b < 0:
            continue
        rewrite_idx = max(idx_a, idx_b)
        if rewrite_idx >= len(flat) or rewrite_idx in rewritten_indices:
            continue

        plan = bullet_plan[rewrite_idx] if rewrite_idx < len(bullet_plan) else {}
        task = plan.get("assigned_jd_task", "different technical work")
        tech = plan.get("primary_technology", "Python")
        tc = plan.get("task_category", "build_system")
        verb = re.sub(r"\\[#$%&_{}]", "",
                      flat[rewrite_idx].split()[0]) if flat[rewrite_idx].split() else "Built"

        try:
            fix = await gpt_json(
                f'This bullet is too similar to another. REWRITE to focus on: "{task}"\n'
                f'DUPLICATE OF: "{flat[min(idx_a, idx_b)][:150]}"\n'
                f'Mention {tech}. Start with "{verb}". 24-34 words. '
                f'Metric: {pick_metric_hint(tc)}. No placeholder words.\n'
                f'Return STRICT JSON: {{"bullet": "..."}}',
                temperature=0.4)
            new_b = fix.get("bullet", "")
            if new_b and len(new_b.split()) >= 15:
                new_b = await fix_capitalization_gpt(new_b)
                new_b = adjust_bullet_length(new_b)
                if not new_b.endswith("."):
                    new_b = new_b.rstrip(".,;: ") + "."
                flat[rewrite_idx] = latex_escape_text(new_b)
                rewritten_indices.add(rewrite_idx)
                log_event(f"✅ [DEDUP] Bullet {rewrite_idx} rewritten")
        except Exception:
            pass

    result, i = [], 0
    for block in all_bullets:
        result.append(flat[i:i + len(block)])
        i += len(block)
    return result


# ═══════════════════════════════════════════════════════════════
# BULLET QUALITY RUBRIC  [BUG FIX]
# ═══════════════════════════════════════════════════════════════

async def score_bullet_quality_rubric(
    all_bullets: List[List[str]],
    bullet_plan: List[Dict],
    jd_text: str,
    target_role: str,
) -> List[List[str]]:
    flat = [b for block in all_bullets for b in block]
    if len(flat) < 3:
        return all_bullets

    checks = []
    for i, b in enumerate(flat):
        plan = bullet_plan[i] if i < len(bullet_plan) else {}
        checks.append({
            "idx": i,
            "bullet": b[:180],
            "assigned_task": plan.get("assigned_jd_task", "")[:80],
            "jd_phrases": plan.get("jd_phrases_to_mirror", [])[:2],
        })

    prompt = f"""Score each resume bullet on 4 axes (0=poor, 1=weak, 2=good, 3=excellent).
Target role: {target_role}

  task_specificity:  Concrete specific task described vs vague generic claim
  jd_phrase_usage:   JD language mirrored vs generic resume-speak
  metric_quality:    Concrete count/latency/throughput vs missing/bare-round-%
  verb_strength:     Strong action verb vs "worked on" / "helped with"

BULLETS:
{json.dumps([{"idx": c["idx"], "bullet": c["bullet"], "task": c["assigned_task"]}
             for c in checks[:12]])}

Return STRICT JSON — idx must be a plain integer:
{{"scores": [{{"idx": 0, "task_specificity": 2, "jd_phrase_usage": 1, "metric_quality": 3,
              "verb_strength": 2, "total": 8, "weakest_axis": "jd_phrase_usage"}}]}}
"""
    try:
        data = await gpt_json(prompt, temperature=0.0)
        raw_scores = data.get("scores") or []
        scores = {}
        for s in raw_scores:
            if not isinstance(s, dict):
                continue
            raw_idx = s.get("idx")
            try:
                idx = int(raw_idx) if not isinstance(raw_idx, int) else raw_idx
            except (TypeError, ValueError):
                continue
            scores[idx] = s
    except Exception:
        log_event("⚠️ [RUBRIC] Scoring failed, skipping quality pass")
        return all_bullets

    weak = sorted(
        [s for s in scores.values() if s.get("total", 12) <= 5],
        key=lambda x: x.get("total", 12))
    log_event(f"📐 [RUBRIC] {len(weak)} bullets scored ≤ 5/12 out of {len(flat)}")

    axis_instructions = {
        "task_specificity": (
            "Make the bullet MUCH MORE SPECIFIC. Name the exact artifact built, "
            "the exact problem solved, the exact dataset/system used."),
        "jd_phrase_usage": (
            "Rewrite to INCORPORATE EXACT LANGUAGE from the JD. "
            "Mirror the JD's vocabulary throughout."),
        "metric_quality": (
            "Replace the metric with a CONCRETE NUMBER ending in an odd digit "
            "(43ms not 40ms, 13K not 10K). Use counts, latency, throughput, or time saved — NOT just percentages."),
        "verb_strength": (
            "Replace the opening verb with a STRONG ACTION VERB: Architected, Engineered, "
            "Deployed, Designed, Automated, Optimized, Productionized, Orchestrated."),
    }

    for s in weak[:6]:
        raw_idx = s.get("idx", -1)
        try:
            idx = int(raw_idx) if not isinstance(raw_idx, int) else raw_idx
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(flat):
            continue
        plan = bullet_plan[idx] if idx < len(bullet_plan) else {}
        task = plan.get("assigned_jd_task", "")
        tech = plan.get("primary_technology", "Python")
        tc = plan.get("task_category", "build_system")
        jd_phrases = plan.get("jd_phrases_to_mirror", [])
        verb = re.sub(r"\\[#$%&_{}]", "",
                      flat[idx].split()[0]) if flat[idx].split() else "Built"
        weakest = s.get("weakest_axis", "task_specificity")
        improvement = axis_instructions.get(weakest, "Improve overall specificity.")

        phrase_hint = ""
        if jd_phrases and weakest == "jd_phrase_usage":
            phrase_hint = f'INCORPORATE THESE EXACT PHRASES: {", ".join(jd_phrases[:2])}\n'

        try:
            fix = await gpt_json(
                f'Improve this resume bullet. Score={s.get("total", "?")}/12, '
                f'weakest: {weakest}.\n'
                f'CURRENT: "{flat[idx][:200]}"\n'
                f'TASK: "{task}"\n'
                f'IMPROVEMENT: {improvement}\n'
                f'{phrase_hint}'
                f'Start with "{verb}". Mention {tech}. Metric: {pick_metric_hint(tc)}\n'
                f'24-34 words. Past tense. ONE sentence. No placeholder words.\n'
                f'Return STRICT JSON: {{"bullet": "..."}}',
                temperature=0.4)
            new_b = fix.get("bullet", "")
            if new_b and len(new_b.split()) >= 15:
                new_b = await fix_capitalization_gpt(new_b)
                new_b = adjust_bullet_length(new_b)
                if not new_b.endswith("."):
                    new_b = new_b.rstrip(".,;: ") + "."
                flat[idx] = latex_escape_text(new_b)
                log_event(f"✅ [RUBRIC] idx={idx} total={s.get('total')} → improved ({weakest})")
        except Exception:
            pass

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
        log_event("✅ [COVERAGE] All must-have keywords present")
        return all_bullets

    log_event(f"⚠️ [COVERAGE] {len(missing_must)} must-have keywords missing: {missing_must[:5]}")

    for kw in missing_must[:3]:
        if not flat:
            break
        weakest_idx = len(flat) - 1
        plan = bullet_plan[weakest_idx] if weakest_idx < len(bullet_plan) else {}
        task = plan.get("assigned_jd_task", "technical contribution")
        verb = re.sub(r"\\[#$%&_{}]", "",
                      flat[weakest_idx].split()[0]) if flat[weakest_idx].split() else "Built"
        try:
            fix = await gpt_json(
                f'Rewrite this bullet to naturally incorporate "{kw}" (must-have JD keyword).\n'
                f'CURRENT: "{flat[weakest_idx][:200]}"\nTASK: "{task}"\n'
                f'Keep starting verb "{verb}". 24-34 words. Keyword must appear naturally.\n'
                f'Return STRICT JSON: {{"bullet": "..."}}',
                temperature=0.3)
            new_b = fix.get("bullet", "")
            if new_b and kw.lower() in new_b.lower() and len(new_b.split()) >= 15:
                new_b = await fix_capitalization_gpt(new_b)
                new_b = adjust_bullet_length(new_b)
                if not new_b.endswith("."):
                    new_b = new_b.rstrip(".,;: ") + "."
                flat[weakest_idx] = latex_escape_text(new_b)
                log_event(f"✅ [REMEDIATE] Injected '{kw}' into bullet {weakest_idx}")
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
    all_keywords: List[str],
) -> Tuple[str, Set[str]]:
    reset_verb_tracking()
    reset_keyword_assignment_tracking()
    reset_technology_tracking()
    reset_metric_type_tracking()

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
            if a < 0:
                break
            b = section.find(e_tag, a)
            if b < 0:
                break
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
                plans[:3], role_archetype, all_keywords, exp_used, 3)
            exp_used.update(used)
            all_blocks.append(bullets)
            block_index += 1
            abs_pos += 3
            i = b + len(e_tag)

    log_event("🔍 [VALIDATE] Scoring bullet-task alignment...")
    all_blocks = await validate_and_fix_task_alignment(
        all_blocks, bullet_plan, jd_text, target_role, role_archetype)

    log_event("🔄 [DEDUP] Checking cross-block redundancy...")
    all_blocks = await deduplicate_across_blocks(
        all_blocks, bullet_plan, jd_text, role_archetype)

    log_event("📐 [RUBRIC] Running 4-axis quality rubric...")
    all_blocks = await score_bullet_quality_rubric(
        all_blocks, bullet_plan, jd_text, target_role)

    # v2.5.0: Enforce metric diversity
    log_event("📊 [METRIC DIV] Enforcing metric type diversity...")
    all_blocks = await enforce_metric_diversity(
        all_blocks, bullet_plan, target_role, jd_text)

    must_have = jd_info.get("must_have", [])
    if must_have:
        log_event("📊 [REMEDIATE] Checking must-have keyword coverage...")
        all_blocks = await remediate_coverage_gaps(all_blocks, must_have, bullet_plan, jd_text)

    # v2.5.0: Sanitize placeholder words in all bullets
    log_event("🔧 [SANITIZE] Checking for placeholder words (XYZ, ABC, etc.)...")
    all_blocks = await sanitize_all_bullets(
        all_blocks, target_company, target_role, exp_companies)

    # Reassemble TeX
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

    log_event(f"✅ [EXPERIENCE] {len(exp_used)} keywords, {len(_used_verbs_global)} unique verbs")
    return "".join(out), exp_used


async def _extract_experience_companies(tex: str) -> List[str]:
    m = section_rx("Experience").search(tex)
    if not m:
        return []
    sec = m.group(1)
    companies = re.findall(r"\\resumeSubheading\{[^}]*\}\{[^}]*\}\{([^}]*)\}", sec)
    if not companies:
        companies = re.findall(r"\\resumeSubheading\{([^}]*)\}", sec)
    return [
        strip_all_macros_keep_text(c).strip()
        for c in companies
        if strip_all_macros_keep_text(c).strip() and len(strip_all_macros_keep_text(c).strip()) > 2
    ]


# ═══════════════════════════════════════════════════════════════
# JD-SEEDED PROJECT GENERATOR — v2.5.0: "Title — one-liner" format
# ═══════════════════════════════════════════════════════════════

async def generate_jd_projects(
    jd_text: str, jd_tasks: List[Dict[str, Any]],
    role_archetype: Dict[str, Any], must_have_keywords: List[str],
    target_role: str,
) -> List[Dict[str, str]]:
    """v2.5.0: Generate exactly 2 projects with separate name and description.
    Returns list of dicts with 'name' and 'description' keys."""
    high_tasks = [t for t in jd_tasks if t.get("priority") == "high"][:2]
    if len(high_tasks) < 2:
        high_tasks = jd_tasks[:2]
    top_tools = must_have_keywords[:8]

    prompt = f"""Generate exactly 2 resume project entries for someone applying for {target_role}.

FULL JOB DESCRIPTION (use EXACT tool names from here):
{jd_text[:3500]}

HIGH-PRIORITY JD TASKS TO MIRROR:
Task 1: {high_tasks[0].get('task_description', '') if high_tasks else ''}
  Tools: {', '.join(high_tasks[0].get('implied_technologies', [])[:3]) if high_tasks else ''}

Task 2: {high_tasks[1].get('task_description', '') if len(high_tasks) > 1 else ''}
  Tools: {', '.join(high_tasks[1].get('implied_technologies', [])[:3]) if len(high_tasks) > 1 else ''}

MUST-INCLUDE TOOLS (from JD): {', '.join(top_tools[:6])}

RULES:
1. Each project has a NAME and a DESCRIPTION (separate fields).
2. NAME: Real CS/ML project name (e.g., "MedNotes Classifier", "StreamSense Pipeline", "SpectraVision Analyzer").
   NEVER use placeholder names like "Project XYZ" or "Tool ABC".
3. DESCRIPTION: ONE sentence, 18-28 words, past tense describing what was built + specific result.
4. Tools MUST come from the JD — no invented tools.
5. Result must be SPECIFIC with a number ending in an odd digit — NOT a bare round percentage.
   Use diverse metric types: counts, from-X-to-Y, time savings, throughput — not just %.
6. Use "%" not "\\%".
7. NEVER use placeholder words: XYZ, ABC, Foo, Bar, widget, gadget, Acme, "some tool".

Return STRICT JSON:
{{
    "projects": [
        {{
            "name": "Descriptive Project Name",
            "description": "One-line past-tense description (18-28 words) with specific result.",
            "tools_used": ["tool1", "tool2"],
            "jd_task_mirrored": "task description"
        }},
        {{
            "name": "Descriptive Project Name",
            "description": "One-line past-tense description (18-28 words) with specific result.",
            "tools_used": ["tool1", "tool2"],
            "jd_task_mirrored": "task description"
        }}
    ]
}}
"""
    for attempt_temp in [0.3, 0.5]:
        try:
            data = await gpt_json(prompt, temperature=attempt_temp)
            projects = data.get("projects", [])
            result = []
            for p in projects[:2]:
                name = str(p.get("name", "")).strip()
                desc = str(p.get("description", "")).strip()
                if not name or not desc:
                    continue
                name = await fix_capitalization_gpt(name)
                desc = await fix_capitalization_gpt(desc)
                desc = adjust_bullet_length(desc)
                if not desc.endswith("."):
                    desc = desc.rstrip(".,;: ") + "."
                # Sanitize any placeholders
                if _PLACEHOLDER_PATTERNS.search(name) or _PLACEHOLDER_PATTERNS.search(desc):
                    combined = f"{name} — {desc}"
                    combined = await sanitize_placeholder_words(combined, role=target_role)
                    parts = combined.split(" — ", 1)
                    if len(parts) == 2:
                        name, desc = parts[0].strip(), parts[1].strip()
                if name and desc and len(desc.split()) >= 10:
                    result.append({"name": name, "description": desc})
                    log_event(f"🔨 [PROJECT] {name} → {p.get('tools_used', [])}")
            if len(result) >= 2:
                return result[:2]
            log_event(f"⚠️ [PROJECT] Only {len(result)} projects, retrying...")
        except Exception as e:
            log_event(f"⚠️ [PROJECT] Attempt failed: {e}")
    return []


def inject_projects_section(tex: str, projects: List[Dict[str, str]]) -> str:
    """v2.5.0: Inject projects as '\\textbf{Title} -- one-liner' format,
    placed after Experience section, before Achievements section."""
    if not projects:
        return tex

    # Build project items in "Title — description" format
    items = []
    for p in projects[:2]:
        name = latex_escape_text(p.get("name", "Project"))
        desc = latex_escape_text(p.get("description", ""))
        items.append(f"    \\resumeItem{{\\textbf{{{name}}} -- {desc}}}")

    items_tex = "\n".join(items)
    projects_block = (
        "%-----------PROJECTS-----------\n"
        "\\section{Projects}\n"
        "  {\\small\n"
        "  \\resumeItemListStart\n"
        + items_tex + "\n"
        + "  \\resumeItemListEnd\n"
        "  }\n"
    )

    # Remove existing Projects section if present
    proj_pat = section_rx("Projects")
    m = proj_pat.search(tex)
    if m:
        tex = tex[:m.start()] + tex[m.end():]

    # Insert after Experience, before Achievements/Skills
    # Priority: after Experience → before Achievements → before Skills → before \end{document}
    achievement_anchors = [
        r"%-----------ACHIEVEMENTS",
        r"%-----------AWARDS",
        r"%-----------HONORS",
        r"\\section{Achievements",
        r"\\section{Awards",
        r"\\section{Honors",
    ]
    for anchor in achievement_anchors:
        am = re.search(re.escape(anchor) if anchor.startswith("%-") else anchor, tex, re.I)
        if am:
            return tex[:am.start()] + projects_block + "\n" + tex[am.start():]

    # Fallback: before Skills
    skills_pat = re.compile(
        r"(%-----------TECHNICAL SKILLS-----------|\\section\*?\{\s*Skills\s*\})", re.I)
    sm = skills_pat.search(tex)
    if sm:
        return tex[:sm.start()] + projects_block + "\n" + tex[sm.start():]

    # Last fallback: before \end{document}
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
    section = m.group(1)
    items = find_resume_items(section)
    if not items:
        return tex, set()
    uncovered_tasks = [t for t in jd_tasks if t.get("priority") in ("medium", "low")][:len(items)]
    unused_kws = [k for k in all_keywords if k.lower() not in used_keywords][:10]
    ak = role_archetype.get("key", "general_tech")
    bf = role_archetype.get("bullet_focus", "")
    prompt = f"""Rewrite these {len(items)} project bullets to better align with this JD.
Each project MUST be in the format: "\\textbf{{Project Name}} -- one-line description with specific result."
TARGET ROLE: {role_archetype.get('name', 'Technical Role')} ({ak}) | Focus: {bf}
JD (first 2000 chars): {jd_text[:2000]}
CURRENT PROJECT BULLETS: {json.dumps([strip_all_macros_keep_text(section[i[1] + 1:i[2]])[:150] for i in items])}
UNUSED JD KEYWORDS: {', '.join(unused_kws[:8])}
JD TASKS: {json.dumps([t['task_description'] for t in uncovered_tasks[:len(items)]])}
Rewrite each: format "ProjectName -- description", 18-28 words in description, past tense,
specific results with diverse metric types (counts, from-to, time, throughput — not just %),
numbers end in odd digits. No placeholder words (XYZ, ABC, Foo, etc).
Return STRICT JSON: {{"bullets": ["\\\\textbf{{Name}} -- description", ...], "keywords_used": ["kw1", ...]}}"""
    try:
        data = await gpt_json(prompt, temperature=0.3)
        new_bullets = data.get("bullets", [])
        proj_used = set(str(k).lower() for k in (data.get("keywords_used") or []))
        if len(new_bullets) != len(items):
            return tex, set()

        cleaned_bullets = []
        for idx in range(len(new_bullets)):
            b = str(new_bullets[idx]).strip()

            # Keep LaTeX macros like \textbf{...} untouched.
            # Do NOT pass full LaTeX through GPT capitalization fixer.
            # Do NOT latex_escape_text the full string.

            if not b.endswith("."):
                b = b.rstrip(".,;: ") + "."

            cleaned_bullets.append(b)

        replacements = cleaned_bullets[:len(items)]
        if len(replacements) < len(items):
            replacements += [None] * (len(items) - len(replacements))

        new_section_text = section
        for idx in range(len(items) - 1, -1, -1):
            if idx < len(replacements) and replacements[idx] is not None:
                s, ob, cb, e = items[idx]
                new_section_text = (
                    new_section_text[:ob + 1]
                    + replacements[idx]
                    + new_section_text[cb:]
                )

        result = tex[:m.start()] + new_section_text + tex[m.end():]
        log_event(f"✅ [PROJECTS] Rewrote {len(cleaned_bullets)} bullets")
        return result, proj_used

    except Exception as e:
        log_event(f"⚠️ [PROJECTS] Rewrite failed: {e}")
        return tex, set()


# ═══════════════════════════════════════════════════════════════
# PDF / TRIM HELPERS — v2.4.0+: enforce min 9 experience bullets
# ═══════════════════════════════════════════════════════════════

MIN_EXPERIENCE_BULLETS = 9

def _pdf_page_count(pdf: Optional[bytes]) -> int:
    if not pdf or len(pdf) < 10:
        return 0
    for m in re.finditer(rb"/Type\s*/Pages\b", pdf):
        cm = re.search(rb"/Count\s+(\d+)", pdf[m.start():m.start() + 512])
        if cm:
            c = int(cm.group(1))
            if c > 0:
                return c
    ac = [int(c) for c in re.findall(rb"/Count\s+(\d+)", pdf)]
    if ac and max(ac) > 0:
        return max(ac)
    lp = re.findall(rb"/Type\s*/Page(?!\s*/Pages)\b(?=[\s/\]>])", pdf)
    if lp:
        return len(lp)
    mb = len(re.findall(rb"/MediaBox\s*\[", pdf))
    if mb > 0:
        return mb
    return 2 if len(pdf) / 1024 > 134 else 1


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
        if not m:
            continue
        full = m.group(1)
        items = find_resume_items(full)
        if not items:
            continue
        s, _, _, e = items[-1]
        ns = full[:s] + full[e:]
        if not find_resume_items(ns):
            return tex[:m.start()] + tex[m.end():], True
        return tex[:m.start()] + ns + tex[m.end():], True
    return tex, False


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
            if len(items) < 2:
                continue
            if sec_name == "Experience":
                total_exp = _count_experience_bullets(tex)
                if total_exp <= MIN_EXPERIENCE_BULLETS:
                    log_event(f"🛡️ [TRIM] Skipping Experience — {total_exp} bullets ≤ {MIN_EXPERIENCE_BULLETS} minimum")
                    continue
            for idx, (s, ob, cb, e) in enumerate(items):
                bullet_text = full[ob + 1:cb]
                score = score_bullet_relevance(bullet_text, all_keywords)
                if sec_name == "Projects":
                    score += 0.05
                candidates.append((score, match, items, idx, full, sec_name))
    if not candidates:
        return tex, False
    candidates.sort(key=lambda x: x[0])
    score, match, items, idx, full, sec_name = candidates[0]
    s, ob, cb, e = items[idx]
    new_full = full[:s] + full[e:]
    result = tex[:match.start()] + new_full + tex[match.end():]
    log_event(f"✂️ [TRIM] Removed bullet (score={score:.2f}) from {sec_name}")
    return result, True


def compute_coverage(tex: str, keywords: List[str]) -> Dict[str, Any]:
    plain = strip_all_macros_keep_text(tex).lower()
    present = sorted({k.lower() for k in keywords if isinstance(k, str) and k.lower() in plain})
    missing = sorted({k.lower() for k in keywords if isinstance(k, str) and k.lower() not in plain})
    total = max(1, len(present) + len(missing))
    return {"ratio": len(present) / total, "present": present, "missing": missing, "total": total}


# ═══════════════════════════════════════════════════════════════
# SECTION SIZE REDUCER — v2.4.0+: add \small to all sections
# ═══════════════════════════════════════════════════════════════

def apply_small_to_sections(tex: str) -> str:
    section_names = [
        "Experience", "Education", "Projects", "Skills",
        "Achievements", "Achievements & Leadership", "Awards",
        "Certifications", "Publications", "Honors",
    ]
    for name in section_names:
        pat = section_rx(name)
        for m in pat.finditer(tex):
            section_text = m.group(1)
            if r"\small" in section_text[:80]:
                continue
            header_match = re.match(r"(\\section\*?\{[^}]*\})", section_text)
            if not header_match:
                continue
            header = header_match.group(1)
            rest = section_text[len(header):]
            new_section = header + "\n{\\small" + rest + "}\n"
            tex = tex[:m.start()] + new_section + tex[m.end():]
            break
    return tex


# ═══════════════════════════════════════════════════════════════
# ADDITIONAL INFO SECTION — NEW in v2.5.0
# ═══════════════════════════════════════════════════════════════

def inject_additional_info_section(
    tex: str, skills_list: List[str], courses: List[str],
    target_role: str, target_company: str,
) -> str:
    """v2.5.0: Add an 'Additional Information' section with skill hashtags
    and relevant coursework for enhanced metadata visibility."""
    # Build skill hashtags (top 15 most relevant)
    skill_tags = [f"\\#{_ensure_cap(s).replace(' ', '')}" for s in skills_list[:15] if s]
    tags_str = " ".join(skill_tags)

    # Build coursework string (top 6)
    course_str = ", ".join(latex_escape_text(c) for c in courses[:6] if c)

    info_lines = []
    if tags_str:
        info_lines.append(f"\\item \\textbf{{Key Skills:}} \\small{{{tags_str}}}")
    if course_str:
        info_lines.append(f"\\item \\textbf{{Relevant Coursework:}} \\small{{{course_str}}}")

    if not info_lines:
        return tex

    info_block = (
        "%-----------ADDITIONAL INFORMATION-----------\n"
        "\\section{Additional Information}\n"
        "{\\small\n"
        "\\begin{itemize}[leftmargin=0.15in, label={}]\n"
        + "\n".join(f"  {line}" for line in info_lines) + "\n"
        "\\end{itemize}\n"
        "}\n"
    )

    # Check if Additional Information section already exists
    addl_pat = section_rx("Additional Information")
    m = addl_pat.search(tex)
    if m:
        return tex[:m.start()] + info_block + tex[m.end():]

    # Insert before \end{document}
    end_doc = tex.rfind(r"\end{document}")
    if end_doc >= 0:
        return tex[:end_doc] + info_block + "\n" + tex[end_doc:]
    return tex


# ═══════════════════════════════════════════════════════════════
# MAIN OPTIMIZER v2.5.0
# ═══════════════════════════════════════════════════════════════

async def optimize_resume(
    base_tex: str, jd_text: str, target_company: str, target_role: str,
    extra_keywords: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    log_event("🟦 [OPTIMIZE] v2.5.0 — metadata, placeholders, metric diversity, project titles")
    clear_skill_validation_cache()

    jd_snippet = jd_text[:500]

    # 1) Keywords
    jd_info = await extract_keywords_with_priority(jd_text)

    # 2) Role archetype + tone
    role_archetype = await classify_role_and_tone(jd_text, target_role)

    # 3) JD tasks
    jd_tasks = await decompose_jd_into_tasks(jd_text, target_company, target_role, role_archetype)

    # 4) JD key phrases
    jd_phrases = await extract_jd_key_phrases(jd_text)

    # 5) Company core
    company_core = await extract_company_core_requirements(target_company, target_role, jd_text)
    core_keywords = await fix_capitalization_batch(
        [str(k).strip() for k in (company_core.get("core_keywords") or []) if str(k).strip()])

    # 6) Ideal candidate
    ideal_candidate = await profile_ideal_candidate(jd_text, target_company, target_role)

    # 7) Validate keywords with JD context
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

    # 7b) Extract ALL JD skills including soft skills
    all_jd_skills = await extract_all_jd_skills(jd_text)

    # 8) Extra keywords
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

    # 9) Experience companies
    exp_companies = await _extract_experience_companies(base_tex)

    # 10) Master plan
    master_plan = await plan_all_12_bullets(
        jd_text, target_company, target_role, jd_tasks, all_keywords,
        ideal_candidate, role_archetype, exp_companies)

    # 11) Coursework
    courses = await extract_coursework_gpt(jd_text, 24)

    # 12) Split
    preamble, body = _split_preamble_body(base_tex)

    # 13) Coursework
    body = replace_relevant_coursework_distinct(body, courses, 8)

    # 14) Experience
    body, exp_used = await rewrite_experience_section(
        body, jd_text, jd_info, target_company, target_role,
        core_keywords, master_plan, role_archetype, all_keywords)

    # 15) Generate 2 JD-seeded projects (v2.5.0: Title — one-liner format)
    log_event("🔨 [PROJECTS] Generating 1-2 JD-mirrored projects (title — description)...")
    project_entries = await generate_jd_projects(
        jd_text, jd_tasks, role_archetype, jd_info.get("must_have", []), target_role)
    body = inject_projects_section(body, project_entries[:2])
    log_event(f"📁 [PROJECTS] Injected {len(project_entries[:2])} projects")

    # 15b) Rewrite pre-existing project bullets
    body, proj_used = await rewrite_projects_section(
        body, jd_text, jd_tasks, role_archetype, all_keywords, exp_used)
    exp_used.update(proj_used)

    # 16) ATS self-simulation
    log_event("🤖 [ATS SIM] Running ATS self-simulation pass...")
    ats_extra = await ats_self_simulation_pass(
        body, jd_text, all_keywords, jd_info.get("must_have", []))
    ats_kw_set = {x.lower() for x in all_keywords}
    for k in ats_extra:
        if k.lower() not in ats_kw_set:
            all_keywords.append(k)
            ats_kw_set.add(k.lower())

# 17) Skills section — ranked by JD relevance, CAPPED to avoid page overflow (v2.5.1)
    skills_raw, seen = [], set()

    def _add(lst):
        for k in lst:
            k = (k or "").strip()
            if k and k.lower() not in seen:
                seen.add(k.lower()); skills_raw.append(k)

    # Add in priority order (must-have first, generic last)
    _add(jd_info.get("must_have", []))
    _add(core_keywords)
    _add(jd_info.get("should_have", []))
    _add(jd_info.get("nice_to_have", []))
    _add([fix_skill_capitalization_sync(k) for k in exp_used
          if isinstance(k, str) and k and len(k.split()) <= 4])
    _add(extra_list)
    _add(ats_extra)
    # Only add all_jd_skills that overlap with must/should (not the full dump)
    _high_priority_set = {k.lower() for k in
                          jd_info.get("must_have", []) + jd_info.get("should_have", [])}
    _add([s for s in all_jd_skills if s.lower() in _high_priority_set])

    skills_validated = await filter_valid_skills(skills_raw, jd_snippet)
    if skills_validated:
        skills_validated = await fix_capitalization_batch(skills_validated)

    # Rank and cap — only keep skills that matter to this JD
    skills_list = rank_skills_by_jd_relevance(
        skills_validated,
        must_have=jd_info.get("must_have", []),
        should_have=jd_info.get("should_have", []),
        nice_to_have=jd_info.get("nice_to_have", []),
        core_keywords=core_keywords,
        jd_text=jd_text,
        max_skills=MAX_SKILLS,
    )

    body = await replace_skills_section(body, skills_list, jd_text)
    log_event(f"📋 [SKILLS] {len(skills_raw)} raw → {len(skills_validated)} valid → {len(skills_list)} final (cap={MAX_SKILLS})")

    # 18) Apply \small to all sections for one-page fit
    body = apply_small_to_sections(body)

    # 19) Merge
    final_tex = _merge_tex(preamble, body)

    # 20) v2.5.0: Inject PDF metadata (author, keywords, coursework, creator)
    log_event("📄 [METADATA] Injecting PDF metadata...")
    final_tex = inject_pdf_metadata(
        final_tex, target_company, target_role, skills_list, courses)

    # 21) Coverage
    coverage = compute_coverage(final_tex, all_keywords)

    # 22) Phrase mirroring report
    phrases_present, phrases_missing = check_phrase_coverage(final_tex, jd_phrases)
    log_event(f"🔤 [PHRASE MIRROR] {len(phrases_present)}/{len(jd_phrases)} JD phrases present")
    log_event(f"📊 [COVERAGE] {coverage['ratio']:.1%}")

    # 23) Log experience bullet count
    exp_bullet_count = _count_experience_bullets(final_tex)
    log_event(f"📝 [EXP BULLETS] {exp_bullet_count} experience bullets (min={MIN_EXPERIENCE_BULLETS})")

    # Extract project bullet strings for response
    project_bullet_strs = [
        f"{p.get('name', '')} -- {p.get('description', '')}" for p in project_entries
    ]

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
        "specific_technologies_used": list(_used_specific_technologies),
        "project_bullets_generated": project_bullet_strs,
        "project_entries": project_entries,
        "experience_bullet_count": exp_bullet_count,
        "courses": courses,
        "skills_breakdown": {
            "must": len(must_v), "core": len(core_keywords),
            "should": len(should_v), "nice": len(nice_v),
            "extra": len(extra_list), "ats_sim": len(ats_extra),
            "jd_all_skills": len(all_jd_skills),
            "total": len(skills_list),
        },
        "pdf_metadata": {
            "author": "Sri Akash Kadali",
            "creator": "Sri Akash Kadali",
            "keywords_count": len(skills_list),
            "courses_in_subject": len(courses),
        },
    }


# ═══════════════════════════════════════════════════════════════
# API ENDPOINT
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

        target_company, target_role = await extract_company_role(jd_text)
        sc, sr = safe_filename(target_company), safe_filename(target_role)

        optimized_tex, info = await optimize_resume(
            raw_tex, jd_text, target_company, target_role, extra_keywords)

        cur_tex = optimized_tex
        resume_keywords = info.get("all_keywords", [])

        def _compile(t):
            r = render_final_tex(t)
            try:
                result = compile_latex_safely(r)
            except Exception as exc:
                Path(f"/tmp/debug_{sc}_{sr}.tex").write_text(r, encoding="utf-8")
                raise HTTPException(500, f"LaTeX failed: {exc}")
            if not result:
                raise HTTPException(500, "LaTeX empty output.")
            return result

        cur_pdf = _compile(cur_tex)
        trims, streak, prev = 0, 0, len(cur_pdf)

        skills_list = info.get("skills_list", [])
        _skills_already_trimmed = False

        while trims < 60:
            if _pdf_page_count(cur_pdf) <= 1:
                break

            exp_count = _count_experience_bullets(cur_tex)

            nt, ok = remove_one_achievement_bullet(cur_tex)
            if not ok:
                if exp_count > MIN_EXPERIENCE_BULLETS:
                    nt, ok = remove_least_relevant_bullet(
                        cur_tex, resume_keywords, ("Experience", "Projects"))
                else:
                    nt, ok = remove_least_relevant_bullet(
                        cur_tex, resume_keywords, ("Projects",))

            # v2.5.1: If no more bullets to trim, shrink the skills list
            if not ok and not _skills_already_trimmed and len(skills_list) > MAX_SKILLS_TIGHT:
                trimmed_skills = rank_skills_by_jd_relevance(
                    skills_list,
                    must_have=info.get("jd_info", {}).get("must_have", []),
                    should_have=info.get("jd_info", {}).get("should_have", []),
                    nice_to_have=info.get("jd_info", {}).get("nice_to_have", []),
                    core_keywords=info.get("jd_info", {}).get("all_keywords", [])[:15],
                    jd_text=jd_text,
                    max_skills=MAX_SKILLS_TIGHT,
                )
                new_skills_tex = render_skills_section_flat(trimmed_skills)
                skills_pat = re.compile(
                    r"(\\section\*?\{Skills\}[\s\S]*?)(?=%-----------|\\section\*?\{|\\end\{document\})",
                    re.I)
                if re.search(skills_pat, cur_tex):
                    nt = re.sub(skills_pat, lambda _: new_skills_tex + "\n", cur_tex)
                    ok = True
                    _skills_already_trimmed = True
                    skills_list = trimmed_skills
                    log_event(f"✂️ [TRIM-SKILLS] Reduced skills to {len(trimmed_skills)} (cap={MAX_SKILLS_TIGHT})")

            if not ok:
                log_event(f"🛡️ [TRIM] No more removable content (exp={exp_count})")
                break
            try:
                np = _compile(nt)
            except HTTPException:
                break
            trims += 1
            ns = len(np)
            if ns >= prev:
                streak += 1
                if streak >= 4:
                    cur_tex, cur_pdf = nt, np; break
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

        phrase_cov = info.get("jd_phrase_coverage", {})

        final_exp_count = _count_experience_bullets(cur_tex)
        log_event(f"📝 [FINAL] {final_exp_count} experience bullets after trimming")

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
            "tex_string": render_final_tex(cur_tex),
            "pdf_base64": base64.b64encode(cur_pdf).decode("ascii"),
            "coverage_ratio": ratio, "coverage_present": matched, "coverage_missing": missing,
            "trim_summary": {"items_removed": trims, "final_pages": _pdf_page_count(cur_pdf),
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
            "skills_list": info.get("skills_list", []),
            "skills_breakdown": info.get("skills_breakdown", {}),
            "ats_extra_keywords": info.get("ats_extra_keywords", []),
            "project_bullets_generated": info.get("project_bullets_generated", []),
            "project_entries": info.get("project_entries", []),
            "experience_bullet_count": final_exp_count,
            "pdf_metadata": info.get("pdf_metadata", {}),
            "courses": info.get("courses", []),
        })
    except HTTPException:
        raise
    except Exception as e:
        log_event(f"💥 [PIPELINE] Failed: {e}")
        raise HTTPException(500, str(e))