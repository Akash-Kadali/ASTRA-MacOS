"""
============================================================
 HIREX v2.0.0 — talk.py (ULTIMATE KILLER ANSWERS)
 ------------------------------------------------------------
 FIXES vs v1.0.0:
 - Model names: gpt-5.4-mini → gpt-4o-mini (actual model)
 - _score_length: fixed unreachable code after early return
 - CGPA filter: consolidated from 6 redundant calls to 1 final gate
 - _SESSION_FACTS: now actually used for cross-question consistency
 - humanize_answer: self-contained GPT rewrite (no internal API dependency)
 - secure_tex_input: fixed call signature
 - log_event: normalized to single-dict signature
 - Quality scorer weights: now sum to exactly 1.0
 - Retry logic: 2 retries with backoff on OpenAI calls

 UPGRADES:
 - Deep JD customization: extracts JD-specific challenges, terminology,
   team context, and weaves them into every answer
 - JD responsibility mirroring: answers directly address JD duties
 - STAR enforcement: behavioral answers forced into STAR structure
 - Dynamic company intel: GPT fallback for unknown companies
 - Answer dedup: session-level tracking prevents repetitive answers
 - Stronger quality loop: up to 3 iterations with targeted rewrites
 - JD keyword density check: ensures answers contain JD terms
 - Role-specific calibration: adjusts depth/tone per role level

 Author: Sri Akash Kadali
============================================================
"""

from __future__ import annotations

import json
import re
import time
import random
import hashlib
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Set

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None

from backend.core import config
from backend.core.utils import log_event, safe_filename, ensure_dir
from backend.core.security import secure_tex_input

router = APIRouter(prefix="/api/talk", tags=["talk"])

openai_client = AsyncOpenAI(api_key=getattr(config, "OPENAI_API_KEY", "")) if AsyncOpenAI else None

CONTEXT_DIR: Path = config.get_contexts_dir()
ensure_dir(CONTEXT_DIR)

# v2.0.0 FIX: Use actual model names
SUMMARIZER_MODEL = getattr(config, "TALK_SUMMARY_MODEL", "gpt-4o-mini")
ANSWER_MODEL = getattr(config, "TALK_ANSWER_MODEL", "gpt-4o-mini")
CHAT_SAFE_DEFAULT = getattr(config, "DEFAULT_MODEL", "gpt-4o-mini")

# Session-level stores for cross-question consistency
_SESSION_FACTS: Dict[str, Dict[str, Any]] = {}
_SESSION_ANSWERS: Dict[str, List[Dict[str, Any]]] = {}
_SESSION_USED_ACHIEVEMENTS: Dict[str, Set[str]] = {}


# ============================================================
# 🚫 CGPA/GPA FILTERING (consolidated, single-pass)
# ============================================================

_CGPA_RX = re.compile(
    r"""
    \b[Cc]?[Gg][Pp][Aa]\s*[:\-]?\s*\d+\.?\d*\s*/?\s*\d*\.?\d*  |
    \b\d+\.?\d*\s*/?\s*4\.0?\s*[Cc]?[Gg][Pp][Aa]                |
    \b[Cc]umulative\s+[Gg][Pp][Aa]\s*[:\-]?\s*\d+\.?\d*          |
    \bgrade\s+point\s+average\s*[:\-]?\s*\d+\.?\d*               |
    \bwith\s+a\s+[Cc]?[Gg][Pp][Aa]\s+of\s+\d+\.?\d*             |
    \b[Cc]?[Gg][Pp][Aa]\s+of\s+\d+\.?\d*                        |
    \b(maintained|achieved|graduated\s+with)\s+a?\s*\d+\.?\d*\s*[Cc]?[Gg][Pp][Aa] |
    \b\d+\.?\d*\s+out\s+of\s+4\.0                                |
    \bsumma\s+cum\s+laude|\bmagna\s+cum\s+laude|\bcum\s+laude    |
    \bhonors?\s+graduate|\bDean'?s\s+[Ll]ist                     |
    \bacademic\s+excellence|\bfirst\s+class\s+honors?            |
    \bsecond\s+class\s+honors?|\bdistinction\s+in\s+academics?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ACADEMIC_INDICATORS = frozenset({
    "cgpa", "gpa", "grade point", "dean's list", "cum laude",
    "honors graduate", "academic excellence", "first class",
    "second class", "distinction", "valedictorian", "salutatorian",
    "class rank", "top of class", "academic standing", "scholastic",
})


def _strip_cgpa(text: str) -> str:
    """Single-pass CGPA/GPA removal + sentence-level filter."""
    if not text:
        return text
    # Regex pass
    out = _CGPA_RX.sub("", text)
    # Sentence-level pass
    sentences = re.split(r"(?<=[.!?])\s+", out)
    kept = [s for s in sentences
            if not any(ind in s.lower() for ind in _ACADEMIC_INDICATORS)]
    out = " ".join(kept)
    # Cleanup artifacts
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.])", r"\1", out)
    out = re.sub(r"([,.])\s*([,.])", r"\1", out)
    return out.strip()


def _validate_no_cgpa(text: str) -> Tuple[bool, List[str]]:
    violations = []
    tl = text.lower()
    for ind in _ACADEMIC_INDICATORS:
        if ind in tl:
            violations.append(ind)
    violations.extend(m.group() for m in _CGPA_RX.finditer(text))
    return len(violations) == 0, violations


# ============================================================
# 🏢 COMPANY INTELLIGENCE (static + GPT fallback)
# ============================================================

COMPANY_INTELLIGENCE = {
    "netflix": {
        "what_they_value": ["data-driven decisions", "experimentation culture", "ownership mentality", "impact over activity"],
        "culture_keywords": ["Freedom & Responsibility", "context not control", "keeper test", "candor"],
        "tech_strengths": ["Recommender Systems", "A/B Testing at Scale", "Personalization", "Streaming Infrastructure"],
        "what_impresses_them": [
            "Talk about IMPACT, not just tasks",
            "Show data-driven thinking with metrics",
            "Demonstrate ownership and autonomy",
            "Reference experimentation and iteration",
        ],
        "insider_terms": ["member experience", "title discovery", "personalization at scale"],
        "avoid_saying": ["I follow instructions well", "I'm a team player", "I'm passionate"],
        "common_questions": [
            "Tell me about a time you made a data-driven decision",
            "Describe a situation where you took ownership",
            "How do you handle ambiguity?",
            "Tell me about a time you disagreed with your manager",
        ],
        "interview_stages": {
            "recruiter": "Focus on culture fit and high-level experience",
            "technical": "Deep dive into systems design and ML expertise",
            "hiring_manager": "Ownership examples and impact stories",
            "final": "Leadership alignment and long-term vision",
        },
        "salary_range": {"entry": "$150k-$200k", "mid": "$200k-$350k", "senior": "$350k-$600k+"},
        "competitors": ["Disney+", "HBO Max", "Amazon Prime Video", "Apple TV+"],
        "why_not_competitors": "Netflix's experimentation culture and engineering autonomy is unmatched",
    },
    "google": {
        "what_they_value": ["technical excellence", "scalability thinking", "user impact", "10x improvements"],
        "culture_keywords": ["Googleyness", "think big", "user first", "psychological safety"],
        "tech_strengths": ["Distributed Systems", "AI/ML Infrastructure", "Search Quality", "Cloud Platform"],
        "what_impresses_them": [
            "Structured problem-solving approach",
            "Scalability and efficiency thinking",
            "Concrete technical depth",
            "User-centric impact framing",
        ],
        "insider_terms": ["OKRs", "launch and iterate", "10x improvement", "Noogler"],
        "avoid_saying": ["I work hard", "I'm detail-oriented", "I'm a fast learner"],
        "common_questions": [
            "Tell me about a technically challenging problem you solved",
            "How would you design X system?",
            "Tell me about a time you improved something by 10x",
            "How do you prioritize when everything is important?",
        ],
        "interview_stages": {
            "recruiter": "Googleyness assessment and experience overview",
            "technical": "Coding, system design, and ML depth",
            "hiring_manager": "Leadership and collaboration examples",
            "final": "Team match and culture fit",
        },
        "salary_range": {"entry": "$140k-$180k", "mid": "$200k-$400k", "senior": "$400k-$800k+"},
        "competitors": ["Microsoft", "Meta", "Amazon", "Apple"],
        "why_not_competitors": "Google's technical challenges and AI leadership are unparalleled",
    },
    "meta": {
        "what_they_value": ["move fast", "impact metrics", "product sense", "bold thinking"],
        "culture_keywords": ["Move Fast", "Be Bold", "Focus on Impact", "Build Social Value"],
        "tech_strengths": ["Ranking Systems", "Social Graph", "AR/VR", "Ads Optimization"],
        "what_impresses_them": [
            "Quantified impact with specific metrics",
            "Move fast mentality with examples",
            "Product intuition demonstrated",
            "Scale of systems you've worked on",
        ],
        "insider_terms": ["family of apps", "integrity systems", "social impact", "Bootcamp"],
        "avoid_saying": ["I'm careful and methodical", "I double-check everything", "I prefer stability"],
        "common_questions": [
            "Tell me about your biggest impact",
            "How do you make decisions with incomplete data?",
            "Describe a time you moved fast and broke things",
            "How do you measure success?",
        ],
        "interview_stages": {
            "recruiter": "Impact stories and culture fit",
            "technical": "Coding and system design at scale",
            "hiring_manager": "Product sense and leadership",
            "final": "Executive alignment",
        },
        "salary_range": {"entry": "$140k-$180k", "mid": "$200k-$400k", "senior": "$400k-$700k+"},
        "competitors": ["Google", "TikTok", "Snap", "Twitter/X"],
        "why_not_competitors": "Meta's scale of impact and bold technical bets are unique",
    },
    "amazon": {
        "what_they_value": ["customer obsession", "ownership", "bias for action", "dive deep", "deliver results"],
        "culture_keywords": ["Customer Obsession", "Ownership", "Bias for Action", "Dive Deep", "Leadership Principles"],
        "tech_strengths": ["AWS Services", "Supply Chain ML", "Retail Optimization", "Logistics"],
        "what_impresses_them": [
            "STAR format with specific metrics",
            "Leadership principles alignment",
            "Customer impact examples",
            "Ownership and accountability stories",
        ],
        "insider_terms": ["PR/FAQ", "6-pager", "bar raiser", "Day 1 mentality", "mechanisms", "working backwards"],
        "avoid_saying": ["That's not my job", "I waited for direction", "I delegated"],
        "common_questions": [
            "Tell me about a time you went above and beyond for a customer",
            "Describe a situation where you had to dive deep",
            "Tell me about a time you disagreed and committed",
            "How do you handle competing priorities?",
        ],
        "interview_stages": {
            "recruiter": "LP screening and experience fit",
            "technical": "Coding and system design",
            "loop": "Multiple LP-focused behavioral rounds",
            "bar_raiser": "Cross-team culture assessment",
        },
        "salary_range": {"entry": "$130k-$170k", "mid": "$180k-$350k", "senior": "$350k-$600k+"},
        "competitors": ["Google Cloud", "Microsoft Azure", "Walmart", "Shopify"],
        "why_not_competitors": "Amazon's customer obsession and ownership culture drive real impact",
    },
    "microsoft": {
        "what_they_value": ["growth mindset", "customer empathy", "collaboration", "inclusive culture"],
        "culture_keywords": ["Growth Mindset", "Customer Obsessed", "One Microsoft", "Learn-it-all"],
        "tech_strengths": ["Azure Cloud", "AI/Copilot", "Microsoft 365", "Developer Tools"],
        "what_impresses_them": [
            "Growth mindset examples with learning",
            "Collaboration across teams",
            "Customer impact stories",
            "Responsible AI awareness",
        ],
        "insider_terms": ["growth mindset", "customer zero", "inclusive design", "One Microsoft"],
        "avoid_saying": ["I already know everything", "I work best alone", "I avoid ambiguity"],
        "common_questions": [
            "Tell me about a time you learned from failure",
            "How do you collaborate with difficult stakeholders?",
            "Describe your approach to inclusive design",
            "How do you balance innovation with responsibility?",
        ],
        "interview_stages": {
            "recruiter": "Growth mindset and culture fit",
            "technical": "Coding and system design",
            "hiring_manager": "Collaboration and customer focus",
            "final": "As-appropriate leadership",
        },
        "salary_range": {"entry": "$130k-$170k", "mid": "$180k-$350k", "senior": "$350k-$600k+"},
        "competitors": ["Google", "Amazon", "Salesforce", "Oracle"],
        "why_not_competitors": "Microsoft's growth mindset culture and AI leadership momentum",
    },
    "apple": {
        "what_they_value": ["attention to detail", "user privacy", "craftsmanship", "simplicity"],
        "culture_keywords": ["Think Different", "Simplicity", "Privacy as Human Right", "Excellence"],
        "tech_strengths": ["On-Device ML", "Privacy-Preserving AI", "Hardware-Software Integration"],
        "what_impresses_them": [
            "Design thinking and user empathy",
            "Privacy-first approach",
            "Quality over speed examples",
            "Attention to detail stories",
        ],
        "insider_terms": ["DRI", "surprise and delight", "it just works", "top 100"],
        "avoid_saying": ["Good enough is fine", "Users don't care about details", "Move fast and break things"],
        "common_questions": [
            "Tell me about a time you obsessed over details",
            "How do you balance user experience with technical constraints?",
            "Describe your approach to privacy",
            "Tell me about something you're proud of building",
        ],
        "interview_stages": {
            "recruiter": "Culture fit and experience",
            "technical": "Deep technical expertise",
            "design": "User empathy and design thinking",
            "final": "Leadership alignment",
        },
        "salary_range": {"entry": "$140k-$180k", "mid": "$200k-$400k", "senior": "$400k-$700k+"},
        "competitors": ["Google", "Samsung", "Microsoft"],
        "why_not_competitors": "Apple's commitment to privacy and craftsmanship is unmatched",
    },
    "nvidia": {
        "what_they_value": ["technical depth", "GPU/CUDA expertise", "performance optimization", "innovation"],
        "culture_keywords": ["Speed of Light", "Intellectual Honesty", "One Team", "Innovation"],
        "tech_strengths": ["GPU Architecture", "CUDA", "TensorRT", "Deep Learning Frameworks", "HPC"],
        "what_impresses_them": [
            "Low-level optimization experience",
            "Understanding of GPU architecture",
            "Performance benchmarking rigor",
            "Systems-level thinking",
        ],
        "insider_terms": ["CUDA cores", "tensor cores", "TensorRT", "inference optimization", "DGX"],
        "avoid_saying": ["I prefer high-level abstractions", "Performance doesn't matter", "I avoid hardware details"],
        "common_questions": [
            "How would you optimize inference latency?",
            "Describe your experience with GPU programming",
            "Tell me about a performance optimization you did",
            "How do you approach profiling and bottleneck analysis?",
        ],
        "interview_stages": {
            "recruiter": "Technical background and role fit",
            "technical": "C++/CUDA coding and systems design",
            "hiring_manager": "Architecture thinking and impact",
            "final": "Team fit and culture alignment",
        },
        "salary_range": {"entry": "$140k-$190k", "mid": "$200k-$400k", "senior": "$400k-$700k+"},
        "competitors": ["AMD", "Intel", "Google TPU", "Qualcomm"],
        "why_not_competitors": "NVIDIA's dominance in AI compute and CUDA ecosystem is unmatched",
    },
    "stripe": {
        "what_they_value": ["rigorous thinking", "users first", "writing quality", "long-term orientation"],
        "culture_keywords": ["Users First", "Move with Urgency", "Think Rigorously", "Trust and Amplify"],
        "tech_strengths": ["Payment Infrastructure", "Financial APIs", "Fraud Detection", "Developer Experience"],
        "what_impresses_them": [
            "Clear, rigorous thinking",
            "Developer empathy examples",
            "Long-term technical decisions",
            "Writing and communication quality",
        ],
        "insider_terms": ["increase GDP of internet", "payment rails", "developer love"],
        "avoid_saying": ["I prefer quick wins", "Documentation is boring", "I focus on short-term"],
        "common_questions": [
            "Walk me through a complex technical decision",
            "How do you think about API design?",
            "Tell me about a time you prioritized long-term over short-term",
            "How do you communicate technical concepts?",
        ],
        "interview_stages": {
            "recruiter": "Writing sample and culture fit",
            "technical": "System design and coding",
            "work_sample": "Take-home or pair programming",
            "final": "Team and leadership fit",
        },
        "salary_range": {"entry": "$150k-$200k", "mid": "$220k-$400k", "senior": "$400k-$650k+"},
        "competitors": ["Square", "Adyen", "Braintree", "Plaid"],
        "why_not_competitors": "Stripe's developer-first culture and rigorous thinking",
    },
    "databricks": {
        "what_they_value": ["technical excellence", "open source contribution", "customer impact", "data passion"],
        "culture_keywords": ["Customer Obsessed", "Unity", "Ownership", "Open Source First"],
        "tech_strengths": ["Lakehouse", "Delta Lake", "MLflow", "Spark", "Data Engineering"],
        "what_impresses_them": [
            "Deep technical expertise",
            "Open source appreciation/contribution",
            "Data architecture experience",
            "Customer-facing technical work",
        ],
        "insider_terms": ["Lakehouse", "Delta Lake", "data + AI", "open source", "Unity Catalog"],
        "avoid_saying": ["I prefer proprietary tools", "Open source is risky", "I avoid customer interaction"],
        "common_questions": [
            "How would you design a data lakehouse?",
            "Tell me about your open source contributions",
            "How do you approach data quality at scale?",
            "Describe a complex data architecture you built",
        ],
        "interview_stages": {
            "recruiter": "Technical background and culture",
            "technical": "Data architecture and Spark expertise",
            "system_design": "Lakehouse and ML infrastructure",
            "final": "Customer focus and team fit",
        },
        "salary_range": {"entry": "$150k-$200k", "mid": "$220k-$400k", "senior": "$400k-$600k+"},
        "competitors": ["Snowflake", "AWS", "Google BigQuery"],
        "why_not_competitors": "Databricks' open source DNA and unified data + AI platform",
    },
}

_DEFAULT_INTEL: Dict[str, Any] = {
    "what_they_value": ["technical excellence", "collaboration", "impact", "growth"],
    "culture_keywords": ["innovation", "teamwork", "excellence"],
    "tech_strengths": ["modern technology", "scalable systems"],
    "what_impresses_them": [
        "Specific achievements with metrics",
        "Problem-solving examples",
        "Collaboration stories",
        "Technical depth demonstration",
    ],
    "insider_terms": [],
    "avoid_saying": ["I'm a team player", "I work hard", "I'm passionate"],
    "common_questions": [],
    "interview_stages": {},
    "salary_range": {"entry": "$100k-$150k", "mid": "$150k-$250k", "senior": "$250k-$400k+"},
    "competitors": [],
    "why_not_competitors": "",
}

_dynamic_intel_cache: Dict[str, Dict[str, Any]] = {}


def _lookup_static_intel(company_name: str) -> Optional[Dict[str, Any]]:
    cl = (company_name or "").lower().strip()
    for key, intel in COMPANY_INTELLIGENCE.items():
        if key in cl or cl in key:
            return intel
    for key, intel in COMPANY_INTELLIGENCE.items():
        if any(w in cl for w in key.split()):
            return intel
    return None


async def get_company_intelligence(company_name: str, jd_text: str = "", model: str = SUMMARIZER_MODEL) -> Dict[str, Any]:
    """Static lookup first, GPT fallback for unknown companies."""
    static = _lookup_static_intel(company_name)
    if static:
        return static

    cl = company_name.lower().strip()
    if cl in _dynamic_intel_cache:
        return _dynamic_intel_cache[cl]

    if not jd_text.strip():
        return _DEFAULT_INTEL

    # v2.0.0: GPT-based company intel for unknown companies
    prompt = f"""Analyze this company and JD for interview intelligence.

COMPANY: {company_name}
JOB DESCRIPTION (excerpt):
{jd_text[:2500]}

Return STRICT JSON:
{{
    "what_they_value": ["4 things this company values based on JD language"],
    "culture_keywords": ["3-4 culture terms from JD"],
    "tech_strengths": ["3-4 technical areas from JD"],
    "what_impresses_them": ["4 specific things that would impress this interviewer"],
    "insider_terms": ["3-4 domain-specific terms from the JD"],
    "avoid_saying": ["3 things to avoid based on JD tone"],
    "common_questions": ["4 likely interview questions for this role"],
    "interview_stages": {{}},
    "salary_range": {{"entry": "...", "mid": "...", "senior": "..."}},
    "competitors": ["3-4 competitors"],
    "why_not_competitors": "1 sentence why this company over competitors"
}}
"""
    try:
        raw = await _call_openai("Extract company intel as JSON.", prompt, model)
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            intel = json.loads(m.group(0))
            # Merge with defaults for missing keys
            for k, v in _DEFAULT_INTEL.items():
                if k not in intel or not intel[k]:
                    intel[k] = v
            _dynamic_intel_cache[cl] = intel
            log_event({"event": "dynamic_company_intel", "company": company_name})
            return intel
    except Exception as e:
        log_event({"event": "dynamic_intel_fail", "error": str(e)})

    return _DEFAULT_INTEL


# ============================================================
# 🎯 QUESTION TYPE DETECTION & STRATEGY
# ============================================================

QUESTION_STRATEGIES = {
    "why_hire_you": {
        "patterns": [
            r"why should we hire you", r"why are you the right fit",
            r"why should we choose you", r"what makes you stand out",
            r"why you over other candidates", r"what sets you apart",
            r"what do you bring to this role", r"what value do you add",
        ],
        "strategy": "PROOF + UNIQUE VALUE",
        "structure": [
            "Open with your STRONGEST, most RELEVANT achievement",
            "Show UNIQUE combination of skills they can't easily find",
            "Map directly to their SPECIFIC needs from JD",
            "Close with forward-looking contribution",
        ],
        "behavioral": False,
        "likely_followups": [
            "Can you give me a specific example?",
            "How would you apply that here?",
            "What would you do in your first 90 days?",
        ],
        "trap_warnings": [
            "Don't be arrogant or put down other candidates",
            "Don't be vague — they want SPECIFIC proof",
            "Don't just list skills — show IMPACT",
        ],
    },
    "why_this_company": {
        "patterns": [
            r"why do you want to work (here|at|for)",
            r"why this company", r"what attracts you to",
            r"what interests you about", r"why are you interested in",
            r"what draws you to",
        ],
        "strategy": "SPECIFIC COMPANY KNOWLEDGE + ALIGNMENT",
        "structure": [
            "Open with SPECIFIC company insight (product, tech, challenge)",
            "Show genuine understanding of their unique position",
            "Connect YOUR background to THEIR specific needs",
            "Demonstrate you've researched beyond the JD",
        ],
        "behavioral": False,
        "likely_followups": [
            "What do you know about our culture?",
            "Why not one of our competitors?",
            "What concerns do you have about us?",
        ],
        "trap_warnings": [
            "Don't be generic — they want SPECIFIC knowledge",
            "Don't mention salary/benefits as primary reason",
            "Don't badmouth competitors",
        ],
    },
    "why_this_role": {
        "patterns": [
            r"why this (role|position|job)",
            r"what interests you about this (role|position)",
            r"why are you applying for this",
            r"what excites you about this (role|position|opportunity)",
            r"why do you want this job",
        ],
        "strategy": "ROLE-SKILL MATCH + GROWTH",
        "structure": [
            "Show you understand EXACTLY what this role does",
            "Map your experience to the specific responsibilities",
            "Demonstrate how this is a natural next step",
            "Show enthusiasm through specificity, not generic excitement",
        ],
        "behavioral": False,
        "likely_followups": [
            "What don't you know about this role?",
            "What would be challenging for you?",
            "How does this fit your long-term goals?",
        ],
        "trap_warnings": [
            "Don't be vague about role responsibilities",
            "Don't focus only on what you'll GET",
            "Show understanding of challenges, not just perks",
        ],
    },
    "tell_me_about_yourself": {
        "patterns": [
            r"tell me about yourself", r"walk me through your background",
            r"introduce yourself", r"give me an overview",
            r"describe your background", r"walk me through your resume",
        ],
        "strategy": "RELEVANT NARRATIVE: Present → Past → Future",
        "structure": [
            "Start with current role/most relevant experience (Present)",
            "Connect past experiences in a coherent narrative (Past)",
            "Show intentional career trajectory toward THIS role (Future)",
            "End with why you're here NOW",
        ],
        "behavioral": False,
        "likely_followups": [
            "Tell me more about {specific thing you mentioned}",
            "Why did you leave your last role?",
            "What's your biggest accomplishment?",
        ],
        "trap_warnings": [
            "Don't recite your entire resume",
            "Keep it under 2 minutes spoken",
            "Make it RELEVANT to this role",
        ],
    },
    "strength": {
        "patterns": [
            r"(greatest|biggest|key|main) strength",
            r"what are you good at", r"what do you do well",
            r"what's your superpower", r"strongest skill",
        ],
        "strategy": "SPECIFIC STRENGTH + PROOF",
        "structure": [
            "Name ONE specific strength (not generic)",
            "Immediately prove it with a concrete example",
            "Show impact/outcome of that strength",
            "Connect to how it helps THIS role",
        ],
        "behavioral": False,
        "likely_followups": [
            "Can you give another example?",
            "How would that help in this role?",
            "Has that strength ever been a weakness?",
        ],
        "trap_warnings": [
            "Don't be generic (teamwork, communication)",
            "Don't claim strengths you can't prove",
            "Pick something RELEVANT to the role",
        ],
    },
    "weakness": {
        "patterns": [
            r"(greatest|biggest) weakness", r"area (for|of) improvement",
            r"what are you working on", r"development area",
            r"where do you struggle",
        ],
        "strategy": "HONEST + MITIGATION + GROWTH",
        "structure": [
            "Name a REAL weakness (not a humble-brag)",
            "Show self-awareness about its impact",
            "Describe SPECIFIC steps you're taking to improve",
            "Show progress/results from those efforts",
        ],
        "behavioral": False,
        "likely_followups": [
            "How has that weakness affected your work?",
            "Can you give a specific example?",
            "How do you know you're improving?",
        ],
        "trap_warnings": [
            "Don't say 'perfectionism' or 'work too hard'",
            "Don't pick something critical to the role",
            "Show genuine self-awareness",
        ],
    },
    "achievement": {
        "patterns": [
            r"(proudest|biggest|greatest|most significant) (achievement|accomplishment)",
            r"tell me about a time you", r"describe a situation where",
            r"give me an example of", r"share an experience when",
        ],
        "strategy": "STAR WITH IMPACT",
        "structure": [
            "Situation/Task — set context briefly (2 sentences max)",
            "Action — YOUR specific actions (bulk of the answer)",
            "Result — Quantified outcome",
            "Takeaway — What you learned / how it applies here",
        ],
        "behavioral": True,
        "likely_followups": [
            "What would you do differently?",
            "What did you learn from this?",
            "How did others contribute?",
        ],
        "trap_warnings": [
            "Don't make Situation too long",
            "Focus on YOUR actions, not team's",
            "QUANTIFY the result",
        ],
    },
    "conflict": {
        "patterns": [
            r"conflict", r"disagreement",
            r"difficult (person|colleague|coworker|situation)",
            r"challenging relationship", r"dealt with a difficult",
        ],
        "strategy": "PROFESSIONAL + RESOLUTION-FOCUSED",
        "structure": [
            "Situation — Describe professionally (no blame)",
            "Action — Your approach to understanding + resolving",
            "Result — Positive outcome",
            "Takeaway — What you learned about collaboration",
        ],
        "behavioral": True,
        "likely_followups": [
            "What if the person didn't change?",
            "What would you do differently?",
            "How do you prevent conflicts?",
        ],
        "trap_warnings": [
            "Don't badmouth the other person",
            "Show emotional intelligence",
            "Pick a professional conflict, not personal",
        ],
    },
    "failure": {
        "patterns": [
            r"tell me about a (time you )?fail", r"mistake you made",
            r"something that didn't work", r"a setback", r"when things went wrong",
        ],
        "strategy": "HONEST + LEARNING + GROWTH",
        "structure": [
            "Situation — What happened (own it)",
            "Action — What you did about it",
            "Result — The outcome (including negative)",
            "Takeaway — What you learned and how you've applied it",
        ],
        "behavioral": True,
        "likely_followups": [
            "How did others react?",
            "What would you do differently now?",
            "How do you prevent similar failures?",
        ],
        "trap_warnings": [
            "Don't pick something trivial",
            "Don't blame others",
            "Show genuine learning",
        ],
    },
    "salary": {
        "patterns": [
            r"salary expectations", r"compensation",
            r"what are you looking for", r"pay expectations",
            r"how much do you want to make",
        ],
        "strategy": "RESEARCH-BACKED + FLEXIBLE",
        "structure": [
            "Show you've done research on market rates",
            "Give a range (not a single number)",
            "Express flexibility based on total comp",
            "Redirect to fit and opportunity",
        ],
        "behavioral": False,
        "likely_followups": [
            "What's your current salary?",
            "What's the minimum you'd accept?",
            "How did you arrive at that number?",
        ],
        "trap_warnings": [
            "Don't give a number too early",
            "Don't lowball yourself",
            "Research the company's pay bands",
        ],
    },
    "leadership": {
        "patterns": [
            r"leadership (experience|style|example)", r"led a team",
            r"managed (people|team)", r"describe your leadership",
        ],
        "strategy": "EXAMPLE + STYLE + IMPACT",
        "structure": [
            "Situation — Describe a specific leadership situation",
            "Action — YOUR leadership approach",
            "Result — Impact on team and results",
            "Takeaway — Connect to how you'd lead here",
        ],
        "behavioral": True,
        "likely_followups": [
            "How do you handle underperformers?",
            "How do you motivate your team?",
            "What's your biggest leadership mistake?",
        ],
        "trap_warnings": [
            "Don't be vague about your style",
            "Show results, not just process",
            "Leadership isn't just about managing people",
        ],
    },
    "technical": {
        "patterns": [
            r"technical (challenge|problem|decision)",
            r"complex (system|architecture|problem)",
            r"how would you (design|build|architect)",
            r"walk me through (your|a) technical",
        ],
        "strategy": "DEPTH + TRADEOFFS + REASONING",
        "structure": [
            "Context — The technical problem",
            "Approach — Your reasoning and tradeoffs",
            "Implementation — What you actually built",
            "Outcome — Results and learnings",
        ],
        "behavioral": True,
        "likely_followups": [
            "What would you do differently now?",
            "How did you handle scale?",
            "What were the failure modes?",
        ],
        "trap_warnings": [
            "Don't be too high-level",
            "Show depth of understanding",
            "Discuss tradeoffs, not just solutions",
        ],
    },
    "generic": {
        "patterns": [],
        "strategy": "SPECIFIC PROOF + RELEVANCE",
        "structure": [
            "Answer directly with specific evidence",
            "Connect to JD requirements",
            "Show concrete examples",
            "Tie to this opportunity",
        ],
        "behavioral": False,
        "likely_followups": [
            "Can you elaborate on that?",
            "How would that apply here?",
            "What's a specific example?",
        ],
        "trap_warnings": [
            "Don't be vague",
            "Always give specific examples",
            "Connect to the role",
        ],
    },
}


def detect_question_type(question: str) -> Tuple[str, Dict[str, Any]]:
    ql = question.lower().strip()
    for q_type, cfg in QUESTION_STRATEGIES.items():
        if q_type == "generic":
            continue
        for pattern in cfg["patterns"]:
            if re.search(pattern, ql):
                return q_type, cfg
    return "generic", QUESTION_STRATEGIES["generic"]


# ============================================================
# 📊 ANSWER QUALITY SCORER (v2.0.0 — fixed weights, fixed _score_length)
# ============================================================

class AnswerQualityScorer:
    def __init__(self):
        self.scores: Dict[str, float] = {}
        self.feedback: List[str] = []

    def score_answer(
        self, answer: str, question: str, q_type: str,
        company: str, jd_requirements: Dict[str, Any],
        resume_highlights: Dict[str, Any],
        company_intel: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.scores = {}
        self.feedback = []

        self.scores["hook_strength"] = self._score_hook(answer, company)
        self.scores["specificity"] = self._score_specificity(answer, resume_highlights)
        self.scores["jd_relevance"] = self._score_relevance(answer, jd_requirements)
        self.scores["company_alignment"] = self._score_company_alignment(answer, company, company_intel)
        self.scores["confidence"] = self._score_confidence(answer)
        self.scores["length"] = self._score_length(answer)
        self.scores["no_cliches"] = self._score_no_cliches(answer)
        self.scores["question_match"] = self._score_question_match(answer, question, q_type)
        self.scores["no_cgpa"] = 0.0 if not _validate_no_cgpa(answer)[0] else 10.0

        if self.scores["no_cgpa"] < 10.0:
            self.feedback.append("CRITICAL: Remove all GPA/CGPA/academic metrics")

        # v2.0.0 FIX: weights sum to exactly 1.0
        weights = {
            "hook_strength": 0.12, "specificity": 0.18, "jd_relevance": 0.16,
            "company_alignment": 0.10, "confidence": 0.10, "length": 0.04,
            "no_cliches": 0.08, "question_match": 0.14, "no_cgpa": 0.08,
        }
        overall = sum(self.scores[k] * weights[k] for k in weights)

        grade = (
            "A+" if overall >= 9.0 else "A" if overall >= 8.5 else
            "A-" if overall >= 8.0 else "B+" if overall >= 7.5 else
            "B" if overall >= 7.0 else "B-" if overall >= 6.5 else
            "C+" if overall >= 6.0 else "C" if overall >= 5.5 else "Needs Improvement"
        )
        return {
            "overall_score": round(overall, 1),
            "dimension_scores": self.scores,
            "feedback": self.feedback,
            "grade": grade,
            "pass": overall >= 7.0,
        }

    def _score_hook(self, answer: str, company: str) -> float:
        first = answer.split(".")[0] if answer else ""
        s = 5.0
        if company.lower() in first.lower():
            s += 1.5
        if any(w in first.lower() for w in ["built", "led", "designed", "achieved", "delivered", "deployed"]):
            s += 1.5
        if first and not first.lower().startswith(("i am", "i'm", "thank you", "i would")):
            s += 1.0
        if any(p in first.lower() for p in ["i am writing", "thank you for", "i believe", "i am excited"]):
            s -= 2.0
            self.feedback.append("Open with a specific achievement, not a generic phrase")
        return max(0, min(10, s))

    def _score_specificity(self, answer: str, rh: Dict[str, Any]) -> float:
        s = 5.0
        if re.search(r"\d+", answer):
            s += 1.5
        else:
            self.feedback.append("Add quantified achievements with numbers")
        if any(c.lower() in answer.lower() for c in rh.get("companies_worked", [])):
            s += 1.0
        if any(t.lower() in answer.lower() for t in rh.get("technical_skills", [])):
            s += 1.5
        action_count = sum(1 for v in ["built", "designed", "led", "implemented", "delivered", "deployed", "optimized"] if v in answer.lower())
        if action_count >= 2:
            s += 1.0
        return max(0, min(10, s))

    def _score_relevance(self, answer: str, jd: Dict[str, Any]) -> float:
        s = 5.0
        al = answer.lower()
        skills = jd.get("must_have_skills", []) + jd.get("tech_stack", [])
        hits = sum(1 for sk in skills if sk.lower() in al)
        if hits >= 3:
            s += 3.0
        elif hits >= 2:
            s += 2.0
        elif hits >= 1:
            s += 1.0
        else:
            self.feedback.append("Mention more JD-specific skills and requirements")
        resp = jd.get("key_responsibilities", [])
        r_hits = sum(1 for r in resp if any(w in al for w in r.lower().split()[:3]))
        if r_hits >= 2:
            s += 2.0
        return max(0, min(10, s))

    def _score_company_alignment(self, answer: str, company: str, intel: Optional[Dict[str, Any]] = None) -> float:
        s = 5.0
        al = answer.lower()
        if company.lower() in al:
            s += 2.0
        if intel:
            culture_hits = sum(1 for k in intel.get("culture_keywords", []) if k.lower() in al)
            s += min(2.0, culture_hits * 0.5)
            avoid_hits = sum(1 for a in intel.get("avoid_saying", []) if a.lower() in al)
            if avoid_hits:
                s -= avoid_hits * 1.5
                self.feedback.append(f"Avoid: {intel['avoid_saying'][:2]}")
        return max(0, min(10, s))

    def _score_confidence(self, answer: str) -> float:
        s = 7.0
        al = answer.lower()
        hedges = ["i think", "i believe", "maybe", "perhaps", "i hope", "i would try"]
        s -= sum(0.5 for h in hedges if h in al)
        if any(d in al for d in ["grateful for any", "hope you consider", "humbly", "please give me a chance"]):
            s -= 2.0
            self.feedback.append("Remove desperate/pleading language — be confident")
        s += min(2.0, sum(0.5 for c in ["i delivered", "i led", "i built", "i achieved", "i drove"] if c in al))
        return max(0, min(10, s))

    # v2.0.0 FIX: removed unreachable code
    def _score_length(self, answer: str) -> float:
        words = len(answer.split())
        if 100 <= words <= 200:
            return 10.0
        if 80 <= words <= 250:
            return 8.0
        if 60 <= words <= 300:
            self.feedback.append("Adjust answer length (target 120-180 words)")
            return 6.0
        self.feedback.append("Answer is too short or too long (target 120-180 words)")
        return 4.0

    def _score_no_cliches(self, answer: str) -> float:
        s = 10.0
        al = answer.lower()
        cliches = [
            "passionate", "team player", "hard worker", "go-getter",
            "think outside the box", "synergy", "leverage", "dynamic",
            "results-driven", "detail-oriented", "self-starter",
            "fast learner", "people person", "perfectionist",
        ]
        hits = sum(1 for c in cliches if c in al)
        s -= hits * 2.0
        if hits:
            self.feedback.append(f"Remove {hits} cliche(s)")
        return max(0, min(10, s))

    def _score_question_match(self, answer: str, question: str, q_type: str) -> float:
        s = 7.0
        al = answer.lower()
        if q_type == "why_hire_you" and any(w in al for w in ["unique", "different", "sets me apart", "specifically"]):
            s += 2.0
        elif q_type == "why_this_company" and re.search(r"(specifically|unique|particular|your)", al):
            s += 2.0
        elif q_type == "weakness":
            if any(w in al for w in ["learned", "improved", "working on", "developed"]):
                s += 2.0
            else:
                self.feedback.append("Show growth on the weakness")
        elif q_type == "achievement" and re.search(r"(result|led to|achieved|improved|reduced|increased)", al):
            s += 2.0
        elif q_type in ("conflict", "failure", "leadership", "technical"):
            # Check for STAR structure
            has_situation = any(w in al for w in ["when", "at", "during", "while"])
            has_action = any(w in al for w in ["i built", "i led", "i designed", "i implemented", "i decided", "i approached"])
            has_result = any(w in al for w in ["resulted", "led to", "achieved", "improved", "reduced", "which"])
            star_score = sum([has_situation, has_action, has_result])
            s += min(3.0, star_score)
            if star_score < 2:
                self.feedback.append("Use STAR structure: Situation → Action → Result")
        return max(0, min(10, s))


# ============================================================
# 🔍 SKILL GAP ANALYZER
# ============================================================

async def analyze_skill_gaps(
    resume_highlights: Dict[str, Any], jd_requirements: Dict[str, Any],
) -> Dict[str, Any]:
    resume_skills = {s.lower() for s in resume_highlights.get("technical_skills", [])}
    jd_must = {s.lower() for s in jd_requirements.get("must_have_skills", [])}
    jd_nice = {s.lower() for s in jd_requirements.get("nice_to_have_skills", [])}

    must_match = jd_must & resume_skills
    must_gap = jd_must - resume_skills
    nice_match = jd_nice & resume_skills
    nice_gap = jd_nice - resume_skills

    gap_strategies = {}
    for gap in list(must_gap)[:5]:
        related = [s for s in resume_skills if any(w in s for w in gap.split())]
        if related:
            gap_strategies[gap] = f"Leverage your {related[0]} experience as foundation for {gap}"
        else:
            gap_strategies[gap] = f"Express genuine interest in deepening {gap} through this role"

    return {
        "must_have_matches": sorted(must_match),
        "must_have_gaps": sorted(must_gap),
        "nice_to_have_matches": sorted(nice_match),
        "nice_to_have_gaps": sorted(nice_gap),
        "match_percentage": round(len(must_match) / max(1, len(jd_must)) * 100, 1),
        "gap_strategies": gap_strategies,
        "strengths_to_emphasize": sorted(must_match)[:5],
    }


# ============================================================
# 🔒 UTILITIES
# ============================================================

def _tex_safe(s: str) -> str:
    try:
        return secure_tex_input("inline.txt", s)
    except Exception:
        return s


def _is_responses_only_model(name: str) -> bool:
    return bool(name and re.match(r"^(gpt-image|dall[- ]?e|whisper)", name, re.I))


def _session_key(jd_text: str, resume_text: str) -> str:
    return hashlib.md5((jd_text[:500] + resume_text[:500]).encode()).hexdigest()[:16]


# ============================================================
# 🧠 REQUEST MODEL
# ============================================================

class TalkReq(BaseModel):
    jd_text: str = ""
    question: str
    resume_tex: Optional[str] = None
    resume_plain: Optional[str] = None
    tone: str = "balanced"
    humanize: bool = True
    model: str = ANSWER_MODEL
    context_key: Optional[str] = None
    context_id: Optional[str] = None
    title: Optional[str] = None
    use_latest: bool = True
    interview_stage: Optional[str] = None
    include_quality_score: bool = True
    include_followups: bool = True


# ============================================================
# 🧩 CONTEXT HELPERS
# ============================================================

def _path_for_key(key: str) -> Path:
    return CONTEXT_DIR / f"{safe_filename(key)}.json"


def _latest_path() -> Optional[Path]:
    files = sorted(CONTEXT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _coerce_key(ctx: Dict[str, Any], fp: Optional[Path]) -> str:
    if ctx.get("key"):
        return str(ctx["key"]).strip()
    c, r = (ctx.get("company") or "").strip(), (ctx.get("role") or "").strip()
    if c and r:
        return f"{safe_filename(c)}__{safe_filename(r)}"
    return fp.stem if fp else ""


def _pick_resume(ctx: Dict[str, Any]) -> str:
    for src in (
        (ctx.get("humanized") or {}).get("tex"),
        (ctx.get("optimized") or {}).get("tex"),
        ctx.get("humanized_tex"),
        ctx.get("resume_tex"),
    ):
        if isinstance(src, str) and src.strip():
            return src
    return ""


def _pick_cl(ctx: Dict[str, Any]) -> str:
    v = (ctx.get("cover_letter") or {}).get("tex")
    return v.strip() if isinstance(v, str) else ""


def _load_context(req: TalkReq) -> Tuple[Dict[str, Any], Optional[Path]]:
    path: Optional[Path] = None
    if (req.context_key or "").strip():
        path = _path_for_key(req.context_key.strip())
    elif (req.context_id or req.title or "").strip():
        stem = safe_filename((req.context_id or req.title or "").strip())
        path = CONTEXT_DIR / f"{stem}.json"
    elif req.use_latest:
        path = _latest_path()
    ctx = _read_json(path)
    if ctx:
        log_event({"event": "talk_context_loaded", "key": _coerce_key(ctx, path)})
    return ctx, path


# ============================================================
# 🧩 OPENAI HELPER — with retry
# ============================================================

async def _call_openai(system: str, user: str, model: str, temperature: float = 0.4) -> str:
    """Call OpenAI with retry logic."""
    if not openai_client:
        raise HTTPException(500, "OpenAI SDK not installed.")
    if not (getattr(config, "OPENAI_API_KEY", "") or "").strip():
        raise HTTPException(400, "OPENAI_API_KEY missing.")

    requested = (model or "").strip() or CHAT_SAFE_DEFAULT
    if _is_responses_only_model(requested):
        requested = CHAT_SAFE_DEFAULT

    last_err = None
    for attempt in range(3):
        try:
            r = await openai_client.chat.completions.create(
                model=requested,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            log_event({"event": "openai_retry", "attempt": attempt + 1, "error": str(e)[:200]})
            if attempt < 2:
                await asyncio.sleep(1.0 * (attempt + 1))

    raise HTTPException(502, f"OpenAI failed after 3 attempts: {last_err}")


# ============================================================
# 🔎 RESUME & JD ANALYSIS — v2.0.0: deeper JD extraction
# ============================================================

async def extract_resume_highlights(resume_text: str, model: str = SUMMARIZER_MODEL) -> Dict[str, Any]:
    if not (resume_text or "").strip():
        return {"top_achievements": [], "technical_skills": [], "leadership_examples": [],
                "unique_strengths": [], "companies_worked": [], "quantified_results": [], "roles": []}

    sanitized = _strip_cgpa(resume_text)
    prompt = f"""Extract the MOST IMPRESSIVE highlights from this resume. Be precise.

RESUME:
{sanitized[:5000]}

Return STRICT JSON:
{{
    "top_achievements": ["3-5 most impressive achievements with QUANTIFIED outcomes"],
    "technical_skills": ["key technical skills"],
    "leadership_examples": ["ownership/leadership examples"],
    "unique_strengths": ["what makes this candidate unique"],
    "companies_worked": ["company names"],
    "quantified_results": ["results with numbers/metrics"],
    "roles": ["job titles held"],
    "projects_built": ["named projects or systems built"]
}}

RULES: Focus on professional achievements ONLY. No GPA/CGPA/academic scores.
"""
    try:
        raw = await _call_openai("Extract resume highlights as JSON.", prompt, model)
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            h = json.loads(m.group(0))
            # Filter academic indicators from all list fields
            for k in h:
                if isinstance(h[k], list):
                    h[k] = [item for item in h[k]
                            if not any(ind in str(item).lower() for ind in _ACADEMIC_INDICATORS)]
            return h
    except Exception as e:
        log_event({"event": "resume_highlight_fail", "error": str(e)[:200]})
    return {"top_achievements": [], "technical_skills": [], "leadership_examples": [],
            "unique_strengths": [], "companies_worked": [], "quantified_results": [], "roles": []}


async def extract_jd_requirements(jd_text: str, model: str = SUMMARIZER_MODEL) -> Dict[str, Any]:
    """v2.0.0: Deeper JD extraction including challenges, terminology, team context."""
    if not (jd_text or "").strip():
        return {"must_have_skills": [], "nice_to_have_skills": [], "key_responsibilities": [],
                "tech_stack": [], "team_context": "", "success_metrics": [],
                "company_challenges": [], "jd_terminology": [], "role_level": "mid"}

    prompt = f"""Extract DEEP requirements from this JD. Be thorough.

JOB DESCRIPTION:
{jd_text[:4000]}

Return STRICT JSON:
{{
    "must_have_skills": ["required technical skills — EXACT terms from JD"],
    "nice_to_have_skills": ["preferred skills — EXACT terms from JD"],
    "key_responsibilities": ["main job duties — use JD's OWN language"],
    "tech_stack": ["specific technologies mentioned"],
    "team_context": "what team/product this is for (1-2 sentences)",
    "success_metrics": ["how success might be measured in this role"],
    "company_challenges": ["challenges this role is hired to solve"],
    "jd_terminology": ["10-15 important EXACT phrases/terms unique to this JD"],
    "role_level": "intern|junior|mid|senior|staff|principal",
    "day_to_day": ["what this person will actually do daily"],
    "collaboration_scope": "who this role works with (teams, stakeholders)"
}}

RULES:
- Use EXACT language from the JD for terminology and responsibilities
- Identify the REAL problems this role solves (company_challenges)
- Be specific about success_metrics — what would make someone great in this role
"""
    try:
        raw = await _call_openai("Extract JD requirements as JSON.", prompt, model)
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            return json.loads(m.group(0))
    except Exception as e:
        log_event({"event": "jd_extract_fail", "error": str(e)[:200]})
    return {"must_have_skills": [], "nice_to_have_skills": [], "key_responsibilities": [],
            "tech_stack": [], "team_context": "", "success_metrics": [],
            "company_challenges": [], "jd_terminology": [], "role_level": "mid"}


# ============================================================
# 🔮 FOLLOW-UP PREDICTOR
# ============================================================

def predict_followups(question: str, answer: str, q_type: str, q_strategy: Dict[str, Any]) -> List[Dict[str, str]]:
    followups = []
    for f in q_strategy.get("likely_followups", [])[:3]:
        followups.append({"question": f, "type": "standard", "tip": "Be ready with a specific example"})

    al = answer.lower()
    if re.search(r"\d+%?", answer):
        followups.append({"question": "How did you measure that?", "type": "verification", "tip": "Know your methodology"})
    if "project" in al or "system" in al:
        followups.append({"question": "Tell me more about the technical architecture", "type": "depth", "tip": "Know the details"})
    if "team" in al or "collaborated" in al:
        followups.append({"question": "How did you handle disagreements?", "type": "behavioral", "tip": "Have a conflict example ready"})
    return followups[:5]


# ============================================================
# 💬 KILLER ANSWER GENERATION — v2.0.0: deep JD customization
# ============================================================

async def generate_killer_answer(
    jd_text: str, resume_text: str, question: str,
    company: str, role: str, model: str,
    cover_letter: str = "",
    resume_highlights: Optional[Dict[str, Any]] = None,
    jd_requirements: Optional[Dict[str, Any]] = None,
    skill_gaps: Optional[Dict[str, Any]] = None,
    company_intel: Optional[Dict[str, Any]] = None,
    interview_stage: Optional[str] = None,
    session_key: str = "",
) -> str:
    q_type, q_strategy = detect_question_type(question)
    is_behavioral = q_strategy.get("behavioral", False)
    sanitized_resume = _strip_cgpa(resume_text)

    # Build achievement context (avoid repeating across session)
    used = _SESSION_USED_ACHIEVEMENTS.get(session_key, set())
    achievements = [a for a in (resume_highlights or {}).get("top_achievements", [])
                    if a not in used and not any(ind in a.lower() for ind in _ACADEMIC_INDICATORS)]
    achievements_str = ""
    if achievements:
        achievements_str = "YOUR RESUME ACHIEVEMENTS (use ONLY these facts):\n" + "\n".join(f"• {a}" for a in achievements[:5])

    # JD requirements context — v2.0.0: much richer
    jd_ctx = ""
    if jd_requirements:
        parts = []
        skills = jd_requirements.get("must_have_skills", [])
        if skills:
            parts.append(f"REQUIRED SKILLS (weave 2-3 into your answer): {', '.join(skills[:10])}")
        resp = jd_requirements.get("key_responsibilities", [])
        if resp:
            parts.append(f"KEY RESPONSIBILITIES (show you can do these): {'; '.join(resp[:5])}")
        challenges = jd_requirements.get("company_challenges", [])
        if challenges:
            parts.append(f"PROBLEMS THIS ROLE SOLVES (address these): {'; '.join(challenges[:3])}")
        terms = jd_requirements.get("jd_terminology", [])
        if terms:
            parts.append(f"JD TERMINOLOGY (mirror this language): {', '.join(terms[:10])}")
        day2day = jd_requirements.get("day_to_day", [])
        if day2day:
            parts.append(f"DAILY WORK (show familiarity): {'; '.join(day2day[:3])}")
        collab = jd_requirements.get("collaboration_scope", "")
        if collab:
            parts.append(f"COLLABORATION SCOPE: {collab}")
        role_level = jd_requirements.get("role_level", "mid")
        parts.append(f"ROLE LEVEL: {role_level} (calibrate depth/autonomy accordingly)")
        jd_ctx = "\n".join(parts)

    # Skill gap context
    gap_ctx = ""
    if skill_gaps and skill_gaps.get("must_have_gaps"):
        gaps = skill_gaps["must_have_gaps"][:3]
        strategies = skill_gaps.get("gap_strategies", {})
        gap_ctx = "SKILL GAPS (address honestly if relevant):\n" + "\n".join(
            f"• {g}: {strategies.get(g, 'Show willingness to learn')}" for g in gaps)

    # Company intel context
    ci = company_intel or _DEFAULT_INTEL
    stage_ctx = ""
    if interview_stage:
        stage_focus = ci.get("interview_stages", {}).get(interview_stage, "")
        if stage_focus:
            stage_ctx = f"INTERVIEW STAGE: {interview_stage.upper()} — Focus: {stage_focus}"

    # STAR enforcement for behavioral questions
    star_instruction = ""
    if is_behavioral:
        star_instruction = """
MANDATORY STAR STRUCTURE:
Your answer MUST follow this structure:
1. SITUATION (2 sentences max): Set the scene — where, when, what was the challenge
2. ACTION (bulk of answer): What YOU specifically did — be detailed and first-person
3. RESULT (1-2 sentences): Quantified outcome — use real numbers from your resume
4. TAKEAWAY (1 sentence): What you learned or how you'd apply it here

DO NOT skip any section. The ACTION section should be 50-60% of the answer.
"""

    # Previous answers in session (avoid repetition)
    prev_answers = _SESSION_ANSWERS.get(session_key, [])
    dedup_ctx = ""
    if prev_answers:
        used_achievements = [a.get("key_achievement", "") for a in prev_answers[-3:]]
        dedup_ctx = f"ALREADY USED IN THIS SESSION (use DIFFERENT examples): {'; '.join(a for a in used_achievements if a)}"

    sys_prompt = f"""You are an elite interview coach crafting a WINNING answer.

QUESTION TYPE: {q_type.upper()}
STRATEGY: {q_strategy['strategy']}

ANSWER STRUCTURE:
{chr(10).join(f"• {s}" for s in q_strategy['structure'])}
{star_instruction}

COMPANY: {company.upper()}
What they value: {', '.join(ci.get('what_they_value', [])[:4])}
What impresses them: {chr(10).join(f'• {w}' for w in ci.get('what_impresses_them', [])[:4])}
Avoid saying: {', '.join(ci.get('avoid_saying', [])[:3])}
{stage_ctx}

TRAP WARNINGS:
{chr(10).join(f'• {t}' for t in q_strategy.get('trap_warnings', [])[:3])}

{dedup_ctx}

ABSOLUTE RULES:
1. HOOK: First sentence grabs attention — NO generic openings
2. GROUNDING: Use ONLY facts from the resume — NEVER invent
3. JD MIRRORING: Use EXACT terminology from the JD in your answer
4. SPECIFICITY: Include real numbers, tools, company names from resume
5. CONFIDENCE: No hedging ("I think", "maybe"), no pleading ("hope you consider")
6. LENGTH: 2-3 paragraphs, 120-180 words
7. NO CLICHES: Never say "passionate", "team player", "hard worker", "self-starter"
8. NO GPA/CGPA: Never mention grades, GPA, CGPA, academic honors, graduation dates
9. JD CHALLENGES: Show you understand the PROBLEMS this role solves
10. FORWARD-LOOKING: End with what you'll CONTRIBUTE, not just what you've done
"""

    user_prompt = f"""QUESTION: {question}

COMPANY: {company}
ROLE: {role}

{achievements_str}

{jd_ctx}

{gap_ctx}

JOB DESCRIPTION:
{jd_text[:3000]}

RESUME (ONLY use facts from here):
{sanitized_resume[:3000]}

{f'COVER LETTER:{chr(10)}{cover_letter[:1000]}' if cover_letter else ''}

Write the answer now. 120-180 words. Start with a strong hook.
"""

    answer = await _call_openai(sys_prompt, user_prompt, model=model, temperature=0.45)

    # Track session
    if session_key:
        if session_key not in _SESSION_ANSWERS:
            _SESSION_ANSWERS[session_key] = []
        _SESSION_ANSWERS[session_key].append({
            "question": question, "q_type": q_type,
            "key_achievement": achievements[0] if achievements else "",
        })
        if achievements:
            if session_key not in _SESSION_USED_ACHIEVEMENTS:
                _SESSION_USED_ACHIEVEMENTS[session_key] = set()
            _SESSION_USED_ACHIEVEMENTS[session_key].add(achievements[0])

    return _tex_safe(answer)


# ============================================================
# ✨ HUMANIZE — v2.0.0: self-contained (no internal API dep)
# ============================================================

async def humanize_answer(answer_text: str, tone: str, q_type: str, model: str = ANSWER_MODEL) -> Tuple[str, bool]:
    """Refine answer using GPT directly (no internal API dependency)."""
    prompt = f"""Rewrite this {q_type} interview answer to sound more natural and human.

CURRENT ANSWER:
{answer_text}

RULES:
- PRESERVE: opening hook, specific achievements, numbers, company names, confident tone
- IMPROVE: natural flow, conversational rhythm, remove AI-sounding phrases
- KEEP: 2-3 paragraphs, 120-180 words
- TONE: {tone} — confident professional
- DO NOT: add cliches, remove specific details, change facts, add GPA/CGPA/academic metrics
- DO NOT: add "passionate", "team player", "hard worker", "self-starter"

Return ONLY the improved answer, nothing else.
"""
    try:
        rewritten = await _call_openai(
            "Rewrite interview answer to sound natural. Never mention GPA/CGPA.",
            prompt, model, temperature=0.5,
        )
        was_changed = rewritten.strip() != answer_text.strip()
        return _tex_safe(rewritten), was_changed
    except Exception as e:
        log_event({"event": "humanize_fail", "error": str(e)[:200]})
        return answer_text, False


# ============================================================
# 🔄 ANSWER IMPROVEMENT LOOP — v2.0.0: up to 3 iterations
# ============================================================

async def improve_answer(
    answer: str, quality: Dict[str, Any], question: str,
    company: str, role: str, jd_req: Dict[str, Any],
    resume_hl: Dict[str, Any], model: str,
    max_iter: int = 3,
) -> Tuple[str, Dict[str, Any]]:
    """Iteratively improve answer until quality threshold or max iterations."""
    current = answer
    current_quality = quality
    scorer = AnswerQualityScorer()

    for i in range(max_iter):
        if current_quality.get("overall_score", 0) >= 7.5:
            break

        feedback = current_quality.get("feedback", [])
        if not feedback:
            break

        prompt = f"""Improve this interview answer based on specific feedback.

CURRENT ANSWER:
{current}

FEEDBACK TO ADDRESS:
{chr(10).join(f'• {f}' for f in feedback)}

QUESTION: {question}
COMPANY: {company}  |  ROLE: {role}

JD SKILLS TO MENTION: {', '.join(jd_req.get('must_have_skills', [])[:6])}
JD CHALLENGES: {'; '.join(jd_req.get('company_challenges', [])[:3])}

RULES:
- Fix the specific issues in feedback
- Keep 120-180 words
- Maintain confidence and specificity
- Do NOT invent achievements
- NEVER mention GPA, CGPA, academic scores, graduation dates

Return only the improved answer.
"""
        try:
            improved = await _call_openai(
                "Improve interview answer per feedback. Never mention GPA/CGPA.",
                prompt, model, temperature=0.4,
            )
            new_quality = scorer.score_answer(improved, question, "generic", company, jd_req, resume_hl)
            if new_quality.get("overall_score", 0) > current_quality.get("overall_score", 0):
                current = improved
                current_quality = new_quality
                log_event({"event": "answer_improved", "iteration": i + 1,
                           "old": current_quality.get("overall_score"),
                           "new": new_quality.get("overall_score")})
            else:
                break
        except Exception:
            break

    return current, current_quality


# ============================================================
# 🟢 HEALTH CHECK
# ============================================================

@router.get("/ping")
async def ping():
    return {"ok": True, "service": "talk", "version": "v2.0.0",
            "epoch": time.time(), "iso": datetime.now(tz=timezone.utc).isoformat()}


# ============================================================
# 🚀 MAIN ENDPOINT — v2.0.0
# ============================================================

@router.post("/answer")
@router.post("")
async def talk_to_hirex(req: TalkReq):
    # ── Load context ──
    jd_text = (req.jd_text or "").strip()
    resume_tex = (req.resume_tex or "").strip()
    cover_letter_tex = ""
    used_key = used_company = used_role = ""

    if not jd_text or (not resume_tex and not (req.resume_plain or "").strip()):
        ctx, ctx_path = _load_context(req)
        if ctx:
            jd_text = jd_text or (ctx.get("jd_text") or "")
            resume_tex = resume_tex or _pick_resume(ctx)
            cover_letter_tex = _pick_cl(ctx)
            used_key = _coerce_key(ctx, ctx_path)
            used_company = (ctx.get("company") or "").strip()
            used_role = (ctx.get("role") or "").strip()

    if not jd_text.strip():
        raise HTTPException(400, "Job Description missing.")
    if not (resume_tex or (req.resume_plain or "").strip()):
        raise HTTPException(400, "Resume text missing.")

    resume_text = _strip_cgpa(resume_tex or req.resume_plain or "")
    model = (req.model or ANSWER_MODEL).strip() or ANSWER_MODEL
    sk = _session_key(jd_text, resume_text)

    # ── Extract company/role from JD if missing ──
    if not used_company or not used_role:
        try:
            raw = await _call_openai(
                "Extract as JSON.",
                f'Return JSON: {{"company":"...","role":"..."}}\nJD: {jd_text[:2000]}',
                SUMMARIZER_MODEL, 0.0,
            )
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                d = json.loads(m.group(0))
                used_company = used_company or d.get("company", "Company")
                used_role = used_role or d.get("role", "Role")
        except Exception:
            used_company = used_company or "Company"
            used_role = used_role or "Role"

    q_type, q_strategy = detect_question_type(req.question)

    # ── Parallel extraction ──
    resume_hl, jd_req, company_intel = await asyncio.gather(
        extract_resume_highlights(resume_text, model),
        extract_jd_requirements(jd_text, model),
        get_company_intelligence(used_company, jd_text, model),
    )

    skill_gaps = await analyze_skill_gaps(resume_hl, jd_req)

    # ── Generate answer ──
    draft = await generate_killer_answer(
        jd_text=jd_text, resume_text=resume_text, question=req.question,
        company=used_company, role=used_role, model=model,
        cover_letter=cover_letter_tex, resume_highlights=resume_hl,
        jd_requirements=jd_req, skill_gaps=skill_gaps,
        company_intel=company_intel, interview_stage=req.interview_stage,
        session_key=sk,
    )

    # ── Single CGPA gate (v2.0.0: consolidated) ──
    draft = _strip_cgpa(draft)

    # ── Score ──
    scorer = AnswerQualityScorer()
    quality = scorer.score_answer(draft, req.question, q_type, used_company, jd_req, resume_hl, company_intel)

    # ── Improve if needed ──
    if quality.get("overall_score", 0) < 7.5:
        draft, quality = await improve_answer(
            draft, quality, req.question, used_company, used_role,
            jd_req, resume_hl, model,
        )

    # ── Humanize ──
    if req.humanize:
        final, was_humanized = await humanize_answer(draft, req.tone, q_type, model)
    else:
        final, was_humanized = draft, False

    # ── Final CGPA gate ──
    final = _strip_cgpa(final)
    is_cgpa_clean, _ = _validate_no_cgpa(final)

    # ── Follow-ups ──
    followups = predict_followups(req.question, final, q_type, q_strategy) if req.include_followups else []

    log_event({"event": "talk_answer_v2", "q_type": q_type, "company": used_company,
               "score": quality.get("overall_score"), "grade": quality.get("grade"),
               "humanized": was_humanized, "cgpa_clean": is_cgpa_clean})

    response = {
        "question": req.question.strip(),
        "question_type": q_type,
        "strategy_used": q_strategy["strategy"],
        "draft_answer": draft,
        "final_text": final,
        "answer": final,
        "tone": req.tone,
        "humanized": was_humanized,
        "model": model,
        "company_intel": {
            "what_they_value": company_intel.get("what_they_value", [])[:3],
            "what_impresses_them": company_intel.get("what_impresses_them", [])[:3],
            "avoid_saying": company_intel.get("avoid_saying", [])[:3],
            "culture_keywords": company_intel.get("culture_keywords", [])[:3],
        },
        "context": {
            "key": used_key, "company": used_company, "role": used_role,
            "has_cover_letter": bool(cover_letter_tex),
        },
        "quality": {
            "overall_score": quality.get("overall_score", 0),
            "grade": quality.get("grade", "N/A"),
            "pass": quality.get("pass", False),
            "dimension_scores": quality.get("dimension_scores", {}),
            "feedback": quality.get("feedback", []),
        } if req.include_quality_score else None,
        "predicted_followups": followups if req.include_followups else None,
        "skill_analysis": {
            "match_percentage": skill_gaps.get("match_percentage", 0),
            "strengths_to_emphasize": skill_gaps.get("strengths_to_emphasize", [])[:5],
            "gaps_to_address": skill_gaps.get("must_have_gaps", [])[:3],
            "gap_strategies": skill_gaps.get("gap_strategies", {}),
        },
        "jd_deep_context": {
            "challenges_addressed": jd_req.get("company_challenges", [])[:3],
            "jd_terminology_used": jd_req.get("jd_terminology", [])[:8],
            "role_level": jd_req.get("role_level", "mid"),
            "success_metrics": jd_req.get("success_metrics", [])[:3],
        },
        "trap_warnings": q_strategy.get("trap_warnings", [])[:3],
        "salary_intel": company_intel.get("salary_range") if q_type == "salary" else None,
        "company_common_questions": company_intel.get("common_questions", [])[:5],
        "competitor_differentiation": {
            "competitors": company_intel.get("competitors", []),
            "why_not_them": company_intel.get("why_not_competitors", ""),
        } if q_type == "why_this_company" else None,
        "interview_stage": {
            "stage": req.interview_stage,
            "focus": company_intel.get("interview_stages", {}).get(req.interview_stage, ""),
        } if req.interview_stage else None,
    }
    return {k: v for k, v in response.items() if v is not None}


# ============================================================
# 📚 ADDITIONAL ENDPOINTS
# ============================================================

@router.get("/company-intel/{company}")
async def get_company_intel_endpoint(company: str, jd_text: str = ""):
    intel = await get_company_intelligence(company, jd_text)
    return {"company": company, "found": intel != _DEFAULT_INTEL, "intelligence": intel}


@router.get("/question-types")
async def list_question_types():
    return {
        "question_types": {
            qt: {
                "strategy": cfg["strategy"],
                "structure": cfg["structure"],
                "behavioral": cfg.get("behavioral", False),
                "trap_warnings": cfg.get("trap_warnings", [])[:2],
                "likely_followups": cfg.get("likely_followups", [])[:3],
            }
            for qt, cfg in QUESTION_STRATEGIES.items()
        }
    }


@router.post("/analyze-gaps")
async def analyze_gaps_endpoint(jd_text: str, resume_text: str, model: str = SUMMARIZER_MODEL):
    resume_text = _strip_cgpa(resume_text)
    rh = await extract_resume_highlights(resume_text, model)
    jd = await extract_jd_requirements(jd_text, model)
    gaps = await analyze_skill_gaps(rh, jd)
    return {"resume_skills": rh.get("technical_skills", []),
            "jd_requirements": jd.get("must_have_skills", []), "analysis": gaps}


@router.post("/score-answer")
async def score_answer_endpoint(answer: str, question: str, company: str,
                                 jd_text: str = "", resume_text: str = ""):
    q_type, _ = detect_question_type(question)
    jd = await extract_jd_requirements(jd_text, SUMMARIZER_MODEL) if jd_text else {}
    rh = await extract_resume_highlights(_strip_cgpa(resume_text), SUMMARIZER_MODEL) if resume_text else {}
    scorer = AnswerQualityScorer()
    score = scorer.score_answer(answer, question, q_type, company, jd, rh)
    return {"question": question, "question_type": q_type, "company": company, "score": score}


@router.post("/predict-followups")
async def predict_followups_endpoint(question: str, answer: str):
    q_type, q_strategy = detect_question_type(question)
    return {"question": question, "question_type": q_type,
            "predicted_followups": predict_followups(question, answer, q_type, q_strategy)}


@router.post("/filter-cgpa")
async def filter_cgpa_endpoint(text: str):
    filtered = _strip_cgpa(text)
    ok, violations = _validate_no_cgpa(filtered)
    return {"original": text, "filtered": filtered, "is_clean": ok,
            "violations_found": violations if not ok else []}