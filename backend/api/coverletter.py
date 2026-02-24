# ============================================================
#  HIREX v1.0.0 — Authentic Cover Letter Generation
#  ------------------------------------------------------------
#  CHANGELOG v1.0.0:
#   • Employer address extracted from JD (if available)
#   • Employer address block OMITTED entirely when not found
#   • Employer name extracted for salutation ("Dear [Name]," vs generic)
#   • Smarter salutation: "Dear Hiring Manager," not location placeholder
#   • Better LaTeX template address injection with conditional blocks
#   • Added: role-specific opening hook selection
#   • Added: duplicate sentence detection
#   • Added: final formatting pass for LaTeX safety
#   • Improved validation: score-based repair instead of all-or-nothing
# ============================================================

from __future__ import annotations

import base64
import json
import re
import threading
import random
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List, Set

import httpx
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI

from backend.core import config
from backend.core.utils import log_event, safe_filename, ensure_dir
from backend.core.compiler import compile_latex_safely
from backend.core.security import secure_tex_input

try:
    from backend.api.render_tex import render_final_tex
except Exception:
    from api.render_tex import render_final_tex

try:
    from backend.api.latex_parse import inject_cover_body as _shared_inject
except Exception:
    try:
        from api.latex_parse import inject_cover_body as _shared_inject
    except Exception:
        _shared_inject = None

router = APIRouter(prefix="/api/coverletter", tags=["coverletter"])

_openai_lock = threading.Lock()
_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    with _openai_lock:
        if _openai_client is None:
            _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


_DEFAULT_OAI_MODEL   = "gpt-4o-mini"
_EXTRACT_MODEL       = getattr(config, "COVERLETTER_EXTRACT_MODEL", None) or _DEFAULT_OAI_MODEL
_DRAFT_MODEL         = getattr(config, "COVERLETTER_MODEL", None)         or _DEFAULT_OAI_MODEL
_INTELLIGENCE_MODEL  = "gpt-4o-mini"

_DISABLE_SHARED_INJECTOR = True


# ============================================================
# 🎯 COMPANY-SPECIFIC HOOKS
# ============================================================

COMPANY_KILLER_HOOKS: Dict[str, Dict[str, List[str]]] = {
    "netflix": {
        "product_insights": [
            "Netflix's recommendation engine personalizing content for 230M+ subscribers demonstrates ML at scale",
            "The A/B testing challenge across Netflix's entire product surface is a compelling ML systems problem",
            "Netflix's shift to an ad-supported tier while maintaining personalization quality is an interesting engineering constraint",
            "Netflix's streaming quality optimization work shows thoughtful infrastructure thinking",
        ],
        "insider_knowledge": [
            "Netflix's 'context not control' culture aligns with how I prefer to build and ship systems",
            "The challenge of maintaining recommendation quality while expanding into new verticals is compelling",
            "Netflix's experimentation platform running thousands of concurrent tests represents the scale I want to work at",
        ],
        "recent_moves": [
            "Netflix's expansion into gaming presents new personalization challenges",
            "The focus on sustainable growth through multiple revenue streams shows strategic discipline",
        ],
    },
    "google": {
        "product_insights": [
            "Google Search's integration of AI-generated summaries is a significant infrastructure and UX challenge",
            "The evolution from PageRank to modern ML-powered ranking is a compelling research-to-production story",
            "Gemini's integration across Google's product surface demonstrates complex system design at scale",
        ],
        "insider_knowledge": [
            "Google's 'launch and iterate' culture while maintaining reliability for billions of users resonates with me",
            "The design doc culture at Google reflects how I approach technical architecture",
        ],
        "recent_moves": [
            "Google's AI search integration shows deliberate technical execution",
            "Cloud's growth trajectory in the enterprise market is compelling to be part of",
        ],
    },
    "meta": {
        "product_insights": [
            "Meta's Feed ranking evolution from chronological to ML-driven is a recommendation systems story worth being part of",
            "The integrity challenge of moderating content at billions-of-posts scale is genuinely interesting ML",
        ],
        "insider_knowledge": [
            "The PyTorch ecosystem Meta open-sourced has been foundational to my ML work",
            "Meta's 'Move Fast' culture balanced with responsibility for 3B+ users is a compelling engineering environment",
        ],
        "recent_moves": [
            "Threads' rapid growth demonstrated Meta's infrastructure capabilities",
            "Llama model releases show commitment to advancing the field openly",
        ],
    },
    "amazon": {
        "product_insights": [
            "Amazon's recommendation systems hiding sophisticated ML behind a simple shopping UX is elegant product thinking",
            "The optimization challenge of same-day delivery across millions of products is compelling logistics ML",
            "AWS serving both early-stage startups and global enterprises demonstrates strong platform design",
        ],
        "insider_knowledge": [
            "Amazon's 'working backwards' approach aligns with how I think about system and product design",
            "The 'Dive Deep' principle matches my belief in understanding the implementation details before abstracting",
        ],
        "recent_moves": [
            "AWS Bedrock making foundation models accessible to developers shows platform-first thinking",
            "Amazon's healthcare expansion opens new ML application areas with real human impact",
        ],
    },
    "microsoft": {
        "product_insights": [
            "Copilot's integration across Microsoft 365 represents a meaningful AI deployment challenge at scale",
            "GitHub Copilot's accuracy improvements over successive releases demonstrate ML iteration velocity",
            "Azure's growth trajectory shows strong enterprise execution",
        ],
        "insider_knowledge": [
            "Microsoft's 'growth mindset' transformation created an engineering culture I find genuinely appealing",
            "The OpenAI partnership positioning Microsoft at the center of the AI ecosystem was a strategic decision with deep technical implications",
        ],
        "recent_moves": [
            "Copilot becoming a unified layer across Microsoft products shows ambitious and coherent product vision",
        ],
    },
    "apple": {
        "product_insights": [
            "Apple Intelligence running ML on-device while maintaining strict privacy is a compelling technical constraint to design around",
            "The Neural Engine's evolution across chip generations shows long-term ML silicon investment",
            "Face ID working reliably across lighting conditions with zero cloud dependency is elegant on-device engineering",
        ],
        "insider_knowledge": [
            "Apple's privacy-first ML approach aligns with my values around user trust",
            "Vertical integration from silicon to software enabling deep ML optimization is a unique engineering environment",
        ],
        "recent_moves": [
            "Vision Pro's spatial computing interface presents genuinely new ML and UX challenges",
            "Apple Intelligence's quality-first rollout pace reflects principled product execution",
        ],
    },
    "stripe": {
        "product_insights": [
            "Stripe's API design set the standard for developer-facing financial infrastructure",
            "The fraud detection challenge of approving transactions instantly while blocking fraud is interesting real-time ML",
            "Stripe Atlas abstracting away entity formation for global founders shows infrastructure thinking beyond core payments",
        ],
        "insider_knowledge": [
            "Stripe's writing-first culture where ideas are evaluated on merit appeals to me",
            "The 'increase the GDP of the internet' mission is unusually concrete for a fintech company",
        ],
        "recent_moves": [
            "Embedded finance expansion broadens the technical problem space considerably",
        ],
    },
    "airbnb": {
        "product_insights": [
            "Airbnb's search ranking balancing guest experience and host fairness is genuine multi-objective optimization",
            "Smart Pricing helping hosts earn more while maintaining platform trust is an elegant ML use case",
        ],
        "insider_knowledge": [
            "Airbnb's design-driven engineering culture tends to produce products that feel cohesive",
            "The 'Belong Anywhere' mission shapes product decisions in ways that are visible even in the technical work",
        ],
        "recent_moves": [
            "Experiences expansion diversifies the ML and personalization challenges",
        ],
    },
    "uber": {
        "product_insights": [
            "Uber's marketplace balancing driver wait times and rider experience in real-time is optimization at impressive scale",
            "ETA prediction accuracy at Uber's volume is a hard ML problem with direct user impact",
        ],
        "insider_knowledge": [
            "Uber's engineering infrastructure for global-scale reliability during peak events is a strong technical foundation",
        ],
        "recent_moves": [
            "The advertising business represents new ML challenges on top of existing marketplace data",
        ],
    },
    "linkedin": {
        "product_insights": [
            "LinkedIn's job matching at scale has measurable real-world career impact",
            "Feed ranking balancing professional content with engagement signals is a nuanced multi-objective problem",
        ],
        "insider_knowledge": [
            "The Economic Graph vision — mapping every professional, company, and skill globally — is an ambitious data challenge",
        ],
        "recent_moves": [
            "AI-powered job application tools are meaningfully changing how hiring works",
        ],
    },
    "spotify": {
        "product_insights": [
            "Discover Weekly's personalization remains one of the clearest examples of ML improving a consumer product",
            "Audio ML challenges spanning music, podcasts, and audiobooks are technically diverse and interesting",
        ],
        "insider_knowledge": [
            "Spotify's squad model balancing team autonomy with product coherence is good organizational design",
            "The data-informed but not data-driven culture leaves room for product intuition alongside experimentation",
        ],
        "recent_moves": [
            "AI DJ represents the next evolution of personalization — from playlists to real-time programming",
        ],
    },
    "databricks": {
        "product_insights": [
            "The Lakehouse architecture unifying data warehousing and ML is an elegant technical vision",
            "MLflow becoming the de facto standard for experiment tracking shows successful open-source strategy",
        ],
        "insider_knowledge": [
            "Databricks' open-source DNA — publishing MLflow, Delta Lake — builds genuine community trust",
            "The 'data plus AI' positioning is clear and differentiated in a crowded market",
        ],
        "recent_moves": [
            "The Mosaic ML acquisition signals a full-stack AI platform vision",
        ],
    },
    "snowflake": {
        "product_insights": [
            "Snowflake's consumption-based pricing aligns vendor and customer incentives unusually well",
            "Data sharing without copying enabling new business models is an architecturally interesting design choice",
        ],
        "insider_knowledge": [
            "Snowflake's engineering focus on query performance at scale is a differentiated technical strength",
        ],
        "recent_moves": [
            "Native apps and Cortex features show a strategic expansion beyond core data warehousing",
        ],
    },
}


def get_killer_hook(company: str, hook_type: str = "product_insights") -> str:
    """Return a company-specific hook string, or empty string if not found."""
    company_lower = (company or "").lower().strip()
    for key, hooks in COMPANY_KILLER_HOOKS.items():
        if key in company_lower or company_lower in key:
            pool = hooks.get(hook_type, hooks.get("product_insights", []))
            return random.choice(pool) if pool else ""
    return ""


# ============================================================
# 🏢 COMPANY INTELLIGENCE DATABASE
# ============================================================

COMPANY_INTELLIGENCE: Dict[str, Dict[str, Any]] = {
    "netflix": {
        "culture_keywords": ["Freedom & Responsibility", "context not control", "highly aligned loosely coupled", "keeper test", "candor"],
        "tech_focus": ["Recommender Systems", "Personalization", "A/B Testing at Scale", "Content Delivery", "Streaming Infrastructure"],
        "engineering_values": ["data-driven decisions", "experimentation culture", "ownership mentality", "impact over activity"],
        "recent_focus": ["gaming expansion", "ad-supported tier", "live events", "content efficiency"],
        "challenges": ["subscriber growth in mature markets", "content cost optimization", "competition from Disney+/HBO"],
        "insider_phrases": ["member experience", "title discovery", "personalization at scale", "streaming quality"],
        "hiring_priorities": ["ML infrastructure", "experimentation platforms", "content algorithms", "data pipelines"],
        "products_to_reference": ["recommendation engine", "A/B testing platform", "streaming infrastructure", "content personalization"],
    },
    "google": {
        "culture_keywords": ["Googleyness", "think big", "user first", "10x thinking", "psychological safety"],
        "tech_focus": ["Scalability", "Distributed Systems", "AI/ML Infrastructure", "Search Quality", "Cloud Platform"],
        "engineering_values": ["code quality", "design docs", "peer review culture", "technical excellence"],
        "recent_focus": ["Gemini AI", "Cloud growth", "Search AI integration", "Pixel ecosystem"],
        "challenges": ["AI competition with OpenAI/Microsoft", "advertising revenue pressure", "regulatory scrutiny"],
        "insider_phrases": ["Noogler", "OKRs", "launch and iterate", "10x improvement"],
        "hiring_priorities": ["AI/ML", "Cloud infrastructure", "Privacy engineering", "Mobile development"],
        "products_to_reference": ["Search", "Gemini", "Cloud Platform", "YouTube", "Android"],
    },
    "meta": {
        "culture_keywords": ["Move Fast", "Be Bold", "Focus on Impact", "Be Open", "Build Social Value"],
        "tech_focus": ["Social Graph", "Ranking Systems", "AR/VR", "Messaging Infrastructure", "Ads Optimization"],
        "engineering_values": ["ship early ship often", "hackathons", "bootcamp culture", "impact metrics"],
        "recent_focus": ["AI assistants", "Threads growth", "Reels monetization", "Llama models"],
        "challenges": ["privacy regulations", "TikTok competition", "metaverse ROI questions"],
        "insider_phrases": ["family of apps", "integrity systems", "social impact", "meaningful connections"],
        "hiring_priorities": ["AI/ML", "Integrity/Safety", "Infrastructure", "AR/VR"],
        "products_to_reference": ["Instagram", "WhatsApp", "Messenger", "Threads", "Llama"],
    },
    "amazon": {
        "culture_keywords": ["Customer Obsession", "Ownership", "Bias for Action", "Dive Deep", "Deliver Results"],
        "tech_focus": ["AWS Services", "Supply Chain ML", "Alexa/Voice", "Retail Optimization", "Logistics"],
        "engineering_values": ["working backwards", "two-pizza teams", "operational excellence", "frugality"],
        "recent_focus": ["AWS AI services", "same-day delivery", "healthcare expansion", "advertising growth"],
        "challenges": ["labor relations", "AWS competition", "retail margins"],
        "insider_phrases": ["PR/FAQ", "6-pager", "bar raiser", "Day 1 mentality", "mechanisms"],
        "hiring_priorities": ["AWS", "ML/AI", "Supply chain", "Advertising"],
        "products_to_reference": ["AWS", "Prime", "Alexa", "One Medical", "Bedrock"],
    },
    "microsoft": {
        "culture_keywords": ["Growth Mindset", "Customer Obsessed", "Diverse and Inclusive", "One Microsoft"],
        "tech_focus": ["Azure Cloud", "Microsoft 365", "AI/Copilot", "Gaming/Xbox", "Developer Tools"],
        "engineering_values": ["learn-it-all not know-it-all", "customer empathy", "responsible AI"],
        "recent_focus": ["Copilot integration", "Azure OpenAI", "Gaming", "Teams platform"],
        "challenges": ["cloud competition with AWS", "AI integration", "gaming market share"],
        "insider_phrases": ["growth mindset", "customer zero", "inclusive design"],
        "hiring_priorities": ["AI/ML", "Azure", "Security", "Developer experience"],
        "products_to_reference": ["Copilot", "Azure", "GitHub", "Teams", "Xbox"],
    },
    "apple": {
        "culture_keywords": ["Think Different", "Simplicity", "Privacy as Human Right", "Excellence"],
        "tech_focus": ["On-Device ML", "Privacy-Preserving AI", "Hardware-Software Integration", "User Experience"],
        "engineering_values": ["attention to detail", "user privacy", "vertical integration", "craftsmanship"],
        "recent_focus": ["Vision Pro", "Apple Intelligence", "Services growth", "Sustainability"],
        "challenges": ["China market", "AI catch-up", "services growth"],
        "insider_phrases": ["DRI", "surprise and delight", "it just works"],
        "hiring_priorities": ["ML on-device", "Privacy engineering", "AR/VR", "Health tech"],
        "products_to_reference": ["iPhone", "Vision Pro", "Apple Intelligence", "Neural Engine"],
    },
    "stripe": {
        "culture_keywords": ["Users First", "Move with Urgency", "Think Rigorously", "Trust and Amplify"],
        "tech_focus": ["Payment Infrastructure", "Financial APIs", "Fraud Detection", "Developer Experience"],
        "engineering_values": ["write like you code", "rigor in thinking", "long-term orientation"],
        "recent_focus": ["Embedded finance", "Global expansion", "Revenue recognition"],
        "challenges": ["fintech competition", "regulatory complexity", "enterprise sales"],
        "insider_phrases": ["increase GDP of internet", "payment rails", "developer love"],
        "hiring_priorities": ["Infrastructure", "ML/Fraud", "Platform", "International"],
        "products_to_reference": ["Payments API", "Radar", "Atlas", "Connect", "Billing"],
    },
    "airbnb": {
        "culture_keywords": ["Belong Anywhere", "Champion the Mission", "Be a Host", "Embrace Adventure"],
        "tech_focus": ["Search & Ranking", "Pricing Algorithms", "Trust & Safety", "Payments"],
        "engineering_values": ["customer empathy", "design-driven", "data-informed"],
        "recent_focus": ["Experiences expansion", "Long-term stays", "AI trip planning"],
        "challenges": ["regulatory battles", "hotel competition", "host supply"],
        "insider_phrases": ["belonging", "host community", "guest journey"],
        "hiring_priorities": ["ML/Search", "Trust & Safety", "Payments", "Mobile"],
        "products_to_reference": ["Search ranking", "Smart Pricing", "Experiences", "AirCover"],
    },
    "uber": {
        "culture_keywords": ["Build Globally", "Celebrate Differences", "Act Like Owners", "Persevere"],
        "tech_focus": ["Marketplace Optimization", "ETA Prediction", "Route Optimization", "Fraud Detection"],
        "engineering_values": ["data-driven", "experimentation", "reliability at scale"],
        "recent_focus": ["Delivery growth", "Advertising", "Freight"],
        "challenges": ["driver supply", "profitability", "regulatory issues"],
        "insider_phrases": ["marketplace balance", "rider experience", "driver earnings"],
        "hiring_priorities": ["ML/Optimization", "Maps", "Marketplace", "Delivery"],
        "products_to_reference": ["Rides", "Eats", "Freight", "Advertising platform"],
    },
    "linkedin": {
        "culture_keywords": ["Members First", "Relationships Matter", "Be Open Honest Constructive", "Act Like Owner"],
        "tech_focus": ["Feed Ranking", "Job Matching", "Graph Systems", "Economic Graph"],
        "engineering_values": ["test and learn", "member value", "data-driven"],
        "recent_focus": ["AI features", "Creator economy", "Skills-based hiring"],
        "challenges": ["engagement growth", "premium conversion", "content quality"],
        "insider_phrases": ["economic graph", "member value", "professional identity"],
        "hiring_priorities": ["AI/ML", "Feed", "Search", "Infrastructure"],
        "products_to_reference": ["Feed", "Jobs", "Learning", "Sales Navigator"],
    },
    "spotify": {
        "culture_keywords": ["Innovative", "Collaborative", "Sincere", "Passionate", "Playful"],
        "tech_focus": ["Audio ML", "Personalization", "Content Delivery", "Creator Tools"],
        "engineering_values": ["squad model", "autonomous teams", "data-informed"],
        "recent_focus": ["Podcasts", "Audiobooks", "AI DJ", "Creator monetization"],
        "challenges": ["profitability", "music licensing costs", "podcast ROI"],
        "insider_phrases": ["Discover Weekly", "audio-first", "creator ecosystem"],
        "hiring_priorities": ["ML/Personalization", "Audio", "Ads", "Payments"],
        "products_to_reference": ["Discover Weekly", "AI DJ", "Wrapped", "Podcast platform"],
    },
    "databricks": {
        "culture_keywords": ["Customer Obsessed", "Unity", "Ownership", "Open Source First"],
        "tech_focus": ["Lakehouse", "Delta Lake", "MLflow", "Spark", "Data Engineering"],
        "engineering_values": ["open source contribution", "technical excellence", "customer impact"],
        "recent_focus": ["Unity Catalog", "Serverless", "AI/ML platform"],
        "challenges": ["Snowflake competition", "enterprise adoption"],
        "insider_phrases": ["Lakehouse", "data + AI", "open source"],
        "hiring_priorities": ["Platform", "ML/AI", "Security", "Enterprise"],
        "products_to_reference": ["Lakehouse", "Delta Lake", "MLflow", "Unity Catalog"],
    },
    "snowflake": {
        "culture_keywords": ["Put Customers First", "Integrity Always", "Think Big", "Be Excellent"],
        "tech_focus": ["Data Cloud", "Data Sharing", "Snowpark", "Data Marketplace"],
        "engineering_values": ["engineering excellence", "customer focus", "innovation"],
        "recent_focus": ["Snowpark", "Native apps", "AI/ML features"],
        "challenges": ["Databricks competition", "consumption concerns"],
        "insider_phrases": ["Data Cloud", "Snowpark", "data sharing economy"],
        "hiring_priorities": ["Platform", "ML/AI", "Security", "Performance"],
        "products_to_reference": ["Data Cloud", "Snowpark", "Marketplace", "Cortex"],
    },
}

DEFAULT_COMPANY_INTELLIGENCE: Dict[str, Any] = {
    "culture_keywords": ["innovation", "collaboration", "excellence", "customer focus"],
    "tech_focus": ["scalable systems", "data-driven decisions", "modern architecture"],
    "engineering_values": ["code quality", "team collaboration", "continuous learning"],
    "recent_focus": ["digital transformation", "AI/ML adoption", "cloud migration"],
    "challenges": ["scaling efficiently", "talent retention", "market competition"],
    "insider_phrases": [],
    "hiring_priorities": ["engineering", "product", "data"],
    "products_to_reference": [],
}


def get_company_intelligence(company_name: str) -> Dict[str, Any]:
    company_lower = (company_name or "").lower().strip()
    for key, intel in COMPANY_INTELLIGENCE.items():
        if key in company_lower or company_lower in key:
            return intel
    for key, intel in COMPANY_INTELLIGENCE.items():
        if any(word in company_lower for word in key.split()):
            return intel
    return DEFAULT_COMPANY_INTELLIGENCE


# ============================================================
# 💡 VALUE PROPOSITION GENERATOR
# ============================================================

VALUE_PROPOSITIONS: Dict[str, List[str]] = {
    "ml_engineer": [
        "bring production ML systems from prototype to scale",
        "bridge the gap between research insights and deployed models",
        "build data pipelines that enable rapid experimentation",
        "implement ML infrastructure that accelerates team velocity",
    ],
    "data_scientist": [
        "translate complex data patterns into actionable business insights",
        "design experiments that drive measurable product improvements",
        "build analytical frameworks that inform strategic decisions",
        "develop predictive models that optimize key metrics",
    ],
    "software_engineer": [
        "architect systems that scale gracefully under load",
        "write maintainable code that teams can build upon",
        "drive technical decisions with long-term thinking",
        "build infrastructure that enables product velocity",
    ],
    "data_engineer": [
        "design data architectures that support real-time analytics",
        "build pipelines that ensure data quality at scale",
        "create infrastructure that democratizes data access",
        "implement systems that reduce time-to-insight",
    ],
    "default": [
        "contribute technical depth with collaborative spirit",
        "drive measurable impact through systematic problem-solving",
        "bring both execution capability and strategic thinking",
        "build solutions that balance innovation with reliability",
    ],
}


def get_value_propositions(role: str) -> List[str]:
    role_lower = (role or "").lower()
    if "ml" in role_lower or "machine learning" in role_lower:
        return VALUE_PROPOSITIONS["ml_engineer"]
    elif "data scientist" in role_lower or "analytics" in role_lower:
        return VALUE_PROPOSITIONS["data_scientist"]
    elif "data engineer" in role_lower:
        return VALUE_PROPOSITIONS["data_engineer"]
    elif "software" in role_lower or "backend" in role_lower or "fullstack" in role_lower:
        return VALUE_PROPOSITIONS["software_engineer"]
    return VALUE_PROPOSITIONS["default"]


# ============================================================
# 🔒 LaTeX & Text Utilities
# ============================================================

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
    if not text:
        return ""
    text = text.replace("&", " and ")
    repl = {
        "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\string~", "^": r"\string^",
        "\\": r"\textbackslash{}",
    }
    out = "".join(repl.get(ch, ch) for ch in text)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def _strip_academic_content(text: str) -> str:
    """Remove all academic mentions — GPA, graduation, coursework, degree dates."""
    if not text:
        return ""
    text = re.sub(r"\bGPA\b[:\s]*\d+(\.\d+)?(/\d+(\.\d+)?)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+(\.\d+)?\s*(GPA|CGPA)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(graduat(ed?|ing|ion))\s*(in|from|date)?\s*\d{4}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(class of|expected|graduating)\s*\d{4}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(relevant\s+)?coursework\b[:\s]*[^.]*\.", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcourses?\s+(include|including|such as)[^.]*\.", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(bachelor'?s?|master'?s?|ph\.?d\.?|b\.?s\.?|m\.?s\.?)\s*(degree)?\s*(in\s+\w+)?\s*,?\s*\d{4}", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\buniversity[^,]*,?\s*\d{4}", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcollege[^,]*,?\s*\d{4}", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\.\s*\.", ".", text)
    return text.strip()


def _strip_star_labels(text: str) -> str:
    if not text:
        return ""
    text = re.sub(
        r"(?i)\(\s*(?:situation|task|actions?|result(?:\s+and\s+impact)?|impact)"
        r"(?:\s*/\s*(?:task|actions?|result|impact))?\s*\)",
        "", text,
    )
    text = re.sub(
        r"(?im)^\s*(?:situation(?:\s*/\s*task)?|task|actions?|result(?:\s+and\s+impact)?|impact)\s*[:\-]\s*",
        "", text,
    )
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_body_whitespace(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"([A-Za-z])\s*\n\s*([A-Za-z])", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _debullettify_and_dedash(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\(\s*[0-9]{1,2}\s*\)\s*", "", text)
    text = re.sub(r"(^|[.?!]\s+)\d{1,2}[.)]\s*", r"\1", text)
    text = re.sub(r"\s*(?:—|–|--)\s*", ", ", text)
    text = re.sub(r"\s-\s", ", ", text)
    text = re.sub(r"\s*,\s*,\s*", ", ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _remove_duplicate_sentences(text: str) -> str:
    """Remove duplicate or near-duplicate sentences."""
    if not text:
        return ""
    paragraphs = text.split("\n\n")
    seen: Set[str] = set()
    out_paras = []
    for para in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", para.strip())
        unique = []
        for s in sentences:
            key = re.sub(r"\s+", " ", s.lower().strip())
            if key and key not in seen:
                seen.add(key)
                unique.append(s)
        if unique:
            out_paras.append(" ".join(unique))
    return "\n\n".join(out_paras)


def _postprocess_body(text: str) -> str:
    text = secure_tex_input(text or "")
    text = _strip_academic_content(text)
    text = _strip_star_labels(text)
    text = _normalize_body_whitespace(text)
    text = _debullettify_and_dedash(text)
    text = _remove_duplicate_sentences(text)
    return _latex_escape_light(text)


async def chat_text(system: str, user: str, model: str) -> str:
    client = _get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()


async def chat_json(user_prompt: str, model: str) -> dict:
    client = _get_openai_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
        )
        content = (resp.choices[0].message.content or "").strip()
        try:
            return json.loads(content)
        except Exception:
            return _json_from_text(content, {})
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
        )
        content = (resp.choices[0].message.content or "").strip()
        return _json_from_text(content, {})


# ============================================================
# 📍 EMPLOYER ADDRESS EXTRACTION  ← NEW
# ============================================================

async def extract_employer_address(jd_text: str, company: str) -> Dict[str, str]:
    """
    Extract employer contact/address details from the JD.
    Returns a dict with keys: name, title, street, city_state, country.
    All values may be empty strings — callers must handle that gracefully.
    """
    prompt = f"""Extract employer/recruiter contact and address details from this job description.

COMPANY: {company}

JOB DESCRIPTION:
{jd_text[:3000]}

Return STRICT JSON (use empty string "" for any field not found):
{{
    "hiring_manager_name": "full name if explicitly stated, else empty string",
    "hiring_manager_title": "title like Recruiter, Engineering Manager, etc., else empty string",
    "street_address": "street address of office if mentioned, else empty string",
    "city_state": "City, State or City, Country if mentioned, else empty string",
    "country": "country if explicitly mentioned, else empty string"
}}

IMPORTANT: Only return values that are EXPLICITLY stated in the JD. Do NOT invent or assume addresses."""

    try:
        data = await chat_json(prompt, model=_EXTRACT_MODEL)
        return {
            "hiring_manager_name":  (data.get("hiring_manager_name")  or "").strip(),
            "hiring_manager_title": (data.get("hiring_manager_title") or "").strip(),
            "street_address":       (data.get("street_address")       or "").strip(),
            "city_state":           (data.get("city_state")           or "").strip(),
            "country":              (data.get("country")              or "").strip(),
        }
    except Exception as e:
        log_event("employer_address_extract_fail", {"error": str(e)})
        return {
            "hiring_manager_name": "", "hiring_manager_title": "",
            "street_address": "", "city_state": "", "country": "",
        }


def _build_employer_address_block(company: str, addr: Dict[str, str]) -> str:
    """
    Build the employer address block for the LaTeX template.
    Returns an empty string if no real address info is available.
    This block sits ABOVE the salutation line.
    """
    lines: List[str] = []

    name  = addr.get("hiring_manager_name", "").strip()
    title = addr.get("hiring_manager_title", "").strip()
    street = addr.get("street_address", "").strip()
    city_state = addr.get("city_state", "").strip()
    country = addr.get("country", "").strip()

    # Only add a block if we have at least one real address component
    has_address = bool(street or city_state or country)
    has_contact = bool(name or title)

    if not has_address and not has_contact:
        return ""  # Return empty — template must NOT show placeholder text

    # Company name
    lines.append(_latex_escape_light(company))

    # Contact person
    if name:
        lines.append(_latex_escape_light(name))
    if title:
        lines.append(_latex_escape_light(title))

    # Address lines
    if street:
        lines.append(_latex_escape_light(street))
    if city_state:
        lines.append(_latex_escape_light(city_state))
    if country:
        lines.append(_latex_escape_light(country))

    return " \\\\\n".join(lines)


def _build_salutation(addr: Dict[str, str]) -> str:
    """
    Build the salutation line.
    Uses the hiring manager name if available, else generic.
    """
    name = (addr.get("hiring_manager_name") or "").strip()
    if name:
        # Use first name only for warmth, or full name for formality
        first_name = name.split()[0]
        return f"Dear {_latex_escape_light(first_name)},"
    return "Dear Hiring Manager,"


# ============================================================
# 🧠 DEEP COMPANY INTELLIGENCE EXTRACTION
# ============================================================

async def extract_deep_company_intel(
    jd_text: str,
    company: str,
    role: str,
) -> Dict[str, Any]:
    base_intel = get_company_intelligence(company)

    prompt = f"""Analyze this job description and extract SPECIFIC details.

COMPANY: {company}
ROLE: {role}

JOB DESCRIPTION:
{jd_text[:4000]}

Return STRICT JSON:
{{
    "team_name": "specific team/org name if mentioned, else empty string",
    "team_mission": "what this specific team does, else empty string",
    "tech_stack": ["specific technologies explicitly mentioned"],
    "key_projects": ["specific projects/products mentioned"],
    "business_impact": "what business problem this role solves",
    "unique_challenges": ["specific technical challenges mentioned"],
    "required_expertise": ["must-have skills"],
    "insider_terminology": ["company-specific terms used in the JD"]
}}

Extract REAL details from the JD only. Do not invent."""

    try:
        jd_intel = await chat_json(prompt, model=_INTELLIGENCE_MODEL)
    except Exception as e:
        log_event("intel_extraction_fail", {"error": str(e)})
        jd_intel = {}

    return {
        **base_intel,
        "team_name":           jd_intel.get("team_name", ""),
        "team_mission":        jd_intel.get("team_mission", ""),
        "jd_tech_stack":       jd_intel.get("tech_stack", []),
        "key_projects":        jd_intel.get("key_projects", []),
        "business_impact":     jd_intel.get("business_impact", ""),
        "unique_challenges":   jd_intel.get("unique_challenges", []),
        "required_expertise":  jd_intel.get("required_expertise", []),
        "insider_terminology": jd_intel.get("insider_terminology", []) + base_intel.get("insider_phrases", []),
    }


async def extract_resume_highlights(resume_text: str) -> Dict[str, Any]:
    if not (resume_text or "").strip():
        return {"top_achievements": [], "technical_skills": [], "project_highlights": [], "quantified_results": []}

    prompt = f"""Extract the MOST IMPRESSIVE PROFESSIONAL highlights from this resume.

RESUME:
{resume_text[:5000]}

Return STRICT JSON:
{{
    "top_achievements": ["3-5 most impressive WORK achievements with context"],
    "technical_skills": ["key technical skills demonstrated in work/projects"],
    "leadership_signals": ["leadership/ownership examples from work"],
    "quantified_results": ["results with numbers/metrics from work only"],
    "company_names": ["companies worked at"],
    "project_highlights": ["key projects with measurable outcomes"]
}}

IMPORTANT:
- Focus ONLY on work experience and projects
- DO NOT include GPA, graduation dates, coursework, or academic achievements
- DO NOT mention university names unless for research WORK
- Professional and specific only"""

    try:
        return await chat_json(prompt, model=_INTELLIGENCE_MODEL)
    except Exception:
        return {"top_achievements": [], "technical_skills": [], "project_highlights": [], "quantified_results": []}


async def extract_company_role(jd_text: str) -> Tuple[str, str]:
    jd_excerpt = (jd_text or "").strip()[:5000]
    prompt = (
        "Extract the company name and exact role title from the job description.\n"
        'Return STRICT JSON: {"company":"…","role":"…"}.\n'
        f"JD:\n{jd_excerpt}"
    )
    try:
        data = await chat_json(prompt, model=_EXTRACT_MODEL)
        company = (data.get("company") or "Company").strip()
        role    = (data.get("role")    or "Role").strip()
        return company, role
    except Exception as e:
        log_event("coverletter_extract_fail", {"error": str(e)})
        return "Company", "Role"


# ============================================================
# 📝 AUTHENTIC COVER LETTER DRAFTING
# ============================================================

_LENGTH_BANDS: Dict[str, Tuple[int, int]] = {
    "short":    (120, 180),
    "standard": (200, 300),
    "long":     (320, 420),
}

_BUZZ_BANNED = [
    "passionate", "dynamic", "cutting edge", "team player", "synergy",
    "results-driven", "fast-paced", "leverage synergies", "mission inspires me",
    "innovative work", "perfect fit", "dream job", "always wanted to",
    "since childhood", "grateful for any opportunity", "humbly request",
    "excited to apply", "thrilled", "honored", "privileged", "astonish",
    "astonishing", "blown away", "game-changer", "revolutionary",
]

_ACADEMIC_BANNED = [
    "gpa", "cgpa", "graduated", "graduating", "graduation", "coursework",
    "courses", "degree", "bachelor", "master", "phd", "university studies",
    "academic", "transcript", "cum laude", "dean's list", "honors",
]

_WORD       = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\./_+]*")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS  = set(
    "a an the and or but if while for with to of in on by from as at "
    "into over under is are was were be been being this that these those "
    "i you he she they we it".split()
)


def _clean_text_local(s: str, banned_phrases: Optional[List[str]] = None) -> str:
    txt = (s or "").replace("&", " and ").replace("—", ", ").replace("–", ", ").strip()
    txt = re.sub(r"^\s*(?:[#`>\-\*•]|\d+[.)])\s+", "", txt, flags=re.MULTILINE)
    banned = set((banned_phrases or []) + _BUZZ_BANNED + _ACADEMIC_BANNED)
    for b in sorted(banned, key=len, reverse=True):
        txt = re.sub(rf"\b{re.escape(b)}\b", "", txt, flags=re.IGNORECASE)
    txt = _strip_academic_content(txt)
    txt = re.sub(r"[\[\]\{\}]+", "", txt)
    txt = re.sub(r"\s+,", ",", txt)
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    return txt


def _enforce_word_band_local(text: str, length: str) -> str:
    lo, hi = _LENGTH_BANDS.get(length, (200, 300))
    words = text.split()
    if lo <= len(words) <= hi:
        return text
    sentences = _SENT_SPLIT.split(text.strip())
    out: List[str] = []
    for s in sentences:
        candidate = " ".join(out + [s]).strip()
        if len(candidate.split()) <= hi:
            out.append(s)
        else:
            break
    return " ".join(out).strip()


def _shape_paragraphs(text: str, mode: str) -> str:
    sents = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    if not sents:
        return text.strip()
    if mode == "short":
        cut = max(1, min(len(sents) - 1, len(sents) // 3))
        return " ".join(sents[:cut]).strip() + "\n\n" + " ".join(sents[cut:]).strip()
    n  = len(sents)
    i1 = max(1, min(n - 2, n // 5))
    i2 = max(i1 + 1, min(n - 1, (n * 4) // 5))
    return (
        " ".join(sents[:i1]).strip() + "\n\n" +
        " ".join(sents[i1:i2]).strip() + "\n\n" +
        " ".join(sents[i2:]).strip()
    )


async def draft_killer_cover_body(
    jd_text: str,
    resume_text: str,
    company: str,
    role: str,
    tone: str,
    length: str,
    company_intel: Dict[str, Any],
    resume_highlights: Dict[str, Any],
) -> str:
    tone   = (tone   or "balanced").strip().lower()
    length = (length or "standard").strip().lower()
    if length not in _LENGTH_BANDS:
        length = "standard"

    has_resume = bool((resume_text or "").strip())

    tone_guidance = {
        "confident":     "confident and direct, but not arrogant",
        "balanced":      "professional yet conversational",
        "humble":        "genuine and thoughtful",
        "conversational":"natural and authentic",
    }.get(tone, "professional yet conversational")

    killer_hook        = get_killer_hook(company, "product_insights")
    insider_hook       = get_killer_hook(company, "insider_knowledge")
    culture_keywords   = company_intel.get("culture_keywords", [])
    tech_focus         = company_intel.get("tech_focus", [])
    products           = company_intel.get("products_to_reference", [])
    business_impact    = company_intel.get("business_impact", "")
    team_name          = company_intel.get("team_name", "")

    top_achievements   = resume_highlights.get("top_achievements", [])
    technical_skills   = resume_highlights.get("technical_skills", [])
    quantified_results = resume_highlights.get("quantified_results", [])

    length_hint = {
        "short":    "Target 150-180 words in 2 paragraphs.",
        "standard": "Target 220-280 words in 3 paragraphs.",
        "long":     "Target 350-400 words in 3-4 paragraphs.",
    }[length]

    sys_prompt = f"""You are writing an authentic, compelling cover letter for {company}.

TONE: {tone_guidance}

GOAL: Write naturally, demonstrating genuine understanding of {company} and connecting your experience to their needs.

COMPANY CONTEXT (reference naturally if relevant):
- Culture: {', '.join(culture_keywords[:3]) if culture_keywords else 'professional excellence'}
- Tech focus: {', '.join(tech_focus[:3]) if tech_focus else 'modern technology'}
- Products: {', '.join(products[:3]) if products else company + ' products'}
{f'- Inspiration: {killer_hook}' if killer_hook else ''}
{f'- Culture insight: {insider_hook}' if insider_hook else ''}
{f'- Team: {team_name}' if team_name else ''}
{f'- Business impact: {business_impact}' if business_impact else ''}

YOUR STRENGTHS (WORK EXPERIENCE ONLY):
- Achievements: {'; '.join(top_achievements[:2]) if top_achievements else 'strong professional background'}
- Skills: {', '.join(technical_skills[:5]) if technical_skills else 'relevant technical skills'}
- Results: {'; '.join(quantified_results[:2]) if quantified_results else 'measurable impact'}

STRUCTURE:

PARAGRAPH 1 — Opening (3-4 sentences):
- Why this role at {company} genuinely interests you
- Reference something specific about their work or approach
- Connect your background naturally
- Be specific but AUTHENTIC

PARAGRAPH 2 — Evidence (4-5 sentences):
- Your strongest relevant WORK achievement
- How your experience maps to their needs
- Specific technical accomplishments (not academic)
- Demonstrate problem-solving and quantified impact

PARAGRAPH 3 — Forward + Close (2-3 sentences):
- What you would contribute in the first few months
- Confident, not presumptuous close
- Express interest in a conversation

RULES:
1. NO academic content (GPA, graduation, coursework, degree)
2. Be specific but AUTHENTIC — no fake insider knowledge
3. {length_hint}
4. Natural tone — write like explaining your interest to a colleague
5. No clichés: "passionate", "dream job", "perfect fit", "excited", "astonishing"
6. Professional experience only
7. First-person singular
8. Use "and" not "&". No em-dashes.
9. No duplicate sentences.

OUTPUT: Body paragraphs ONLY — no salutation, no signature."""

    user_prompt = f"""Write the cover letter body for:

ROLE: {role}
COMPANY: {company}

JOB DESCRIPTION:
{jd_text[:4000]}

{"RESUME (professional experience):" if has_resume else ""}
{resume_text[:4000] if has_resume else "Focus on general professional capability aligned with the JD."}

Write naturally and authentically. Specific where you can, general where you must.
"""

    draft = await chat_text(sys_prompt, user_prompt, model=_DRAFT_MODEL)
    body  = _clean_text_local(draft)
    body  = await _validate_and_repair_authentic(
        body, company, role, jd_text, resume_text,
        company_intel, resume_highlights, length, tone,
    )
    body = _shape_paragraphs(body, length)
    body = _enforce_word_band_local(body, length)
    return _postprocess_body(body)


async def _validate_and_repair_authentic(
    body: str,
    company: str,
    role: str,
    jd_text: str,
    resume_text: str,
    company_intel: Dict[str, Any],
    resume_highlights: Dict[str, Any],
    length: str,
    tone: str,
    max_repairs: int = 2,
) -> str:
    issues: List[str] = []
    body_lower = body.lower()

    for cliche in _BUZZ_BANNED:
        if cliche.lower() in body_lower:
            issues.append(f"Contains cliché: '{cliche}'")

    for academic in _ACADEMIC_BANNED:
        if academic.lower() in body_lower:
            issues.append(f"Contains academic content: '{academic}'")

    if company.lower() not in body_lower:
        issues.append("Company name not mentioned")

    first_para = body.split("\n\n")[0] if body else ""
    if company.lower() not in first_para.lower():
        issues.append("Opening paragraph should mention company")

    if not re.search(r"\b(first|initial|early|would|will|contribute|drive|bring)\b", body_lower):
        issues.append("Missing forward-looking value statement")

    if issues and max_repairs > 0:
        repair_prompt = f"""Rewrite this cover letter to fix these issues:
{chr(10).join(f'- {i}' for i in issues)}

Current draft:
{body}

CRITICAL REQUIREMENTS:
- Company: {company}
- Role: {role}
- Opening paragraph must mention {company} naturally
- NO academic content (GPA, graduation, coursework, university)
- NO clichés (passionate, excited, dream job, astonishing, etc.)
- Add forward-looking contribution if missing
- Keep it {length} length
- Maintain natural, authentic tone
- No duplicate sentences

Return only the improved body paragraphs."""
        try:
            repaired = await chat_text(
                "You are improving a cover letter to be more authentic and professional.",
                repair_prompt,
                model=_DRAFT_MODEL,
            )
            return await _validate_and_repair_authentic(
                _clean_text_local(repaired),
                company, role, jd_text, resume_text,
                company_intel, resume_highlights, length, tone,
                max_repairs - 1,
            )
        except Exception:
            pass

    return body


# ============================================================
# ✨ Humanize via internal service
# ============================================================

async def humanize_text(body_text: str, tone: str) -> str:
    api_base = (getattr(config, "API_BASE_URL", "") or "").rstrip("/") or "http://127.0.0.1:8000"
    url      = f"{api_base}/api/superhuman/rewrite"
    payload  = {"text": body_text, "mode": "coverletter", "tone": tone, "latex_safe": True}
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(url, json=payload)
        r.raise_for_status()
        data   = r.json()
        result = data.get("rewritten") or data.get("text") or body_text
        return _strip_academic_content(result)
    except Exception as e:
        log_event("superhuman_handoff_fail", {"error": str(e)})
        return body_text


# ============================================================
# 📄 Header & Template Injection  ← FIXED employer address
# ============================================================

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
    employer_address_block: str = "",   # ← NEW: rendered block or ""
    salutation: str = "Dear Hiring Manager,",  # ← NEW: computed salutation
) -> str:
    def esc(v: str) -> str:
        return _latex_escape_light(secure_tex_input(v or ""))

    # ── Standard scalar substitutions ──────────────────────────────────────
    subst = {
        "COMPANY":        company,
        "ROLE":           role,
        "CANDIDATE_NAME": candidate,
        "NAME":           candidate,
        "DATE":           date_str,
        "EMAIL":          email,
        "PHONE":          phone,
        "CITYSTATE":      citystate if citystate else "",
        "SALUTATION":     salutation,
    }
    for k, v in subst.items():
        if v or k in ("COMPANY", "ROLE", "CANDIDATE_NAME", "DATE", "SALUTATION"):
            tex = tex.replace(f"{{{{{k}}}}}", esc(v))
            tex = tex.replace(f"%<<{k}>>%", esc(v))
        else:
            tex = tex.replace(f"{{{{{k}}}}}", "")
            tex = tex.replace(f"%<<{k}>>%", "")

    # ── Employer address block ──────────────────────────────────────────────
    # Template may use {{EMPLOYER_ADDRESS}} or %<<EMPLOYER_ADDRESS>>%
    # If we have a real block, inject it. Otherwise remove the placeholder
    # completely so nothing stray appears above "Dear Hiring Manager,".
    if employer_address_block:
        tex = tex.replace("{{EMPLOYER_ADDRESS}}", employer_address_block)
        tex = tex.replace("%<<EMPLOYER_ADDRESS>>%", employer_address_block)
    else:
        # Remove placeholder AND any surrounding LaTeX newline commands
        for placeholder in ("{{EMPLOYER_ADDRESS}}", "%<<EMPLOYER_ADDRESS>>%"):
            # e.g. \noindent{{EMPLOYER_ADDRESS}}\\ or \mbox{...}\n
            tex = re.sub(
                r"[^\n]*" + re.escape(placeholder) + r"[^\n]*\n?",
                "",
                tex,
            )

    # ── \def patterns ──────────────────────────────────────────────────────
    def_patterns = {
        r"(\\def\\Company\{)(.*?)(\})":       company,
        r"(\\def\\Role\{)(.*?)(\})":          role,
        r"(\\def\\CandidateName\{)(.*?)(\})": candidate,
        r"(\\def\\Date\{)(.*?)(\})":          date_str,
    }
    for pat, val in def_patterns.items():
        tex = re.sub(pat, lambda m, v=val: f"{m.group(1)}{esc(v)}{m.group(3)}", tex, flags=re.I)

    return tex


def _inject_between_salutation_and_signoff(base_tex: str, body_tex: str) -> Optional[str]:
    pat = r"(Dear[^\n]*?,\s*\n)([\s\S]*?)(\n\s*Sincerely,\s*\\\\[\s\S]*?$)"
    if re.search(pat, base_tex, flags=re.I):
        return re.sub(pat, lambda m: f"{m.group(1)}{body_tex}\n{m.group(3)}", base_tex, flags=re.I)
    return None


def inject_body_into_template(base_tex: str, body_tex: str) -> str:
    swapped = _inject_between_salutation_and_signoff(base_tex, body_tex)
    if swapped is not None:
        return swapped

    safe_body = re.sub(r"\\documentclass[\s\S]*?\\begin\{document\}", "", body_tex or "", flags=re.I)
    safe_body = re.sub(r"\\end\{document\}\s*$", "", safe_body, flags=re.I).strip()

    anchor_pat = r"(%-+BODY-START-+%)(.*?)(%-+BODY-END-+%)"
    if re.search(anchor_pat, base_tex, flags=re.S):
        return re.sub(anchor_pat, lambda m: f"{m.group(1)}\n{safe_body}\n{m.group(3)}", base_tex, flags=re.S)

    if re.search(r"\\end\{document\}\s*$", base_tex, flags=re.I):
        return re.sub(
            r"\\end\{document\}\s*$",
            lambda _: f"\n{safe_body}\n\\end{{document}}\n",
            base_tex, flags=re.I,
        )

    return base_tex.rstrip() + f"\n\n{safe_body}\n\\end{{document}}\n"


# ============================================================
# 🚀 MAIN ENDPOINT
# ============================================================

@router.post("")
async def generate_coverletter(
    jd_text:       str  = Form(...),
    resume_tex:    str  = Form(""),
    use_humanize:  bool = Form(True),
    tone:          str  = Form("balanced"),
    length:        str  = Form("standard"),
):
    """
    Generate an authentic, compelling cover letter.

    Improvements in v3.3.0:
    - Employer address block extracted from JD when available
    - Address block OMITTED entirely if not found (no placeholder text)
    - Salutation uses hiring manager name when found, else "Dear Hiring Manager,"
    - Duplicate sentence detection
    - Score-based repair validation
    """
    if not (config.OPENAI_API_KEY or "").strip():
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY missing.")
    if not (jd_text or "").strip():
        raise HTTPException(status_code=400, detail="jd_text is required.")

    # ── 1. Extract company + role ───────────────────────────────────────────
    company, role = await extract_company_role(jd_text)
    log_event("coverletter_start", {"company": company, "role": role})

    # ── 2. Extract employer address (run in parallel with intel) ───────────
    employer_addr   = await extract_employer_address(jd_text, company)
    employer_block  = _build_employer_address_block(company, employer_addr)
    salutation      = _build_salutation(employer_addr)

    log_event("employer_address_extracted", {
        "has_address": bool(employer_block),
        "salutation": salutation,
    })

    # ── 3. Deep company intelligence ────────────────────────────────────────
    company_intel = await extract_deep_company_intel(jd_text, company, role)
    log_event("company_intel_extracted", {
        "company": company,
        "hooks_available": bool(get_killer_hook(company)),
        "insider_terms_count": len(company_intel.get("insider_terminology", [])),
    })

    # ── 4. Resume highlights ─────────────────────────────────────────────────
    resume_highlights = await extract_resume_highlights(resume_tex)

    # ── 5. Draft body ─────────────────────────────────────────────────────────
    body_text = await draft_killer_cover_body(
        jd_text=jd_text,
        resume_text=resume_tex,
        company=company,
        role=role,
        tone=tone,
        length=length,
        company_intel=company_intel,
        resume_highlights=resume_highlights,
    )

    # ── 6. Humanize ───────────────────────────────────────────────────────────
    if use_humanize:
        body_text = await humanize_text(body_text, tone)
        body_text = _postprocess_body(body_text)

    # ── 7. Load template ──────────────────────────────────────────────────────
    base_path = config.BASE_COVERLETTER_PATH
    try:
        with open(base_path, encoding="utf-8") as f:
            base_tex = f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template not found: {base_path}")

    # ── 8. Fill header (with fixed employer address logic) ────────────────────
    today_str       = datetime.now().strftime("%B %d, %Y")
    candidate       = getattr(config, "CANDIDATE_NAME",      "Sri Akash Kadali")
    applicant_email = getattr(config, "APPLICANT_EMAIL",     "kadali18@umd.edu")
    applicant_phone = getattr(config, "APPLICANT_PHONE",     "+1 240-726-9356")
    applicant_city  = getattr(config, "APPLICANT_CITYSTATE", "")

    base_tex = _fill_header_fields(
        base_tex,
        company=company,
        role=role,
        candidate=candidate,
        date_str=today_str,
        email=applicant_email,
        phone=applicant_phone,
        citystate=applicant_city,
        employer_address_block=employer_block,  # "" if nothing found
        salutation=salutation,
    )

    # ── 9. Inject body ────────────────────────────────────────────────────────
    try:
        injected = inject_body_into_template(base_tex, body_text)
    except re.error as e:
        log_event("inject_error", {"error": str(e)})
        injected = f"{base_tex}\n\n{body_text}\n"
        if not injected.strip().endswith("\\end{document}"):
            injected += "\n\\end{document}\n"

    # ── 10. Compile ───────────────────────────────────────────────────────────
    final_tex = render_final_tex(injected)
    pdf_bytes = compile_latex_safely(final_tex) or b""
    pdf_b64   = base64.b64encode(pdf_bytes).decode("utf-8")

    # ── 11. Save outputs ──────────────────────────────────────────────────────
    out_pdf_path = config.get_sample_coverletter_pdf_path(company, role)
    ensure_dir(out_pdf_path.parent)
    if pdf_bytes:
        out_pdf_path.write_bytes(pdf_bytes)

    company_slug = safe_filename(company)
    role_slug    = safe_filename(role)
    context_key  = f"{company_slug}__{role_slug}"

    ctx_dir  = config.get_contexts_dir()
    ensure_dir(ctx_dir)
    ctx_path = ctx_dir / f"{context_key}.json"

    existing: Dict[str, Any] = {}
    if ctx_path.exists():
        try:
            existing = json.loads(ctx_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    context_payload = {
        **existing,
        "key":     context_key,
        "company": company,
        "role":    role,
        "jd_text": jd_text,
        "employer_address": {
            "block_used":      bool(employer_block),
            "salutation_used": salutation,
            **employer_addr,
        },
        "company_intel": {
            "culture_keywords":  company_intel.get("culture_keywords", []),
            "tech_focus":        company_intel.get("tech_focus", []),
            "insider_terms":     company_intel.get("insider_terminology", [])[:5],
            "hook_used":         bool(get_killer_hook(company)),
        },
        "cover_letter": {
            "tex":       final_tex,
            "pdf_path":  str(out_pdf_path),
            "pdf_b64":   pdf_b64,
            "tone":      tone,
            "length":    length,
            "humanized": bool(use_humanize),
        },
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    ctx_path.write_text(json.dumps(context_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    log_event("coverletter_generated", {
        "company":         company,
        "role":            role,
        "tone":            tone,
        "length":          length,
        "humanized":       use_humanize,
        "hook_used":       bool(get_killer_hook(company)),
        "address_used":    bool(employer_block),
        "salutation":      salutation,
        "chars":           len(body_text),
    })

    return JSONResponse({
        "company":       company,
        "role":          role,
        "tone":          tone,
        "use_humanize":  use_humanize,
        "tex_string":    final_tex,
        "pdf_base64":    pdf_b64,
        "pdf_path":      str(out_pdf_path),
        "context_key":   context_key,
        "context_path":  str(ctx_path),
        "employer_address_used": bool(employer_block),
        "salutation_used":       salutation,
        "company_intel_used": {
            "culture_keywords": company_intel.get("culture_keywords", [])[:3],
            "tech_focus":       company_intel.get("tech_focus", [])[:3],
            "insider_terms":    company_intel.get("insider_terminology", [])[:3],
            "hook_available":   bool(get_killer_hook(company)),
        },
        "id":        context_key,
        "memory_id": context_key,
    })