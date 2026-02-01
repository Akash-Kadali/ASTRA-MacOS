# ============================================================
#  HIREX v3.1.0 — KILLER Cover Letter Generation
#  ------------------------------------------------------------
#  KEY UPGRADES:
#   • ASTONISHING company-specific opening hooks
#   • NO academic mentions (GPA, graduation, coursework)
#   • Deep product knowledge signals
#   • Recent news/initiative references
#   • Problem-aware hooks that show insider understanding
#   • Recruiter-stopping first sentences
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


_DEFAULT_OAI_MODEL = "gpt-4o-mini"
_EXTRACT_MODEL = getattr(config, "COVERLETTER_EXTRACT_MODEL", None) or _DEFAULT_OAI_MODEL
_DRAFT_MODEL = getattr(config, "COVERLETTER_MODEL", None) or _DEFAULT_OAI_MODEL
_INTELLIGENCE_MODEL = "gpt-4o-mini"

_DISABLE_SHARED_INJECTOR = True


# ============================================================
# 🎯 ASTONISHING COMPANY-SPECIFIC HOOKS
# ============================================================

COMPANY_KILLER_HOOKS = {
    "netflix": {
        "product_insights": [
            "The way Netflix's recommendation engine surfaced 'Stranger Things' to me before it became a phenomenon showed me the power of personalization at scale",
            "I've studied how Netflix's A/B testing framework drives decisions on everything from thumbnail images to entire UI redesigns",
            "Netflix's 2023 shift to an ad-supported tier while maintaining personalization quality is exactly the ML challenge I want to solve",
            "The technical blog post on how Netflix reduced streaming rebuffer rates using predictive prefetching changed how I think about ML in production"
        ],
        "insider_knowledge": [
            "Netflix's 'context not control' philosophy mirrors how I've built my most successful ML systems",
            "The challenge of maintaining recommendation quality across 230M+ subscribers while expanding into gaming is fascinating",
            "Netflix's experimentation platform running thousands of A/B tests simultaneously is the scale I want to work at"
        ],
        "recent_moves": [
            "Netflix's expansion into live events and gaming presents new personalization challenges",
            "The password-sharing crackdown combined with ad-tier launch shows Netflix is optimizing for sustainable growth"
        ]
    },
    "google": {
        "product_insights": [
            "When Google Search started answering my queries with AI-generated summaries, I recognized the massive infrastructure challenge behind it",
            "I've traced how Google's PageRank evolved into the ML-powered ranking system that now processes billions of queries",
            "The Gemini integration across Google products represents a system design challenge at unprecedented scale"
        ],
        "insider_knowledge": [
            "Google's culture of 'launch and iterate' while maintaining reliability for billions of users is the balance I strive for",
            "The technical rigor of Google's design doc culture has influenced how I approach system architecture",
            "Google's commitment to responsible AI development, especially after the Gemini launch learnings, aligns with my values"
        ],
        "recent_moves": [
            "Google's response to the AI search disruption by integrating Gemini shows strategic technical execution",
            "The Cloud growth trajectory competing with AWS while leveraging AI differentiation is compelling"
        ]
    },
    "meta": {
        "product_insights": [
            "Watching Meta's Feed ranking evolve from chronological to ML-driven engagement optimization taught me about recommendation systems at scale",
            "The technical challenge of Reels competing with TikTok's algorithm while serving different user intents is fascinating",
            "Meta's integrity systems detecting misinformation across billions of posts daily is the ML scale I want to work at"
        ],
        "insider_knowledge": [
            "Meta's 'Move Fast' culture combined with the responsibility of serving 3B+ users requires exactly the balance I bring",
            "The PyTorch ecosystem that Meta open-sourced has been central to my ML work",
            "Meta's pivot from metaverse investment to AI-first strategy shows adaptive technical leadership"
        ],
        "recent_moves": [
            "Threads reaching 100M users in days showed Meta's infrastructure can scale anything",
            "The Llama model releases democratizing AI while Meta builds commercial applications is strategic"
        ]
    },
    "amazon": {
        "product_insights": [
            "Amazon's 'Customers who bought this' changed e-commerce, and I've studied how that simple UX hides sophisticated ML",
            "The technical challenge of same-day delivery optimization across millions of products is operations research at its finest",
            "AWS's ability to serve both startups and enterprises with the same infrastructure is product-market fit I admire"
        ],
        "insider_knowledge": [
            "Amazon's 'working backwards' from the customer aligns perfectly with how I approach ML system design",
            "The two-pizza team structure enabling Amazon's pace of innovation is how I've seen the best teams operate",
            "Amazon's leadership principle of 'Dive Deep' matches my belief that technical leaders must understand implementation details"
        ],
        "recent_moves": [
            "AWS Bedrock making foundation models accessible shows Amazon's platform-first thinking",
            "Amazon's healthcare expansion with One Medical acquisition opens new ML applications"
        ]
    },
    "microsoft": {
        "product_insights": [
            "Copilot's integration across Microsoft 365 represents the largest AI deployment in enterprise software history",
            "Azure's growth from distant third to genuine AWS competitor shows execution at scale",
            "The GitHub Copilot accuracy improvements I've tracked show Microsoft's ML iteration velocity"
        ],
        "insider_knowledge": [
            "Microsoft's 'growth mindset' transformation under Satya Nadella created the culture I want to join",
            "The OpenAI partnership giving Microsoft AI leadership while Google scrambled was strategic brilliance",
            "Microsoft's ability to ship AI features to enterprise customers who actually pay is unique"
        ],
        "recent_moves": [
            "Copilot becoming the unified AI interface across all Microsoft products is ambitious UX",
            "The Activision acquisition combined with AI for gaming opens new technical frontiers"
        ]
    },
    "apple": {
        "product_insights": [
            "Apple Intelligence running ML models on-device while maintaining privacy is the technical constraint I find most interesting",
            "The Neural Engine evolution from A11 to M-series shows Apple's decade-long ML silicon investment",
            "Face ID's ability to work across lighting conditions with zero cloud dependency is elegant ML engineering"
        ],
        "insider_knowledge": [
            "Apple's obsession with privacy-preserving ML aligns with my belief that user trust is non-negotiable",
            "The vertical integration allowing Apple to optimize ML from silicon to software is unique",
            "Apple's 'say no to 1000 things' philosophy producing focused products resonates with my approach"
        ],
        "recent_moves": [
            "Vision Pro's spatial computing ML challenges are the next frontier I want to explore",
            "Apple Intelligence's delayed rollout shows they prioritize quality over speed"
        ]
    },
    "stripe": {
        "product_insights": [
            "Stripe's API design is so elegant that it became the standard all fintech APIs are measured against",
            "The fraud detection challenge of approving good transactions instantly while blocking fraud at scale is ML I understand",
            "Stripe Atlas enabling company formation globally shows infrastructure thinking beyond payments"
        ],
        "insider_knowledge": [
            "Stripe's written culture where ideas win based on rigor, not seniority, is where I thrive",
            "The 'increase the GDP of the internet' mission is ambitious enough to attract my commitment",
            "Stripe's developer-first approach building products that engineers love to integrate matches my values"
        ],
        "recent_moves": [
            "Stripe's embedded finance push making every platform a fintech company expands the problem space",
            "The revenue recognition and tax products show Stripe solving the entire financial stack"
        ]
    },
    "airbnb": {
        "product_insights": [
            "Airbnb's search ranking balancing guest preferences with host fairness is a multi-objective optimization I find fascinating",
            "The pricing suggestions helping hosts optimize revenue while maintaining platform trust is elegant ML",
            "Trust & Safety detecting problematic listings across millions of properties globally is scale I want to work at"
        ],
        "insider_knowledge": [
            "Airbnb's 'Belong Anywhere' mission creating genuine human connection through technology resonates with me",
            "The design-driven culture where engineers and designers collaborate deeply produces better products",
            "Airbnb's transparent culture with open financials builds the trust I value in organizations"
        ],
        "recent_moves": [
            "Airbnb Experiences expansion beyond stays diversifies the ML challenges",
            "The AI trip planning features hint at Airbnb becoming a travel intelligence platform"
        ]
    },
    "uber": {
        "product_insights": [
            "Uber's marketplace balancing rider wait times with driver earnings in real-time is optimization at massive scale",
            "The ETA prediction accuracy improvements I've tracked show ML iteration at city-scale",
            "Surge pricing's ability to balance supply and demand dynamically is economics and ML combined"
        ],
        "insider_knowledge": [
            "Uber's reliability at global scale during peak events like NYE shows infrastructure I want to build",
            "The transition from growth-at-all-costs to profitable operations required technical efficiency I admire",
            "Uber's multi-modal future combining rides, delivery, and freight is a platform play I find compelling"
        ],
        "recent_moves": [
            "Uber's advertising business leveraging rider attention is a new revenue ML challenge",
            "The autonomous vehicle partnerships show Uber preparing for the next transportation era"
        ]
    },
    "linkedin": {
        "product_insights": [
            "LinkedIn's job matching connecting 1B+ professionals with opportunities is ML with real-world career impact",
            "The Feed ranking balancing professional content with engagement is harder than consumer social",
            "Skills-based hiring replacing credential-based hiring is a mission I believe in"
        ],
        "insider_knowledge": [
            "LinkedIn's 'Members First' philosophy in a Microsoft-owned entity shows maintained independence",
            "The Economic Graph vision of mapping the global economy's skills and opportunities is ambitious",
            "LinkedIn Learning's integration with job skills gaps is education meeting employment"
        ],
        "recent_moves": [
            "LinkedIn's AI-powered job descriptions and outreach tools are changing recruiting",
            "The creator monetization push makes LinkedIn a professional media platform"
        ]
    },
    "spotify": {
        "product_insights": [
            "Discover Weekly's ability to surface music I didn't know I'd love is ML personalization at its best",
            "The audio ML challenges of understanding podcasts, music, and audiobooks differently is fascinating",
            "Spotify's collaborative filtering at scale with 500M+ users sets the standard for recommendations"
        ],
        "insider_knowledge": [
            "Spotify's squad model giving teams autonomy while maintaining product coherence is organizational design I admire",
            "The artist-friendly stance while building a sustainable business shows stakeholder balance",
            "Spotify's data-informed but not data-driven culture leaves room for creative product decisions"
        ],
        "recent_moves": [
            "AI DJ creating personalized radio shows is the next evolution of music personalization",
            "Audiobooks expansion competing with Audible diversifies the content ML challenges"
        ]
    },
    "databricks": {
        "product_insights": [
            "The Lakehouse architecture solving the data warehouse vs. data lake debate is elegant technical vision",
            "MLflow becoming the standard for ML experiment tracking shows open-source-first strategy working",
            "Unity Catalog solving data governance across the entire data lifecycle is ambitious"
        ],
        "insider_knowledge": [
            "Databricks' open-source DNA with Delta Lake and MLflow builds genuine community trust",
            "The 'data + AI' positioning as distinct from pure cloud vendors is clear differentiation",
            "Databricks' technical leadership publishing research while building products is rare"
        ],
        "recent_moves": [
            "The Mosaic ML acquisition shows Databricks building the full AI stack",
            "Serverless compute removing infrastructure management overhead accelerates adoption"
        ]
    },
    "snowflake": {
        "product_insights": [
            "Snowflake's consumption-based pricing aligning vendor success with customer value is smart economics",
            "Data sharing without copying data is a technical achievement enabling new business models",
            "Snowpark bringing code to data instead of data to code shows architectural thinking"
        ],
        "insider_knowledge": [
            "Snowflake's engineering excellence producing performance improvements every release builds trust",
            "The Data Cloud vision of connecting data across organizations is ambitious infrastructure",
            "Snowflake's customer obsession reflected in NPS scores is culture I want to join"
        ],
        "recent_moves": [
            "Native apps on Snowflake creating a data application ecosystem expands the platform",
            "The AI/ML features competing with Databricks show strategic expansion"
        ]
    }
}


def get_killer_hook(company: str, hook_type: str = "product_insights") -> str:
    """Get a company-specific killer opening hook."""
    company_lower = company.lower().strip()
    
    for key, hooks in COMPANY_KILLER_HOOKS.items():
        if key in company_lower or company_lower in key:
            hook_list = hooks.get(hook_type, hooks.get("product_insights", []))
            if hook_list:
                return random.choice(hook_list)
    
    return ""


# ============================================================
# 🏢 DEEP COMPANY INTELLIGENCE DATABASE
# ============================================================

COMPANY_INTELLIGENCE = {
    "netflix": {
        "culture_keywords": ["Freedom & Responsibility", "context not control", "highly aligned loosely coupled", "keeper test", "candor"],
        "tech_focus": ["Recommender Systems", "Personalization", "A/B Testing at Scale", "Content Delivery", "Streaming Infrastructure"],
        "engineering_values": ["data-driven decisions", "experimentation culture", "ownership mentality", "impact over activity"],
        "recent_focus": ["gaming expansion", "ad-supported tier", "live events", "content efficiency"],
        "challenges": ["subscriber growth in mature markets", "content cost optimization", "competition from Disney+/HBO"],
        "insider_phrases": ["member experience", "title discovery", "personalization at scale", "streaming quality"],
        "hiring_priorities": ["ML infrastructure", "experimentation platforms", "content algorithms", "data pipelines"],
        "products_to_reference": ["recommendation engine", "A/B testing platform", "streaming infrastructure", "content personalization"]
    },
    "google": {
        "culture_keywords": ["Googleyness", "think big", "user first", "10x thinking", "psychological safety"],
        "tech_focus": ["Scalability", "Distributed Systems", "AI/ML Infrastructure", "Search Quality", "Cloud Platform"],
        "engineering_values": ["code quality", "design docs", "peer review culture", "technical excellence"],
        "recent_focus": ["Gemini AI", "Cloud growth", "Search AI integration", "Pixel ecosystem"],
        "challenges": ["AI competition with OpenAI/Microsoft", "advertising revenue pressure", "regulatory scrutiny"],
        "insider_phrases": ["Noogler", "OKRs", "launch and iterate", "10x improvement"],
        "hiring_priorities": ["AI/ML", "Cloud infrastructure", "Privacy engineering", "Mobile development"],
        "products_to_reference": ["Search", "Gemini", "Cloud Platform", "YouTube", "Android"]
    },
    "meta": {
        "culture_keywords": ["Move Fast", "Be Bold", "Focus on Impact", "Be Open", "Build Social Value"],
        "tech_focus": ["Social Graph", "Ranking Systems", "AR/VR", "Messaging Infrastructure", "Ads Optimization"],
        "engineering_values": ["ship early ship often", "hackathons", "bootcamp culture", "impact metrics"],
        "recent_focus": ["AI assistants", "Threads growth", "Reels monetization", "Llama models"],
        "challenges": ["privacy regulations", "TikTok competition", "metaverse ROI questions"],
        "insider_phrases": ["family of apps", "integrity systems", "social impact", "meaningful connections"],
        "hiring_priorities": ["AI/ML", "Integrity/Safety", "Infrastructure", "AR/VR"],
        "products_to_reference": ["Instagram", "WhatsApp", "Messenger", "Threads", "Llama"]
    },
    "amazon": {
        "culture_keywords": ["Customer Obsession", "Ownership", "Bias for Action", "Dive Deep", "Deliver Results"],
        "tech_focus": ["AWS Services", "Supply Chain ML", "Alexa/Voice", "Retail Optimization", "Logistics"],
        "engineering_values": ["working backwards", "two-pizza teams", "operational excellence", "frugality"],
        "recent_focus": ["AWS AI services", "same-day delivery", "healthcare expansion", "advertising growth"],
        "challenges": ["labor relations", "AWS competition", "retail margins"],
        "insider_phrases": ["PR/FAQ", "6-pager", "bar raiser", "Day 1 mentality", "mechanisms"],
        "hiring_priorities": ["AWS", "ML/AI", "Supply chain", "Advertising"],
        "products_to_reference": ["AWS", "Prime", "Alexa", "One Medical", "Bedrock"]
    },
    "microsoft": {
        "culture_keywords": ["Growth Mindset", "Customer Obsessed", "Diverse and Inclusive", "One Microsoft"],
        "tech_focus": ["Azure Cloud", "Microsoft 365", "AI/Copilot", "Gaming/Xbox", "Developer Tools"],
        "engineering_values": ["learn-it-all not know-it-all", "customer empathy", "responsible AI"],
        "recent_focus": ["Copilot integration", "Azure OpenAI", "Gaming", "Teams platform"],
        "challenges": ["cloud competition with AWS", "AI integration", "gaming market share"],
        "insider_phrases": ["growth mindset", "customer zero", "inclusive design"],
        "hiring_priorities": ["AI/ML", "Azure", "Security", "Developer experience"],
        "products_to_reference": ["Copilot", "Azure", "GitHub", "Teams", "Xbox"]
    },
    "apple": {
        "culture_keywords": ["Think Different", "Simplicity", "Privacy as Human Right", "Excellence"],
        "tech_focus": ["On-Device ML", "Privacy-Preserving AI", "Hardware-Software Integration", "User Experience"],
        "engineering_values": ["attention to detail", "user privacy", "vertical integration", "craftsmanship"],
        "recent_focus": ["Vision Pro", "Apple Intelligence", "Services growth", "Sustainability"],
        "challenges": ["China market", "AI catch-up", "services growth"],
        "insider_phrases": ["DRI", "surprise and delight", "it just works"],
        "hiring_priorities": ["ML on-device", "Privacy engineering", "AR/VR", "Health tech"],
        "products_to_reference": ["iPhone", "Vision Pro", "Apple Intelligence", "Neural Engine"]
    },
    "stripe": {
        "culture_keywords": ["Users First", "Move with Urgency", "Think Rigorously", "Trust and Amplify"],
        "tech_focus": ["Payment Infrastructure", "Financial APIs", "Fraud Detection", "Developer Experience"],
        "engineering_values": ["write like you code", "rigor in thinking", "long-term orientation"],
        "recent_focus": ["Embedded finance", "Global expansion", "Revenue recognition"],
        "challenges": ["fintech competition", "regulatory complexity", "enterprise sales"],
        "insider_phrases": ["increase GDP of internet", "payment rails", "developer love"],
        "hiring_priorities": ["Infrastructure", "ML/Fraud", "Platform", "International"],
        "products_to_reference": ["Payments API", "Radar", "Atlas", "Connect", "Billing"]
    },
    "airbnb": {
        "culture_keywords": ["Belong Anywhere", "Champion the Mission", "Be a Host", "Embrace Adventure"],
        "tech_focus": ["Search & Ranking", "Pricing Algorithms", "Trust & Safety", "Payments"],
        "engineering_values": ["customer empathy", "design-driven", "data-informed"],
        "recent_focus": ["Experiences expansion", "Long-term stays", "AI trip planning"],
        "challenges": ["regulatory battles", "hotel competition", "host supply"],
        "insider_phrases": ["belonging", "host community", "guest journey"],
        "hiring_priorities": ["ML/Search", "Trust & Safety", "Payments", "Mobile"],
        "products_to_reference": ["Search ranking", "Smart Pricing", "Experiences", "AirCover"]
    },
    "uber": {
        "culture_keywords": ["Build Globally", "Celebrate Differences", "Act Like Owners", "Persevere"],
        "tech_focus": ["Marketplace Optimization", "ETA Prediction", "Route Optimization", "Fraud Detection"],
        "engineering_values": ["data-driven", "experimentation", "reliability at scale"],
        "recent_focus": ["Delivery growth", "Advertising", "Freight"],
        "challenges": ["driver supply", "profitability", "regulatory issues"],
        "insider_phrases": ["marketplace balance", "rider experience", "driver earnings"],
        "hiring_priorities": ["ML/Optimization", "Maps", "Marketplace", "Delivery"],
        "products_to_reference": ["Rides", "Eats", "Freight", "Advertising platform"]
    },
    "linkedin": {
        "culture_keywords": ["Members First", "Relationships Matter", "Be Open Honest Constructive", "Act Like Owner"],
        "tech_focus": ["Feed Ranking", "Job Matching", "Graph Systems", "Economic Graph"],
        "engineering_values": ["test and learn", "member value", "data-driven"],
        "recent_focus": ["AI features", "Creator economy", "Skills-based hiring"],
        "challenges": ["engagement growth", "premium conversion", "content quality"],
        "insider_phrases": ["economic graph", "member value", "professional identity"],
        "hiring_priorities": ["AI/ML", "Feed", "Search", "Infrastructure"],
        "products_to_reference": ["Feed", "Jobs", "Learning", "Sales Navigator"]
    },
    "spotify": {
        "culture_keywords": ["Innovative", "Collaborative", "Sincere", "Passionate", "Playful"],
        "tech_focus": ["Audio ML", "Personalization", "Content Delivery", "Creator Tools"],
        "engineering_values": ["squad model", "autonomous teams", "data-informed"],
        "recent_focus": ["Podcasts", "Audiobooks", "AI DJ", "Creator monetization"],
        "challenges": ["profitability", "music licensing costs", "podcast ROI"],
        "insider_phrases": ["Discover Weekly", "audio-first", "creator ecosystem"],
        "hiring_priorities": ["ML/Personalization", "Audio", "Ads", "Payments"],
        "products_to_reference": ["Discover Weekly", "AI DJ", "Wrapped", "Podcast platform"]
    },
    "databricks": {
        "culture_keywords": ["Customer Obsessed", "Unity", "Ownership", "Open Source First"],
        "tech_focus": ["Lakehouse", "Delta Lake", "MLflow", "Spark", "Data Engineering"],
        "engineering_values": ["open source contribution", "technical excellence", "customer impact"],
        "recent_focus": ["Unity Catalog", "Serverless", "AI/ML platform"],
        "challenges": ["Snowflake competition", "enterprise adoption"],
        "insider_phrases": ["Lakehouse", "data + AI", "open source"],
        "hiring_priorities": ["Platform", "ML/AI", "Security", "Enterprise"],
        "products_to_reference": ["Lakehouse", "Delta Lake", "MLflow", "Unity Catalog"]
    },
    "snowflake": {
        "culture_keywords": ["Put Customers First", "Integrity Always", "Think Big", "Be Excellent"],
        "tech_focus": ["Data Cloud", "Data Sharing", "Snowpark", "Data Marketplace"],
        "engineering_values": ["engineering excellence", "customer focus", "innovation"],
        "recent_focus": ["Snowpark", "Native apps", "AI/ML features"],
        "challenges": ["Databricks competition", "consumption concerns"],
        "insider_phrases": ["Data Cloud", "Snowpark", "data sharing economy"],
        "hiring_priorities": ["Platform", "ML/AI", "Security", "Performance"],
        "products_to_reference": ["Data Cloud", "Snowpark", "Marketplace", "Cortex"]
    }
}

DEFAULT_COMPANY_INTELLIGENCE = {
    "culture_keywords": ["innovation", "collaboration", "excellence", "customer focus"],
    "tech_focus": ["scalable systems", "data-driven decisions", "modern architecture"],
    "engineering_values": ["code quality", "team collaboration", "continuous learning"],
    "recent_focus": ["digital transformation", "AI/ML adoption", "cloud migration"],
    "challenges": ["scaling efficiently", "talent retention", "market competition"],
    "insider_phrases": [],
    "hiring_priorities": ["engineering", "product", "data"],
    "products_to_reference": []
}


def get_company_intelligence(company_name: str) -> Dict[str, Any]:
    """Get deep intelligence about a company."""
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

VALUE_PROPOSITIONS = {
    "ml_engineer": [
        "bring production ML systems from prototype to scale",
        "bridge the gap between research insights and deployed models",
        "build data pipelines that enable rapid experimentation",
        "implement ML infrastructure that accelerates team velocity"
    ],
    "data_scientist": [
        "translate complex data patterns into actionable business insights",
        "design experiments that drive measurable product improvements",
        "build analytical frameworks that inform strategic decisions",
        "develop predictive models that optimize key metrics"
    ],
    "software_engineer": [
        "architect systems that scale gracefully under load",
        "write maintainable code that teams can build upon",
        "drive technical decisions with long-term thinking",
        "build infrastructure that enables product velocity"
    ],
    "data_engineer": [
        "design data architectures that support real-time analytics",
        "build pipelines that ensure data quality at scale",
        "create infrastructure that democratizes data access",
        "implement systems that reduce time-to-insight"
    ],
    "default": [
        "contribute technical depth with collaborative spirit",
        "drive measurable impact through systematic problem-solving",
        "bring both execution capability and strategic thinking",
        "build solutions that balance innovation with reliability"
    ]
}


def get_value_propositions(role: str) -> List[str]:
    """Get role-specific value propositions."""
    role_lower = role.lower()
    
    if "ml" in role_lower or "machine learning" in role_lower:
        return VALUE_PROPOSITIONS["ml_engineer"]
    elif "data scientist" in role_lower or "analytics" in role_lower:
        return VALUE_PROPOSITIONS["data_scientist"]
    elif "data engineer" in role_lower:
        return VALUE_PROPOSITIONS["data_engineer"]
    elif "software" in role_lower or "backend" in role_lower or "fullstack" in role_lower:
        return VALUE_PROPOSITIONS["software_engineer"]
    else:
        return VALUE_PROPOSITIONS["default"]


# ============================================================
# 🔬 TECHNICAL DEPTH SIGNALS
# ============================================================

TECHNICAL_DEPTH_SIGNALS = {
    "ml": [
        "implemented custom loss functions for domain-specific optimization",
        "designed feature stores for real-time ML serving",
        "built A/B testing frameworks for model evaluation",
        "optimized inference latency for production deployment"
    ],
    "data": [
        "architected data pipelines processing terabytes daily",
        "implemented data quality frameworks with automated monitoring",
        "designed schema evolution strategies for backward compatibility",
        "built real-time streaming pipelines with exactly-once semantics"
    ],
    "systems": [
        "designed distributed systems handling millions of requests",
        "implemented caching strategies reducing latency significantly",
        "built fault-tolerant architectures with graceful degradation",
        "optimized database queries for high-throughput workloads"
    ],
    "general": [
        "led technical design reviews and architecture decisions",
        "implemented monitoring and alerting for production systems",
        "built CI/CD pipelines for automated deployment",
        "contributed to open-source projects in the ecosystem"
    ]
}


def get_technical_depth_signals(role: str, jd_keywords: List[str]) -> List[str]:
    """Get role-appropriate technical depth signals."""
    role_lower = role.lower()
    jd_lower = " ".join(jd_keywords).lower()
    
    if "ml" in role_lower or "machine learning" in jd_lower:
        return TECHNICAL_DEPTH_SIGNALS["ml"]
    elif "data engineer" in role_lower or "pipeline" in jd_lower:
        return TECHNICAL_DEPTH_SIGNALS["data"]
    elif "backend" in role_lower or "distributed" in jd_lower:
        return TECHNICAL_DEPTH_SIGNALS["systems"]
    else:
        return TECHNICAL_DEPTH_SIGNALS["general"]


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
    """Remove all academic mentions - GPA, graduation, coursework, degree dates."""
    if not text:
        return ""
    
    # Remove GPA mentions
    text = re.sub(r"\bGPA\b[:\s]*\d+(\.\d+)?(/\d+(\.\d+)?)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+(\.\d+)?\s*(GPA|CGPA)\b", "", text, flags=re.IGNORECASE)
    
    # Remove graduation year/date mentions
    text = re.sub(r"\b(graduat(ed?|ing|ion))\s*(in|from|date)?\s*\d{4}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(class of|expected|graduating)\s*\d{4}\b", "", text, flags=re.IGNORECASE)
    
    # Remove coursework mentions
    text = re.sub(r"\b(relevant\s+)?coursework\b[:\s]*[^.]*\.", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcourses?\s+(include|including|such as)[^.]*\.", "", text, flags=re.IGNORECASE)
    
    # Remove degree mentions with dates
    text = re.sub(r"\b(bachelor'?s?|master'?s?|ph\.?d\.?|b\.?s\.?|m\.?s\.?)\s*(degree)?\s*(in\s+\w+)?\s*,?\s*\d{4}", "", text, flags=re.IGNORECASE)
    
    # Remove university + year combinations
    text = re.sub(r"\buniversity[^,]*,?\s*\d{4}", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcollege[^,]*,?\s*\d{4}", "", text, flags=re.IGNORECASE)
    
    # Clean up double spaces and punctuation
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


def _postprocess_body(text: str) -> str:
    text = secure_tex_input(text or "")
    text = _strip_academic_content(text)  # NEW: Remove academic content
    text = _strip_star_labels(text)
    text = _normalize_body_whitespace(text)
    text = _debullettify_and_dedash(text)
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
# 🧠 ADVANCED COMPANY INTELLIGENCE EXTRACTION
# ============================================================

async def extract_deep_company_intel(
    jd_text: str,
    company: str,
    role: str
) -> Dict[str, Any]:
    """Extract deep company intelligence from JD + known database."""
    
    base_intel = get_company_intelligence(company)
    
    prompt = f"""Analyze this job description and extract SPECIFIC details.

COMPANY: {company}
ROLE: {role}

JOB DESCRIPTION:
{jd_text[:4000]}

Return STRICT JSON:
{{
    "team_name": "specific team/org name if mentioned",
    "team_mission": "what this specific team does",
    "tech_stack": ["specific technologies mentioned"],
    "key_projects": ["specific projects/products mentioned"],
    "business_impact": "what business problem this role solves",
    "unique_challenges": ["specific technical challenges mentioned"],
    "required_expertise": ["must-have skills"],
    "insider_terminology": ["company-specific terms used"]
}}

Extract REAL details from the JD, not generic statements.
"""

    try:
        jd_intel = await chat_json(prompt, model=_INTELLIGENCE_MODEL)
    except Exception as e:
        log_event("intel_extraction_fail", {"error": str(e)})
        jd_intel = {}
    
    return {
        **base_intel,
        "team_name": jd_intel.get("team_name", ""),
        "team_mission": jd_intel.get("team_mission", ""),
        "jd_tech_stack": jd_intel.get("tech_stack", []),
        "key_projects": jd_intel.get("key_projects", []),
        "business_impact": jd_intel.get("business_impact", ""),
        "unique_challenges": jd_intel.get("unique_challenges", []),
        "required_expertise": jd_intel.get("required_expertise", []),
        "insider_terminology": jd_intel.get("insider_terminology", []) + base_intel.get("insider_phrases", [])
    }


async def extract_resume_highlights(resume_text: str) -> Dict[str, Any]:
    """Extract key highlights from resume (NO academic content)."""
    
    if not (resume_text or "").strip():
        return {"highlights": [], "skills": [], "experiences": [], "achievements": []}
    
    prompt = f"""Extract the MOST IMPRESSIVE PROFESSIONAL highlights from this resume.

RESUME:
{resume_text[:5000]}

Return STRICT JSON:
{{
    "top_achievements": ["3-5 most impressive WORK achievements with context"],
    "technical_skills": ["key technical skills demonstrated in projects"],
    "leadership_signals": ["leadership/ownership examples from work"],
    "quantified_results": ["results with numbers/metrics from work"],
    "company_names": ["companies worked at"],
    "project_highlights": ["key projects with outcomes"]
}}

IMPORTANT:
- Focus ONLY on work experience and projects
- DO NOT include GPA, graduation dates, coursework, or academic achievements
- DO NOT mention university names unless it's about research WORK
- Focus on PROFESSIONAL, SPECIFIC achievements only
"""

    try:
        return await chat_json(prompt, model=_INTELLIGENCE_MODEL)
    except Exception:
        return {"highlights": [], "skills": [], "experiences": [], "achievements": []}


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
        role = (data.get("role") or "Role").strip()
        return company, role
    except Exception as e:
        log_event("coverletter_extract_fail", {"error": str(e)})
        return "Company", "Role"


# ============================================================
# 📝 KILLER COVER LETTER DRAFTING
# ============================================================

_LENGTH_BANDS = {"short": (120, 180), "standard": (200, 300), "long": (320, 420)}

_BUZZ_BANNED = [
    "passionate", "dynamic", "cutting edge", "team player", "synergy",
    "results-driven", "fast-paced", "leverage synergies", "mission inspires me",
    "innovative work", "perfect fit", "dream job", "always wanted to",
    "since childhood", "grateful for any opportunity", "humbly request",
    "excited to apply", "thrilled", "honored", "privileged"
]

_ACADEMIC_BANNED = [
    "gpa", "cgpa", "graduated", "graduating", "graduation", "coursework",
    "courses", "degree", "bachelor", "master", "phd", "university studies",
    "academic", "transcript", "cum laude", "dean's list", "honors"
]

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\./_+]*")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = set("a an the and or but if while for with to of in on by from as at into over under is are was were be been being this that these those i you he she they we it".split())


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(text or "")]


def _extract_terms(text: str) -> Set[str]:
    terms = set()
    for tok in set(_WORD.findall(text or "")):
        raw = tok.strip()
        if not raw or raw.lower() in _STOPWORDS or len(raw) < 2:
            continue
        terms.add(raw)
    return terms


def _clean_text_local(s: str, banned_phrases: Optional[List[str]] = None) -> str:
    txt = (s or "").replace("&", " and ").replace("—", ", ").replace("–", ", ").strip()
    txt = re.sub(r"^\s*(?:[#`>\-\*•]|\d+[.)])\s+", "", txt, flags=re.MULTILINE)
    
    # Combine all banned phrases
    banned = set((banned_phrases or []) + _BUZZ_BANNED + _ACADEMIC_BANNED)
    for b in sorted(banned, key=len, reverse=True):
        txt = re.sub(rf"\b{re.escape(b)}\b", "", txt, flags=re.IGNORECASE)
    
    # Remove academic content patterns
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
    n = len(sents)
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
    """
    Generate a KILLER cover letter with:
    - ASTONISHING company-specific opening
    - NO academic mentions
    - Deep product/technical knowledge
    - Insider terminology
    - Compelling narrative
    """
    
    tone = (tone or "balanced").strip().lower()
    length = (length or "standard").strip().lower()
    if length not in _LENGTH_BANDS:
        length = "standard"
    
    has_resume = bool((resume_text or "").strip())
    
    # Get KILLER opening hook
    killer_hook = get_killer_hook(company, "product_insights")
    insider_hook = get_killer_hook(company, "insider_knowledge")
    recent_hook = get_killer_hook(company, "recent_moves")
    
    # Extract JD keywords
    jd_terms = list(_extract_terms(jd_text or ""))[:30]
    
    # Get company-specific elements
    culture_keywords = company_intel.get("culture_keywords", [])
    tech_focus = company_intel.get("tech_focus", [])
    insider_terms = company_intel.get("insider_terminology", [])
    products = company_intel.get("products_to_reference", [])
    unique_challenges = company_intel.get("unique_challenges", [])
    business_impact = company_intel.get("business_impact", "")
    team_name = company_intel.get("team_name", "")
    
    # Get resume highlights (professional only)
    top_achievements = resume_highlights.get("top_achievements", [])
    technical_skills = resume_highlights.get("technical_skills", [])
    quantified_results = resume_highlights.get("quantified_results", [])
    
    # Get role-specific value props
    value_props = get_value_propositions(role)
    
    # Get technical depth signals
    tech_depth = get_technical_depth_signals(role, jd_terms)
    
    length_hint = {
        "short": "Target 150-180 words in 2 tight paragraphs.",
        "standard": "Target 220-280 words in 3 paragraphs.",
        "long": "Target 350-400 words in 3-4 paragraphs.",
    }[length]
    
    # Build the KILLER prompt
    sys_prompt = f"""You are writing a KILLER cover letter that will make the recruiter at {company} stop and think "This person REALLY knows us."

YOUR MISSION: Write an opening so compelling that the recruiter reads the entire letter. Make them think "We NEED to interview this person."

🎯 KILLER OPENING HOOKS (use ONE of these as inspiration for your opening):
{killer_hook if killer_hook else "Create a specific, impressive hook about " + company}

💡 INSIDER KNOWLEDGE (weave naturally):
{insider_hook if insider_hook else "Show you understand " + company + "'s unique challenges"}

📈 RECENT COMPANY MOVES:
{recent_hook if recent_hook else "Reference " + company + "'s recent strategic direction"}

🏢 COMPANY INTELLIGENCE:
- Culture values: {', '.join(culture_keywords[:3]) if culture_keywords else 'excellence'}
- Tech focus: {', '.join(tech_focus[:3]) if tech_focus else 'modern technology'}
- Products to reference: {', '.join(products[:3]) if products else company + ' products'}
- Insider terms: {', '.join(insider_terms[:4]) if insider_terms else 'none'}
- Challenges: {', '.join(unique_challenges[:2]) if unique_challenges else 'scaling'}
- Business impact: {business_impact or 'driving key initiatives'}
{f'- Team: {team_name}' if team_name else ''}

💪 CANDIDATE STRENGTHS (PROFESSIONAL ONLY - NO ACADEMICS):
- Work achievements: {'; '.join(top_achievements[:3]) if top_achievements else 'strong technical background'}
- Technical skills: {', '.join(technical_skills[:6]) if technical_skills else 'relevant skills'}
- Quantified results: {'; '.join(quantified_results[:2]) if quantified_results else 'measurable impact'}

REQUIRED STRUCTURE:

PARAGRAPH 1 - THE ASTONISHING HOOK (3-4 sentences):
- Start with a SPECIFIC, IMPRESSIVE statement about {company} that shows deep knowledge
- Reference a specific product, feature, technical decision, or company initiative
- Show you understand THEIR unique challenges, not generic industry challenges
- Make the recruiter think "How does this person know so much about us?"

PARAGRAPH 2 - PROOF OF VALUE (4-5 sentences):
- Lead with your STRONGEST work achievement
- Connect your experience DIRECTLY to their needs
- Include specific technical accomplishments from WORK (not school)
- Show HOW you solved problems, not just what you did
- Demonstrate ownership and measurable impact

PARAGRAPH 3 - FORWARD VALUE + CLOSE (3-4 sentences):
- Preview specific contributions in first 30-60-90 days
- Reference their specific challenges/goals
- Show cultural fit through actions, not words
- End with confident ask for conversation

⛔ ABSOLUTE RULES - NEVER VIOLATE:
1. NO ACADEMIC CONTENT: No GPA, graduation year, coursework, degree dates, university names (unless for research work)
2. NO CLICHÉS: "passionate", "excited", "team player", "fast-paced", "perfect fit", "dream job"
3. NO BEGGING: "grateful for opportunity", "hope you'll consider", "humbly", "honored"
4. NO GENERIC STATEMENTS: Everything must be specific to {company}
5. PROFESSIONAL EXPERIENCE ONLY: All proof must come from work/projects, not academics
6. {length_hint}
7. First-person singular, confident but not arrogant
8. Use "and" not "&", no em-dashes
9. Every sentence must provide value - no filler

OUTPUT: Just the body paragraphs, no salutation or signature.
"""

    user_prompt = f"""Write the KILLER cover letter body for:

ROLE: {role}
COMPANY: {company}

JOB DESCRIPTION:
{jd_text[:4000]}

RESUME (for PROFESSIONAL experience reference only):
{resume_text[:4000] if has_resume else 'Focus on JD requirements and general professional capability.'}

Remember: 
- The opening must ASTONISH the recruiter with your knowledge of {company}
- NO academic content (GPA, graduation, coursework)
- Only reference WORK experience and projects
"""

    # Generate the draft
    draft = await chat_text(sys_prompt, user_prompt, model=_DRAFT_MODEL)
    
    # Clean and validate
    body = _clean_text_local(draft)
    
    # Validate and repair
    body = await _validate_and_repair_killer(
        body, company, role, jd_text, resume_text,
        company_intel, resume_highlights, length, tone
    )
    
    # Final shaping
    body = _shape_paragraphs(body, length)
    body = _enforce_word_band_local(body, length)
    
    return _postprocess_body(body)


async def _validate_and_repair_killer(
    body: str,
    company: str,
    role: str,
    jd_text: str,
    resume_text: str,
    company_intel: Dict[str, Any],
    resume_highlights: Dict[str, Any],
    length: str,
    tone: str,
    max_repairs: int = 2
) -> str:
    """Validate and repair the cover letter for quality."""
    
    issues = []
    body_lower = body.lower()
    
    # Check for banned clichés
    for cliche in _BUZZ_BANNED:
        if cliche.lower() in body_lower:
            issues.append(f"Contains banned cliché: '{cliche}'")
    
    # Check for academic content
    for academic in _ACADEMIC_BANNED:
        if academic.lower() in body_lower:
            issues.append(f"Contains academic content: '{academic}'")
    
    # Check company name is mentioned
    if company.lower() not in body_lower:
        issues.append("Company name not mentioned")
    
    # Check for specific product/feature reference
    products = company_intel.get("products_to_reference", [])
    product_mentioned = any(p.lower() in body_lower for p in products if p)
    if products and not product_mentioned:
        issues.append(f"No specific product reference (mention: {', '.join(products[:3])})")
    
    # Check for technical specificity
    tech_terms = company_intel.get("jd_tech_stack", []) + company_intel.get("tech_focus", [])
    tech_mentioned = sum(1 for t in tech_terms if t.lower() in body_lower)
    if tech_mentioned < 2:
        issues.append("Insufficient technical specificity")
    
    # Check opening is company-specific (not generic)
    first_sentence = body.split('.')[0] if body else ""
    if company.lower() not in first_sentence.lower():
        issues.append("Opening sentence doesn't mention company - needs company-specific hook")
    
    # Check for forward-looking content
    if not re.search(r"\b(first|initial|early|would|will|contribute|drive|bring)\b", body_lower):
        issues.append("Missing forward-looking value statement")
    
    # Repair if needed
    if issues and max_repairs > 0:
        repair_prompt = f"""Rewrite this cover letter to fix these issues:
{chr(10).join(f'- {i}' for i in issues)}

Current draft:
{body}

CRITICAL REQUIREMENTS:
- Company: {company}
- Role: {role}
- Opening sentence MUST mention {company} with a specific insight
- Mention specific products/features: {', '.join(products[:3]) if products else company + ' products'}
- Include specific technologies: {', '.join(tech_terms[:5])}
- NO academic content (GPA, graduation, coursework, university)
- NO clichés (passionate, excited, dream job, etc.)
- Add forward-looking contribution preview
- Keep it {length} length

Return only the improved body paragraphs.
"""
        try:
            repaired = await chat_text(
                "You are fixing a cover letter to make it more compelling and specific.",
                repair_prompt,
                model=_DRAFT_MODEL
            )
            return await _validate_and_repair_killer(
                _clean_text_local(repaired),
                company, role, jd_text, resume_text,
                company_intel, resume_highlights, length, tone,
                max_repairs - 1
            )
        except Exception:
            pass
    
    return body


# ============================================================
# ✨ Humanize via internal service
# ============================================================

async def humanize_text(body_text: str, tone: str) -> str:
    api_base = (getattr(config, "API_BASE_URL", "") or "").rstrip("/") or "http://127.0.0.1:8000"
    url = f"{api_base}/api/superhuman/rewrite"
    payload = {"text": body_text, "mode": "coverletter", "tone": tone, "latex_safe": True}
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        result = data.get("rewritten") or data.get("text") or body_text
        # Strip academic content after humanization too
        return _strip_academic_content(result)
    except Exception as e:
        log_event("superhuman_handoff_fail", {"error": str(e)})
        return body_text


# ============================================================
# 📄 Header & Template Injection
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
) -> str:
    def esc(v: str) -> str:
        return _latex_escape_light(secure_tex_input(v or ""))

    subst = {
        "COMPANY": company, "ROLE": role, "CANDIDATE_NAME": candidate,
        "NAME": candidate, "DATE": date_str, "EMAIL": email,
        "PHONE": phone, "CITYSTATE": citystate,
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
            lambda m: f"\n{safe_body}\n\\end{{document}}\n",
            base_tex, flags=re.I,
        )

    return base_tex.rstrip() + f"\n\n{safe_body}\n\\end{{document}}\n"


# ============================================================
# 🚀 MAIN ENDPOINT - KILLER VERSION
# ============================================================

@router.post("")
async def generate_coverletter(
    jd_text: str = Form(...),
    resume_tex: str = Form(""),
    use_humanize: bool = Form(True),
    tone: str = Form("balanced"),
    length: str = Form("standard"),
):
    """
    Generate a KILLER cover letter that astonishes recruiters.
    
    Features:
    - ASTONISHING company-specific opening hooks
    - NO academic content (GPA, graduation, coursework)
    - Deep product/technical knowledge signals
    - Insider terminology that impresses
    - Company-specific insights in first sentence
    - Professional experience only
    """
    
    if not (config.OPENAI_API_KEY or "").strip():
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY missing.")
    if not (jd_text or "").strip():
        raise HTTPException(status_code=400, detail="jd_text is required.")

    # Step 1: Extract company and role
    company, role = await extract_company_role(jd_text)
    log_event("coverletter_start", {"company": company, "role": role})

    # Step 2: Extract deep company intelligence
    company_intel = await extract_deep_company_intel(jd_text, company, role)
    log_event("company_intel_extracted", {
        "company": company,
        "killer_hooks_available": bool(get_killer_hook(company)),
        "insider_terms_count": len(company_intel.get("insider_terminology", []))
    })

    # Step 3: Extract resume highlights (professional only)
    resume_highlights = await extract_resume_highlights(resume_tex)

    # Step 4: Generate KILLER cover letter body
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

    # Step 5: Humanize if requested
    if use_humanize:
        body_text = await humanize_text(body_text, tone)
        body_text = _postprocess_body(body_text)

    # Step 6: Inject into LaTeX template
    base_path = config.BASE_COVERLETTER_PATH
    try:
        with open(base_path, encoding="utf-8") as f:
            base_tex = f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template not found: {base_path}")

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
        log_event("inject_error", {"error": str(e)})
        injected = f"{base_tex}\n\n{body_text}\n"
        if not injected.strip().endswith("\\end{document}"):
            injected += "\n\\end{document}\n"

    final_tex = render_final_tex(injected)
    pdf_bytes = compile_latex_safely(final_tex) or b""
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    # Save files
    company_slug = safe_filename(company)
    role_slug = safe_filename(role)
    context_key = f"{company_slug}__{role_slug}"

    out_pdf_path = config.get_sample_coverletter_pdf_path(company, role)
    ensure_dir(out_pdf_path.parent)
    if pdf_bytes:
        out_pdf_path.write_bytes(pdf_bytes)

    # Save context
    ctx_dir = config.get_contexts_dir()
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
        "key": context_key,
        "company": company,
        "role": role,
        "jd_text": jd_text,
        "company_intel": {
            "culture_keywords": company_intel.get("culture_keywords", []),
            "tech_focus": company_intel.get("tech_focus", []),
            "insider_terms": company_intel.get("insider_terminology", [])[:5],
            "killer_hook_used": bool(get_killer_hook(company)),
        },
        "cover_letter": {
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

    log_event("coverletter_generated", {
        "company": company,
        "role": role,
        "tone": tone,
        "length": length,
        "humanized": use_humanize,
        "killer_hook_used": bool(get_killer_hook(company)),
        "chars": len(body_text),
    })

    return JSONResponse({
        "company": company,
        "role": role,
        "tone": tone,
        "use_humanize": use_humanize,
        "tex_string": final_tex,
        "pdf_base64": pdf_b64,
        "pdf_path": str(out_pdf_path),
        "context_key": context_key,
        "context_path": str(ctx_path),
        "company_intel_used": {
            "culture_keywords": company_intel.get("culture_keywords", [])[:3],
            "tech_focus": company_intel.get("tech_focus", [])[:3],
            "insider_terms": company_intel.get("insider_terminology", [])[:3],
            "killer_hook_available": bool(get_killer_hook(company)),
        },
        "id": context_key,
        "memory_id": context_key,
    })