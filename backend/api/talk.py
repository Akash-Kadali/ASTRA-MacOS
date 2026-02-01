"""
============================================================
 HIREX v4.0.0 — talk.py (ULTIMATE KILLER ANSWERS)
 ------------------------------------------------------------
 MAJOR UPGRADES:
 - Anti-hallucination grounding with resume fact verification
 - Answer quality scoring (specificity, relevance, hook strength)
 - Skill gap analysis with honest addressing strategies
 - Interview stage awareness (recruiter vs technical vs final)
 - Follow-up question prediction
 - Red flag detection and proactive addressing
 - Personal brand/theme consistency
 - Multi-answer session consistency
 - Company-specific question bank
 - Behavioral STAR enforcement
 - Salary/negotiation intelligence
 - Output quality validation before returning

 Author: Sri Akash Kadali
============================================================
"""

from __future__ import annotations

import json
import re
import time
import random
import hashlib
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

SUMMARIZER_MODEL = getattr(config, "TALK_SUMMARY_MODEL", "gpt-4o-mini")
ANSWER_MODEL = getattr(config, "TALK_ANSWER_MODEL", "gpt-4o-mini")
CHAT_SAFE_DEFAULT = getattr(config, "DEFAULT_MODEL", "gpt-4o-mini")

# Session-level fact store for consistency across multiple questions
_SESSION_FACTS: Dict[str, Dict[str, Any]] = {}
_SESSION_ANSWERS: Dict[str, List[Dict[str, Any]]] = {}


# ============================================================
# 🏢 COMPREHENSIVE COMPANY INTELLIGENCE DATABASE
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
            "Reference experimentation and iteration"
        ],
        "insider_terms": ["member experience", "title discovery", "personalization at scale"],
        "avoid_saying": ["I follow instructions well", "I'm a team player", "I'm passionate"],
        "common_questions": [
            "Tell me about a time you made a data-driven decision",
            "Describe a situation where you took ownership",
            "How do you handle ambiguity?",
            "Tell me about a time you disagreed with your manager"
        ],
        "interview_stages": {
            "recruiter": "Focus on culture fit and high-level experience",
            "technical": "Deep dive into systems design and ML expertise",
            "hiring_manager": "Ownership examples and impact stories",
            "final": "Leadership alignment and long-term vision"
        },
        "salary_range": {"entry": "$150k-$200k", "mid": "$200k-$350k", "senior": "$350k-$600k+"},
        "competitors": ["Disney+", "HBO Max", "Amazon Prime Video", "Apple TV+"],
        "why_not_competitors": "Netflix's experimentation culture and engineering autonomy is unmatched"
    },
    "google": {
        "what_they_value": ["technical excellence", "scalability thinking", "user impact", "10x improvements"],
        "culture_keywords": ["Googleyness", "think big", "user first", "psychological safety"],
        "tech_strengths": ["Distributed Systems", "AI/ML Infrastructure", "Search Quality", "Cloud Platform"],
        "what_impresses_them": [
            "Structured problem-solving approach",
            "Scalability and efficiency thinking",
            "Concrete technical depth",
            "User-centric impact framing"
        ],
        "insider_terms": ["OKRs", "launch and iterate", "10x improvement", "Noogler"],
        "avoid_saying": ["I work hard", "I'm detail-oriented", "I'm a fast learner"],
        "common_questions": [
            "Tell me about a technically challenging problem you solved",
            "How would you design X system?",
            "Tell me about a time you improved something by 10x",
            "How do you prioritize when everything is important?"
        ],
        "interview_stages": {
            "recruiter": "Googleyness assessment and experience overview",
            "technical": "Coding, system design, and ML depth",
            "hiring_manager": "Leadership and collaboration examples",
            "final": "Team match and culture fit"
        },
        "salary_range": {"entry": "$140k-$180k", "mid": "$200k-$400k", "senior": "$400k-$800k+"},
        "competitors": ["Microsoft", "Meta", "Amazon", "Apple"],
        "why_not_competitors": "Google's technical challenges and AI leadership are unparalleled"
    },
    "meta": {
        "what_they_value": ["move fast", "impact metrics", "product sense", "bold thinking"],
        "culture_keywords": ["Move Fast", "Be Bold", "Focus on Impact", "Build Social Value"],
        "tech_strengths": ["Ranking Systems", "Social Graph", "AR/VR", "Ads Optimization"],
        "what_impresses_them": [
            "Quantified impact with specific metrics",
            "Move fast mentality with examples",
            "Product intuition demonstrated",
            "Scale of systems you've worked on"
        ],
        "insider_terms": ["family of apps", "integrity systems", "social impact", "Bootcamp"],
        "avoid_saying": ["I'm careful and methodical", "I double-check everything", "I prefer stability"],
        "common_questions": [
            "Tell me about your biggest impact",
            "How do you make decisions with incomplete data?",
            "Describe a time you moved fast and broke things",
            "How do you measure success?"
        ],
        "interview_stages": {
            "recruiter": "Impact stories and culture fit",
            "technical": "Coding and system design at scale",
            "hiring_manager": "Product sense and leadership",
            "final": "Executive alignment"
        },
        "salary_range": {"entry": "$140k-$180k", "mid": "$200k-$400k", "senior": "$400k-$700k+"},
        "competitors": ["Google", "TikTok", "Snap", "Twitter/X"],
        "why_not_competitors": "Meta's scale of impact and bold technical bets are unique"
    },
    "amazon": {
        "what_they_value": ["customer obsession", "ownership", "bias for action", "dive deep", "deliver results"],
        "culture_keywords": ["Customer Obsession", "Ownership", "Bias for Action", "Dive Deep", "Leadership Principles"],
        "tech_strengths": ["AWS Services", "Supply Chain ML", "Retail Optimization", "Logistics"],
        "what_impresses_them": [
            "STAR format with specific metrics",
            "Leadership principles alignment",
            "Customer impact examples",
            "Ownership and accountability stories"
        ],
        "insider_terms": ["PR/FAQ", "6-pager", "bar raiser", "Day 1 mentality", "mechanisms", "working backwards"],
        "avoid_saying": ["That's not my job", "I waited for direction", "I delegated"],
        "common_questions": [
            "Tell me about a time you went above and beyond for a customer",
            "Describe a situation where you had to dive deep",
            "Tell me about a time you disagreed and committed",
            "How do you handle competing priorities?"
        ],
        "interview_stages": {
            "recruiter": "LP screening and experience fit",
            "technical": "Coding and system design",
            "loop": "Multiple LP-focused behavioral rounds",
            "bar_raiser": "Cross-team culture assessment"
        },
        "salary_range": {"entry": "$130k-$170k", "mid": "$180k-$350k", "senior": "$350k-$600k+"},
        "competitors": ["Google Cloud", "Microsoft Azure", "Walmart", "Shopify"],
        "why_not_competitors": "Amazon's customer obsession and ownership culture drive real impact"
    },
    "microsoft": {
        "what_they_value": ["growth mindset", "customer empathy", "collaboration", "inclusive culture"],
        "culture_keywords": ["Growth Mindset", "Customer Obsessed", "One Microsoft", "Learn-it-all"],
        "tech_strengths": ["Azure Cloud", "AI/Copilot", "Microsoft 365", "Developer Tools"],
        "what_impresses_them": [
            "Growth mindset examples with learning",
            "Collaboration across teams",
            "Customer impact stories",
            "Responsible AI awareness"
        ],
        "insider_terms": ["growth mindset", "customer zero", "inclusive design", "One Microsoft"],
        "avoid_saying": ["I already know everything", "I work best alone", "I avoid ambiguity"],
        "common_questions": [
            "Tell me about a time you learned from failure",
            "How do you collaborate with difficult stakeholders?",
            "Describe your approach to inclusive design",
            "How do you balance innovation with responsibility?"
        ],
        "interview_stages": {
            "recruiter": "Growth mindset and culture fit",
            "technical": "Coding and system design",
            "hiring_manager": "Collaboration and customer focus",
            "final": "As-appropriate leadership"
        },
        "salary_range": {"entry": "$130k-$170k", "mid": "$180k-$350k", "senior": "$350k-$600k+"},
        "competitors": ["Google", "Amazon", "Salesforce", "Oracle"],
        "why_not_competitors": "Microsoft's growth mindset culture and AI leadership momentum"
    },
    "apple": {
        "what_they_value": ["attention to detail", "user privacy", "craftsmanship", "simplicity"],
        "culture_keywords": ["Think Different", "Simplicity", "Privacy as Human Right", "Excellence"],
        "tech_strengths": ["On-Device ML", "Privacy-Preserving AI", "Hardware-Software Integration"],
        "what_impresses_them": [
            "Design thinking and user empathy",
            "Privacy-first approach",
            "Quality over speed examples",
            "Attention to detail stories"
        ],
        "insider_terms": ["DRI", "surprise and delight", "it just works", "top 100"],
        "avoid_saying": ["Good enough is fine", "Users don't care about details", "Move fast and break things"],
        "common_questions": [
            "Tell me about a time you obsessed over details",
            "How do you balance user experience with technical constraints?",
            "Describe your approach to privacy",
            "Tell me about something you're proud of building"
        ],
        "interview_stages": {
            "recruiter": "Culture fit and experience",
            "technical": "Deep technical expertise",
            "design": "User empathy and design thinking",
            "final": "Leadership alignment"
        },
        "salary_range": {"entry": "$140k-$180k", "mid": "$200k-$400k", "senior": "$400k-$700k+"},
        "competitors": ["Google", "Samsung", "Microsoft"],
        "why_not_competitors": "Apple's commitment to privacy and craftsmanship is unmatched"
    },
    "stripe": {
        "what_they_value": ["rigorous thinking", "users first", "writing quality", "long-term orientation"],
        "culture_keywords": ["Users First", "Move with Urgency", "Think Rigorously", "Trust and Amplify"],
        "tech_strengths": ["Payment Infrastructure", "Financial APIs", "Fraud Detection", "Developer Experience"],
        "what_impresses_them": [
            "Clear, rigorous thinking",
            "Developer empathy examples",
            "Long-term technical decisions",
            "Writing and communication quality"
        ],
        "insider_terms": ["increase GDP of internet", "payment rails", "developer love"],
        "avoid_saying": ["I prefer quick wins", "Documentation is boring", "I focus on short-term"],
        "common_questions": [
            "Walk me through a complex technical decision",
            "How do you think about API design?",
            "Tell me about a time you prioritized long-term over short-term",
            "How do you communicate technical concepts?"
        ],
        "interview_stages": {
            "recruiter": "Writing sample and culture fit",
            "technical": "System design and coding",
            "work_sample": "Take-home or pair programming",
            "final": "Team and leadership fit"
        },
        "salary_range": {"entry": "$150k-$200k", "mid": "$220k-$400k", "senior": "$400k-$650k+"},
        "competitors": ["Square", "Adyen", "Braintree", "Plaid"],
        "why_not_competitors": "Stripe's developer-first culture and rigorous thinking"
    },
    "airbnb": {
        "what_they_value": ["mission alignment", "customer empathy", "design-driven", "belonging"],
        "culture_keywords": ["Belong Anywhere", "Champion the Mission", "Be a Host", "Embrace Adventure"],
        "tech_strengths": ["Search & Ranking", "Pricing Algorithms", "Trust & Safety", "Payments"],
        "what_impresses_them": [
            "Mission and purpose alignment",
            "Customer/user empathy stories",
            "Creative problem-solving",
            "Design-thinking approach"
        ],
        "insider_terms": ["belonging", "host community", "guest journey", "Airbnb it"],
        "avoid_saying": ["I'm purely technical", "I don't care about mission", "Users are just data points"],
        "common_questions": [
            "Why does our mission resonate with you?",
            "Tell me about a time you advocated for the user",
            "How do you balance host and guest needs?",
            "Describe a creative solution you developed"
        ],
        "interview_stages": {
            "recruiter": "Mission alignment and culture",
            "technical": "System design and coding",
            "cross-functional": "Design and product collaboration",
            "final": "Leadership and values"
        },
        "salary_range": {"entry": "$140k-$180k", "mid": "$200k-$380k", "senior": "$380k-$600k+"},
        "competitors": ["Booking.com", "VRBO", "Hotels.com"],
        "why_not_competitors": "Airbnb's mission of belonging and design-driven culture"
    },
    "uber": {
        "what_they_value": ["systems thinking", "marketplace understanding", "reliability at scale", "data-driven"],
        "culture_keywords": ["Build Globally", "Act Like Owners", "Persevere", "Celebrate Differences"],
        "tech_strengths": ["Marketplace Optimization", "ETA Prediction", "Route Optimization", "Fraud Detection"],
        "what_impresses_them": [
            "Systems and marketplace thinking",
            "Reliability and scale examples",
            "Data-driven decision making",
            "Impact metrics at scale"
        ],
        "insider_terms": ["marketplace balance", "rider experience", "driver earnings", "surge"],
        "avoid_saying": ["I prefer simple problems", "Scale doesn't matter to me", "I avoid complexity"],
        "common_questions": [
            "How would you optimize a two-sided marketplace?",
            "Tell me about a system you built for reliability",
            "How do you make decisions with conflicting metrics?",
            "Describe a time you solved a complex systems problem"
        ],
        "interview_stages": {
            "recruiter": "Experience and culture fit",
            "technical": "System design and marketplace thinking",
            "coding": "Algorithmic problem-solving",
            "final": "Leadership and team fit"
        },
        "salary_range": {"entry": "$140k-$180k", "mid": "$200k-$380k", "senior": "$380k-$550k+"},
        "competitors": ["Lyft", "DoorDash", "Instacart"],
        "why_not_competitors": "Uber's global scale and marketplace complexity"
    },
    "databricks": {
        "what_they_value": ["technical excellence", "open source contribution", "customer impact", "data passion"],
        "culture_keywords": ["Customer Obsessed", "Unity", "Ownership", "Open Source First"],
        "tech_strengths": ["Lakehouse", "Delta Lake", "MLflow", "Spark", "Data Engineering"],
        "what_impresses_them": [
            "Deep technical expertise",
            "Open source appreciation/contribution",
            "Data architecture experience",
            "Customer-facing technical work"
        ],
        "insider_terms": ["Lakehouse", "Delta Lake", "data + AI", "open source", "Unity Catalog"],
        "avoid_saying": ["I prefer proprietary tools", "Open source is risky", "I avoid customer interaction"],
        "common_questions": [
            "How would you design a data lakehouse?",
            "Tell me about your open source contributions",
            "How do you approach data quality at scale?",
            "Describe a complex data architecture you built"
        ],
        "interview_stages": {
            "recruiter": "Technical background and culture",
            "technical": "Data architecture and Spark expertise",
            "system_design": "Lakehouse and ML infrastructure",
            "final": "Customer focus and team fit"
        },
        "salary_range": {"entry": "$150k-$200k", "mid": "$220k-$400k", "senior": "$400k-$600k+"},
        "competitors": ["Snowflake", "AWS", "Google BigQuery"],
        "why_not_competitors": "Databricks' open source DNA and unified data + AI platform"
    },
    "snowflake": {
        "what_they_value": ["engineering excellence", "customer focus", "big thinking", "integrity"],
        "culture_keywords": ["Put Customers First", "Integrity Always", "Think Big", "Be Excellent"],
        "tech_strengths": ["Data Cloud", "Data Sharing", "Snowpark", "Data Marketplace"],
        "what_impresses_them": [
            "Technical excellence examples",
            "Customer success stories",
            "Big thinking and ambition",
            "Performance optimization experience"
        ],
        "insider_terms": ["Data Cloud", "Snowpark", "data sharing economy", "zero-copy cloning"],
        "avoid_saying": ["Small improvements are fine", "Customers are annoying", "Good enough works"],
        "common_questions": [
            "How would you optimize query performance at scale?",
            "Tell me about a time you helped a customer succeed",
            "How do you approach data governance?",
            "Describe your most ambitious technical project"
        ],
        "interview_stages": {
            "recruiter": "Experience and culture fit",
            "technical": "Database internals and performance",
            "system_design": "Data architecture at scale",
            "final": "Customer focus and values"
        },
        "salary_range": {"entry": "$150k-$200k", "mid": "$220k-$400k", "senior": "$400k-$600k+"},
        "competitors": ["Databricks", "AWS Redshift", "Google BigQuery"],
        "why_not_competitors": "Snowflake's engineering excellence and data sharing vision"
    },
    "linkedin": {
        "what_they_value": ["member value", "relationships", "data-driven", "inclusive culture"],
        "culture_keywords": ["Members First", "Relationships Matter", "Be Open Honest Constructive", "Act Like Owner"],
        "tech_strengths": ["Feed Ranking", "Job Matching", "Graph Systems", "Economic Graph"],
        "what_impresses_them": [
            "Member/user empathy examples",
            "Data-driven decision making",
            "Collaboration stories",
            "Scale and impact metrics"
        ],
        "insider_terms": ["economic graph", "member value", "professional identity", "InDay"],
        "avoid_saying": ["Users are just metrics", "I work alone", "Data doesn't matter"],
        "common_questions": [
            "How would you improve job matching?",
            "Tell me about a time you improved member experience",
            "How do you balance engagement with member value?",
            "Describe a data-driven decision you made"
        ],
        "interview_stages": {
            "recruiter": "Culture fit and experience",
            "technical": "Coding and system design",
            "hiring_manager": "Member focus and collaboration",
            "final": "Values alignment"
        },
        "salary_range": {"entry": "$140k-$180k", "mid": "$200k-$380k", "senior": "$380k-$550k+"},
        "competitors": ["Indeed", "Glassdoor", "ZipRecruiter"],
        "why_not_competitors": "LinkedIn's professional graph and member-first culture"
    },
    "spotify": {
        "what_they_value": ["innovation", "collaboration", "user empathy", "autonomy"],
        "culture_keywords": ["Innovative", "Collaborative", "Sincere", "Passionate", "Playful"],
        "tech_strengths": ["Audio ML", "Personalization", "Content Delivery", "Creator Tools"],
        "what_impresses_them": [
            "Product passion and user empathy",
            "Personalization expertise",
            "Collaborative spirit",
            "Creative problem-solving"
        ],
        "insider_terms": ["Discover Weekly", "audio-first", "creator ecosystem", "squad model"],
        "avoid_saying": ["I don't listen to music", "Users don't know what they want", "I prefer hierarchy"],
        "common_questions": [
            "How would you improve music recommendations?",
            "Tell me about a time you collaborated across teams",
            "How do you balance creator and listener needs?",
            "Describe a personalization system you built"
        ],
        "interview_stages": {
            "recruiter": "Culture fit and product passion",
            "technical": "ML and system design",
            "squad": "Team collaboration assessment",
            "final": "Leadership and values"
        },
        "salary_range": {"entry": "$130k-$170k", "mid": "$180k-$350k", "senior": "$350k-$500k+"},
        "competitors": ["Apple Music", "YouTube Music", "Amazon Music"],
        "why_not_competitors": "Spotify's personalization leadership and squad autonomy"
    }
}

DEFAULT_COMPANY_INTELLIGENCE = {
    "what_they_value": ["technical excellence", "collaboration", "impact", "growth"],
    "culture_keywords": ["innovation", "teamwork", "excellence"],
    "tech_strengths": ["modern technology", "scalable systems"],
    "what_impresses_them": [
        "Specific achievements with metrics",
        "Problem-solving examples",
        "Collaboration stories",
        "Technical depth demonstration"
    ],
    "insider_terms": [],
    "avoid_saying": ["I'm a team player", "I work hard", "I'm passionate"],
    "common_questions": [],
    "interview_stages": {},
    "salary_range": {"entry": "$100k-$150k", "mid": "$150k-$250k", "senior": "$250k-$400k+"},
    "competitors": [],
    "why_not_competitors": ""
}


def get_company_intelligence(company_name: str) -> Dict[str, Any]:
    """Get comprehensive intelligence about a company."""
    company_lower = (company_name or "").lower().strip()
    
    for key, intel in COMPANY_INTELLIGENCE.items():
        if key in company_lower or company_lower in key:
            return intel
    
    for key, intel in COMPANY_INTELLIGENCE.items():
        if any(word in company_lower for word in key.split()):
            return intel
    
    return DEFAULT_COMPANY_INTELLIGENCE


# ============================================================
# 🎯 ENHANCED QUESTION TYPE DETECTION & STRATEGY
# ============================================================

QUESTION_STRATEGIES = {
    "why_hire_you": {
        "patterns": [
            r"why should we hire you",
            r"why are you the right fit",
            r"why should we choose you",
            r"what makes you stand out",
            r"why you over other candidates",
            r"what sets you apart",
            r"why are you the best candidate",
            r"what do you bring to this role",
            r"what value do you add"
        ],
        "strategy": "PROOF + UNIQUE VALUE",
        "structure": [
            "Open with your STRONGEST, most RELEVANT achievement",
            "Show UNIQUE combination of skills they can't easily find",
            "Map directly to their SPECIFIC needs from JD",
            "Close with forward-looking contribution"
        ],
        "hook_templates": [
            "I've already solved the exact problem you're hiring for — {achievement}.",
            "The intersection of {skill1} and {skill2} that you need is exactly where I've built my career.",
            "In my last role, I {achievement}, which is precisely what this position requires.",
            "What sets me apart is that I don't just {skill} — I've {specific_outcome}."
        ],
        "likely_followups": [
            "Can you give me a specific example?",
            "How would you apply that here?",
            "What would you do in your first 90 days?",
            "How do you know you can do this at our scale?"
        ],
        "trap_warnings": [
            "Don't be arrogant or put down other candidates",
            "Don't be vague - they want SPECIFIC proof",
            "Don't just list skills - show IMPACT"
        ]
    },
    "why_this_company": {
        "patterns": [
            r"why do you want to work (here|at|for)",
            r"why this company",
            r"why .+\?",
            r"what attracts you to",
            r"what interests you about",
            r"why are you interested in",
            r"what draws you to"
        ],
        "strategy": "SPECIFIC COMPANY KNOWLEDGE + ALIGNMENT",
        "structure": [
            "Open with SPECIFIC company insight (product, tech, challenge)",
            "Show genuine understanding of their unique position",
            "Connect YOUR background to THEIR specific needs",
            "Demonstrate you've researched beyond the JD"
        ],
        "hook_templates": [
            "What drew me to {company} specifically is {specific_insight}.",
            "I've been following {company}'s work on {specific_thing}, and it aligns with {your_experience}.",
            "The challenge of {company_challenge} is one I've tackled before, and I'm drawn to solving it at {company}'s scale.",
            "{company}'s approach to {specific_approach} mirrors how I've built my most successful systems."
        ],
        "likely_followups": [
            "What do you know about our culture?",
            "Why not one of our competitors?",
            "What concerns do you have about us?",
            "How did you hear about us?"
        ],
        "trap_warnings": [
            "Don't be generic - they want SPECIFIC knowledge",
            "Don't mention salary/benefits as primary reason",
            "Don't badmouth competitors"
        ]
    },
    "why_this_role": {
        "patterns": [
            r"why this (role|position|job)",
            r"what interests you about this (role|position)",
            r"why are you applying for this",
            r"what excites you about this (role|position|opportunity)",
            r"why do you want this job"
        ],
        "strategy": "ROLE-SKILL MATCH + GROWTH",
        "structure": [
            "Show you understand EXACTLY what this role does",
            "Map your experience to the specific responsibilities",
            "Demonstrate how this is a natural next step",
            "Show enthusiasm through specificity, not generic excitement"
        ],
        "hook_templates": [
            "This role sits at the intersection of {area1} and {area2}, which is exactly where I've focused my career.",
            "The {specific_responsibility} in this role is something I've been doing successfully for {time}.",
            "I've built my career around {skill}, and this role is the natural next step to {goal}.",
            "What draws me to this specific role is {specific_aspect} — I've seen firsthand how impactful this work can be."
        ],
        "likely_followups": [
            "What don't you know about this role?",
            "What would be challenging for you?",
            "How does this fit your long-term goals?",
            "What would you change about this role?"
        ],
        "trap_warnings": [
            "Don't be vague about role responsibilities",
            "Don't focus only on what you'll GET",
            "Show understanding of challenges, not just perks"
        ]
    },
    "tell_me_about_yourself": {
        "patterns": [
            r"tell me about yourself",
            r"walk me through your background",
            r"introduce yourself",
            r"give me an overview of your experience",
            r"describe your background",
            r"walk me through your resume"
        ],
        "strategy": "RELEVANT NARRATIVE + TRAJECTORY",
        "structure": [
            "Start with current role/most relevant experience (Present)",
            "Connect past experiences in a coherent narrative (Past)",
            "Show intentional career trajectory toward THIS role (Future)",
            "End with why you're here NOW"
        ],
        "hook_templates": [
            "I'm a {role_type} who has spent the last {time} building {what_you_build}.",
            "My career has been focused on {theme}, most recently at {company} where I {achievement}.",
            "I've spent my career at the intersection of {area1} and {area2}, which led me to {this_opportunity}.",
            "What defines my work is {defining_theme} — at {company}, this meant {specific_example}."
        ],
        "likely_followups": [
            "Tell me more about {specific thing you mentioned}",
            "Why did you leave your last role?",
            "What's your biggest accomplishment?",
            "What are you looking for in your next role?"
        ],
        "trap_warnings": [
            "Don't recite your entire resume",
            "Don't start from childhood",
            "Keep it under 2 minutes spoken",
            "Make it RELEVANT to this role"
        ]
    },
    "strength": {
        "patterns": [
            r"(greatest|biggest|key|main) strength",
            r"what are you good at",
            r"what do you do well",
            r"what's your superpower",
            r"strongest skill",
            r"what do you excel at"
        ],
        "strategy": "SPECIFIC STRENGTH + PROOF",
        "structure": [
            "Name ONE specific strength (not generic)",
            "Immediately prove it with a concrete example",
            "Show impact/outcome of that strength",
            "Connect to how it helps THIS role"
        ],
        "hook_templates": [
            "My core strength is {specific_strength} — for example, at {company} I {specific_example}.",
            "I'm exceptionally good at {specific_thing}, which I demonstrated when I {achievement}.",
            "What I do best is {strength} — this is why I was able to {specific_outcome}.",
            "If I had to pick one thing, it's my ability to {strength}. At {company}, this meant {example}."
        ],
        "likely_followups": [
            "Can you give another example?",
            "How would that help in this role?",
            "What's your second greatest strength?",
            "Has that strength ever been a weakness?"
        ],
        "trap_warnings": [
            "Don't be generic (teamwork, communication)",
            "Don't claim strengths you can't prove",
            "Pick something RELEVANT to the role"
        ]
    },
    "weakness": {
        "patterns": [
            r"(greatest|biggest) weakness",
            r"area (for|of) improvement",
            r"what are you working on",
            r"development area",
            r"where do you struggle",
            r"what would you improve about yourself"
        ],
        "strategy": "HONEST + MITIGATION + GROWTH",
        "structure": [
            "Name a REAL weakness (not fake humble-brag)",
            "Show self-awareness about its impact",
            "Describe SPECIFIC steps you're taking to improve",
            "Show progress/results from those efforts"
        ],
        "hook_templates": [
            "I've learned that I {real_weakness}. To address this, I've {specific_action}.",
            "Earlier in my career, I struggled with {weakness}. I've since {how_you_improved}.",
            "One area I'm actively developing is {weakness}. Recently, I {specific_improvement_action}.",
            "I tend to {weakness}, so I've built systems to {mitigation}."
        ],
        "likely_followups": [
            "How has that weakness affected your work?",
            "Can you give a specific example?",
            "What triggered you to work on this?",
            "How do you know you're improving?"
        ],
        "trap_warnings": [
            "Don't say 'perfectionism' or 'work too hard'",
            "Don't pick something critical to the role",
            "Don't be TOO honest about fatal flaws",
            "Show genuine self-awareness"
        ]
    },
    "achievement": {
        "patterns": [
            r"(proudest|biggest|greatest|most significant) (achievement|accomplishment)",
            r"tell me about a time you",
            r"describe a situation where",
            r"give me an example of",
            r"share an experience when"
        ],
        "strategy": "STAR WITH IMPACT",
        "structure": [
            "Set context briefly (Situation/Task) - 15%",
            "Focus on YOUR specific actions - 60%",
            "Quantify the result/impact - 20%",
            "Connect learning to THIS role - 5%"
        ],
        "hook_templates": [
            "The achievement I'm most proud of is {achievement} because {why_meaningful}.",
            "At {company}, I faced {challenge}. I {actions}, which resulted in {outcome}.",
            "When {situation}, I {your_action}. This led to {quantified_result}.",
            "My most impactful work was {achievement}, where I {specific_contribution}."
        ],
        "likely_followups": [
            "What would you do differently?",
            "What did you learn from this?",
            "How did others contribute?",
            "What was the hardest part?"
        ],
        "trap_warnings": [
            "Don't make Situation too long",
            "Focus on YOUR actions, not team's",
            "QUANTIFY the result",
            "Don't pick something irrelevant"
        ]
    },
    "conflict": {
        "patterns": [
            r"conflict",
            r"disagreement",
            r"difficult (person|colleague|coworker|situation)",
            r"challenging relationship",
            r"how do you handle",
            r"dealt with a difficult"
        ],
        "strategy": "PROFESSIONAL + RESOLUTION-FOCUSED",
        "structure": [
            "Describe situation professionally (no blame)",
            "Show your approach to understanding the other side",
            "Explain actions YOU took to resolve",
            "Show positive outcome and learning"
        ],
        "hook_templates": [
            "I approach conflicts as opportunities to find better solutions. For example, when {situation}...",
            "In a recent disagreement about {topic}, I first sought to understand {their_perspective}...",
            "When I encountered {challenge}, I focused on {your_approach}...",
            "I believe most conflicts stem from {insight}. When {situation}, I {action}..."
        ],
        "likely_followups": [
            "What if the person didn't change?",
            "What would you do differently?",
            "How do you prevent conflicts?",
            "Tell me about a conflict you didn't resolve well"
        ],
        "trap_warnings": [
            "Don't badmouth the other person",
            "Don't avoid the question",
            "Show emotional intelligence",
            "Pick a professional conflict, not personal"
        ]
    },
    "failure": {
        "patterns": [
            r"tell me about a (time you )?fail",
            r"mistake you made",
            r"something that didn't work",
            r"a setback",
            r"when things went wrong"
        ],
        "strategy": "HONEST + LEARNING + GROWTH",
        "structure": [
            "Describe the failure honestly",
            "Take responsibility (no blame-shifting)",
            "Explain what you learned",
            "Show how you've applied that learning"
        ],
        "hook_templates": [
            "One failure that taught me a lot was when I {failure}. I learned {learning}.",
            "Early in my career, I {mistake}. This taught me to {lesson}.",
            "A project that didn't go as planned was {project}. The key learning was {insight}.",
            "I failed when I {failure}, but it led me to {positive_change}."
        ],
        "likely_followups": [
            "How did others react?",
            "What would you do differently now?",
            "How do you prevent similar failures?",
            "What was the impact of the failure?"
        ],
        "trap_warnings": [
            "Don't pick something trivial",
            "Don't blame others",
            "Show genuine learning",
            "Don't pick something that shows poor judgment"
        ]
    },
    "salary": {
        "patterns": [
            r"salary expectations",
            r"compensation",
            r"what are you looking for",
            r"pay expectations",
            r"how much do you want to make",
            r"what's your expected salary"
        ],
        "strategy": "RESEARCH-BACKED + FLEXIBLE",
        "structure": [
            "Show you've done research on market rates",
            "Give a range (not a single number)",
            "Express flexibility based on total comp",
            "Redirect to fit and opportunity"
        ],
        "hook_templates": [
            "Based on my research and experience level, I'm targeting {range}, though I'm flexible based on total compensation.",
            "I've researched the market rate for this role, which seems to be {range}. I'm open to discussing based on the full package.",
            "My expectation is in the {range} range, but I'm more focused on finding the right fit and growth opportunity.",
            "For this level of role in {location}, I understand the range is typically {range}. I'd love to understand your budget."
        ],
        "likely_followups": [
            "What's your current salary?",
            "What's the minimum you'd accept?",
            "How did you arrive at that number?",
            "What other factors matter to you?"
        ],
        "trap_warnings": [
            "Don't give a number too early",
            "Don't lowball yourself",
            "Don't lie about current salary",
            "Research the company's pay bands"
        ]
    },
    "leadership": {
        "patterns": [
            r"leadership (experience|style|example)",
            r"led a team",
            r"managed (people|team)",
            r"describe your leadership",
            r"how do you lead"
        ],
        "strategy": "EXAMPLE + STYLE + IMPACT",
        "structure": [
            "Describe a specific leadership situation",
            "Explain YOUR leadership approach",
            "Show impact on team and results",
            "Connect to how you'd lead here"
        ],
        "hook_templates": [
            "My leadership style is {style}. For example, when I led {team/project}, I {approach}.",
            "I believe effective leadership is about {philosophy}. At {company}, I demonstrated this by {example}.",
            "Leading {team} taught me that {insight}. I achieved this by {actions}.",
            "When I led {initiative}, I focused on {approach}, which resulted in {outcome}."
        ],
        "likely_followups": [
            "How do you handle underperformers?",
            "How do you motivate your team?",
            "What's your biggest leadership mistake?",
            "How do you make difficult decisions?"
        ],
        "trap_warnings": [
            "Don't be vague about your style",
            "Show results, not just process",
            "Be honest about team contributions",
            "Leadership isn't just about managing people"
        ]
    },
    "technical": {
        "patterns": [
            r"technical (challenge|problem|decision)",
            r"complex (system|architecture|problem)",
            r"how would you (design|build|architect)",
            r"walk me through (your|a) technical"
        ],
        "strategy": "DEPTH + TRADEOFFS + REASONING",
        "structure": [
            "Describe the technical context clearly",
            "Explain your approach and reasoning",
            "Discuss tradeoffs you considered",
            "Share the outcome and learnings"
        ],
        "hook_templates": [
            "The most challenging technical problem I solved was {problem}. I approached it by {approach}.",
            "When designing {system}, I had to balance {tradeoff1} with {tradeoff2}. I chose {choice} because {reasoning}.",
            "At {company}, I built {system} that {outcome}. The key technical decision was {decision}.",
            "A complex architecture decision I made was {decision}. The tradeoffs were {tradeoffs}."
        ],
        "likely_followups": [
            "What would you do differently now?",
            "How did you handle scale?",
            "What were the failure modes?",
            "How did you test this?"
        ],
        "trap_warnings": [
            "Don't be too high-level",
            "Show depth of understanding",
            "Discuss tradeoffs, not just solutions",
            "Be honest about what you don't know"
        ]
    },
    "generic": {
        "patterns": [],
        "strategy": "SPECIFIC PROOF + RELEVANCE",
        "structure": [
            "Answer directly with specific evidence",
            "Connect to JD requirements",
            "Show concrete examples",
            "Tie to this opportunity"
        ],
        "hook_templates": [
            "Based on my experience with {relevant_experience}, I {answer}.",
            "This connects directly to my work at {company}, where I {example}.",
            "The most relevant example is when I {specific_example}.",
            "I approach this by {approach}, which I demonstrated when {example}."
        ],
        "likely_followups": [
            "Can you elaborate on that?",
            "How would that apply here?",
            "What's a specific example?",
            "What did you learn from that?"
        ],
        "trap_warnings": [
            "Don't be vague",
            "Always give specific examples",
            "Connect to the role"
        ]
    }
}


def detect_question_type(question: str) -> Tuple[str, Dict[str, Any]]:
    """Detect the type of question and return appropriate strategy."""
    question_lower = question.lower().strip()
    
    for q_type, q_config in QUESTION_STRATEGIES.items():
        if q_type == "generic":
            continue
        for pattern in q_config["patterns"]:
            if re.search(pattern, question_lower):
                return q_type, q_config
    
    return "generic", QUESTION_STRATEGIES["generic"]


# ============================================================
# 🛡️ ANTI-HALLUCINATION GROUNDING SYSTEM
# ============================================================

class FactGrounder:
    """Verify claims against resume facts to prevent hallucination."""
    
    def __init__(self):
        self.resume_facts: Dict[str, Any] = {}
        self.verified_claims: Set[str] = set()
        self.unverified_claims: Set[str] = set()
    
    async def extract_facts(self, resume_text: str, model: str) -> Dict[str, Any]:
        """Extract verifiable facts from resume."""
        if not resume_text.strip():
            return {}
        
        prompt = f"""Extract ONLY verifiable facts from this resume. Be precise and conservative.

RESUME:
{resume_text[:5000]}

Return STRICT JSON:
{{
    "companies": ["list of company names worked at"],
    "roles": ["list of job titles held"],
    "technologies": ["list of technologies mentioned"],
    "metrics": ["any quantified achievements with numbers"],
    "projects": ["named projects or systems"],
    "education": ["degrees, schools, certifications"],
    "years_experience": "approximate total years",
    "skills_claimed": ["skills explicitly claimed"],
    "achievements_verbatim": ["achievement statements as written"]
}}

ONLY include facts explicitly stated. Do NOT infer or add anything.
"""
        
        try:
            result = await _gen_text_smart("Extract facts as JSON.", prompt, model)
            match = re.search(r"\{[\s\S]*\}", result)
            if match:
                self.resume_facts = json.loads(match.group(0))
                return self.resume_facts
        except Exception as e:
            log_event("fact_extraction_fail", {"error": str(e)})
        
        return {}
    
    def verify_claim(self, claim: str, resume_text: str) -> Tuple[bool, str]:
        """Verify if a claim is grounded in resume facts."""
        claim_lower = claim.lower()
        resume_lower = resume_text.lower()
        
        # Check for specific keywords
        for company in self.resume_facts.get("companies", []):
            if company.lower() in claim_lower and company.lower() in resume_lower:
                self.verified_claims.add(claim)
                return True, f"Verified: mentions {company}"
        
        for tech in self.resume_facts.get("technologies", []):
            if tech.lower() in claim_lower and tech.lower() in resume_lower:
                self.verified_claims.add(claim)
                return True, f"Verified: mentions {tech}"
        
        for metric in self.resume_facts.get("metrics", []):
            # Check if numbers/percentages are from resume
            numbers_in_claim = re.findall(r'\d+%?', claim)
            numbers_in_resume = re.findall(r'\d+%?', resume_text)
            if numbers_in_claim and any(n in numbers_in_resume for n in numbers_in_claim):
                self.verified_claims.add(claim)
                return True, "Verified: metric from resume"
        
        # If claim has specific details not in resume, flag it
        self.unverified_claims.add(claim)
        return False, "Warning: claim may not be grounded in resume"


# ============================================================
# 📊 ANSWER QUALITY SCORING
# ============================================================

class AnswerQualityScorer:
    """Score answer quality across multiple dimensions."""
    
    def __init__(self):
        self.scores: Dict[str, float] = {}
        self.feedback: List[str] = []
    
    def score_answer(
        self,
        answer: str,
        question: str,
        q_type: str,
        company: str,
        jd_requirements: Dict[str, Any],
        resume_highlights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Score answer quality and provide feedback."""
        
        self.scores = {}
        self.feedback = []
        answer_lower = answer.lower()
        
        # 1. Hook Strength (0-10)
        first_sentence = answer.split('.')[0] if answer else ""
        hook_score = self._score_hook(first_sentence, company)
        self.scores["hook_strength"] = hook_score
        
        # 2. Specificity (0-10)
        specificity_score = self._score_specificity(answer, resume_highlights)
        self.scores["specificity"] = specificity_score
        
        # 3. Relevance to JD (0-10)
        relevance_score = self._score_relevance(answer, jd_requirements)
        self.scores["jd_relevance"] = relevance_score
        
        # 4. Company Alignment (0-10)
        company_score = self._score_company_alignment(answer, company)
        self.scores["company_alignment"] = company_score
        
        # 5. Confidence Tone (0-10)
        confidence_score = self._score_confidence(answer)
        self.scores["confidence"] = confidence_score
        
        # 6. Length Appropriateness (0-10)
        length_score = self._score_length(answer)
        self.scores["length"] = length_score
        
        # 7. No-Cliche Check (0-10)
        cliche_score = self._score_no_cliches(answer)
        self.scores["no_cliches"] = cliche_score
        
        # 8. Answer-Question Match (0-10)
        match_score = self._score_question_match(answer, question, q_type)
        self.scores["question_match"] = match_score
        
        # Calculate overall score
        weights = {
            "hook_strength": 0.15,
            "specificity": 0.20,
            "jd_relevance": 0.15,
            "company_alignment": 0.10,
            "confidence": 0.10,
            "length": 0.05,
            "no_cliches": 0.10,
            "question_match": 0.15
        }
        
        overall = sum(self.scores[k] * weights[k] for k in weights)
        
        return {
            "overall_score": round(overall, 1),
            "dimension_scores": self.scores,
            "feedback": self.feedback,
            "grade": self._get_grade(overall),
            "pass": overall >= 7.0
        }
    
    def _score_hook(self, first_sentence: str, company: str) -> float:
        """Score the opening hook."""
        score = 5.0  # Base score
        
        # Bonus for company mention in first sentence
        if company.lower() in first_sentence.lower():
            score += 1.5
        
        # Bonus for specific achievement
        if any(word in first_sentence.lower() for word in ["built", "led", "designed", "achieved", "delivered"]):
            score += 1.5
        
        # Bonus for confidence
        if first_sentence and not first_sentence.lower().startswith(("i am", "i'm", "thank you", "i would")):
            score += 1.0
        
        # Penalty for generic opening
        if any(phrase in first_sentence.lower() for phrase in ["i am writing", "thank you for", "i believe", "i am excited"]):
            score -= 2.0
            self.feedback.append("Opening is generic - start with a specific achievement or insight")
        
        return max(0, min(10, score))
    
    def _score_specificity(self, answer: str, resume_highlights: Dict[str, Any]) -> float:
        """Score how specific the answer is."""
        score = 5.0
        
        # Count specific elements
        has_numbers = bool(re.search(r'\d+', answer))
        has_company_names = any(c.lower() in answer.lower() for c in resume_highlights.get("companies_worked", []))
        has_technologies = any(t.lower() in answer.lower() for t in resume_highlights.get("technical_skills", []))
        has_action_verbs = sum(1 for v in ["built", "designed", "led", "implemented", "delivered", "launched"] if v in answer.lower())
        
        if has_numbers:
            score += 1.5
        else:
            self.feedback.append("Add quantified achievements with numbers")
        
        if has_company_names:
            score += 1.0
        
        if has_technologies:
            score += 1.5
        
        if has_action_verbs >= 2:
            score += 1.0
        
        return max(0, min(10, score))
    
    def _score_relevance(self, answer: str, jd_requirements: Dict[str, Any]) -> float:
        """Score relevance to JD requirements."""
        score = 5.0
        answer_lower = answer.lower()
        
        # Check for JD skill mentions
        skills = jd_requirements.get("must_have_skills", []) + jd_requirements.get("tech_stack", [])
        skills_mentioned = sum(1 for s in skills if s.lower() in answer_lower)
        
        if skills_mentioned >= 3:
            score += 3.0
        elif skills_mentioned >= 2:
            score += 2.0
        elif skills_mentioned >= 1:
            score += 1.0
        else:
            self.feedback.append("Mention more JD-specific skills and requirements")
        
        # Check for responsibility alignment
        responsibilities = jd_requirements.get("key_responsibilities", [])
        resp_aligned = sum(1 for r in responsibilities if any(word in answer_lower for word in r.lower().split()[:3]))
        
        if resp_aligned >= 2:
            score += 2.0
        
        return max(0, min(10, score))
    
    def _score_company_alignment(self, answer: str, company: str) -> float:
        """Score alignment with company culture."""
        score = 5.0
        answer_lower = answer.lower()
        
        intel = get_company_intelligence(company)
        
        # Check for company name
        if company.lower() in answer_lower:
            score += 2.0
        
        # Check for culture keyword usage
        culture_hits = sum(1 for k in intel.get("culture_keywords", []) if k.lower() in answer_lower)
        score += min(2.0, culture_hits * 0.5)
        
        # Check for avoided phrases
        avoid_hits = sum(1 for a in intel.get("avoid_saying", []) if a.lower() in answer_lower)
        if avoid_hits > 0:
            score -= avoid_hits * 1.5
            self.feedback.append(f"Avoid phrases like: {intel.get('avoid_saying', [])[:2]}")
        
        return max(0, min(10, score))
    
    def _score_confidence(self, answer: str) -> float:
        """Score confidence level of the answer."""
        score = 7.0  # Start high, deduct for issues
        answer_lower = answer.lower()
        
        # Penalty for hedging language
        hedges = ["i think", "i believe", "maybe", "perhaps", "i hope", "i would try"]
        hedge_count = sum(1 for h in hedges if h in answer_lower)
        score -= hedge_count * 0.5
        
        # Penalty for desperate language
        desperate = ["grateful for any", "hope you consider", "humbly", "please give me a chance"]
        if any(d in answer_lower for d in desperate):
            score -= 2.0
            self.feedback.append("Remove desperate/pleading language - be confident")
        
        # Bonus for confident language
        confident = ["i delivered", "i led", "i built", "i achieved", "i drove"]
        confidence_hits = sum(1 for c in confident if c in answer_lower)
        score += min(2.0, confidence_hits * 0.5)
        
        return max(0, min(10, score))
    
    def _score_length(self, answer: str) -> float:
        """Score answer length appropriateness."""
        words = len(answer.split())
        
        if 100 <= words <= 200:
            return 10.0
        elif 80 <= words <= 250:
            return 8.0
        elif 60 <= words <= 300:
            return 6.0
            self.feedback.append("Adjust answer length (target 120-180 words)")
        else:
            self.feedback.append("Answer is too short or too long")
            return 4.0
    
    def _score_no_cliches(self, answer: str) -> float:
        """Score absence of clichés."""
        score = 10.0
        answer_lower = answer.lower()
        
        cliches = [
            "passionate", "team player", "hard worker", "go-getter",
            "think outside the box", "synergy", "leverage", "dynamic",
            "results-driven", "detail-oriented", "self-starter",
            "fast learner", "people person", "perfectionist"
        ]
        
        cliche_count = sum(1 for c in cliches if c in answer_lower)
        score -= cliche_count * 2.0
        
        if cliche_count > 0:
            self.feedback.append(f"Remove clichés: found {cliche_count} generic phrases")
        
        return max(0, min(10, score))
    
    def _score_question_match(self, answer: str, question: str, q_type: str) -> float:
        """Score how well the answer matches the question type."""
        score = 7.0
        answer_lower = answer.lower()
        
        if q_type == "why_hire_you":
            if "unique" in answer_lower or "different" in answer_lower or "sets me apart" in answer_lower:
                score += 2.0
            if any(v in answer_lower for v in ["built", "achieved", "delivered"]):
                score += 1.0
        
        elif q_type == "why_this_company":
            # Should mention company-specific things
            if re.search(r"(specifically|unique|particular|your)", answer_lower):
                score += 2.0
        
        elif q_type == "weakness":
            # Should show growth
            if any(w in answer_lower for w in ["learned", "improved", "working on", "developed"]):
                score += 2.0
            else:
                self.feedback.append("Show growth/improvement on the weakness")
        
        elif q_type == "achievement":
            # Should have STAR elements
            has_result = bool(re.search(r'(result|led to|achieved|improved)', answer_lower))
            if has_result:
                score += 2.0
            else:
                self.feedback.append("Include clear result/outcome of your achievement")
        
        return max(0, min(10, score))
    
    def _get_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 9.0:
            return "A+"
        elif score >= 8.5:
            return "A"
        elif score >= 8.0:
            return "A-"
        elif score >= 7.5:
            return "B+"
        elif score >= 7.0:
            return "B"
        elif score >= 6.5:
            return "B-"
        elif score >= 6.0:
            return "C+"
        elif score >= 5.5:
            return "C"
        else:
            return "Needs Improvement"


# ============================================================
# 🔍 SKILL GAP ANALYZER
# ============================================================

async def analyze_skill_gaps(
    resume_highlights: Dict[str, Any],
    jd_requirements: Dict[str, Any],
    model: str
) -> Dict[str, Any]:
    """Analyze gaps between resume and JD requirements."""
    
    resume_skills = set(s.lower() for s in resume_highlights.get("technical_skills", []))
    jd_must_have = set(s.lower() for s in jd_requirements.get("must_have_skills", []))
    jd_nice_to_have = set(s.lower() for s in jd_requirements.get("nice_to_have_skills", []))
    
    # Find gaps
    must_have_gaps = jd_must_have - resume_skills
    nice_to_have_gaps = jd_nice_to_have - resume_skills
    
    # Find matches
    must_have_matches = jd_must_have & resume_skills
    nice_to_have_matches = jd_nice_to_have & resume_skills
    
    # Generate addressing strategies for gaps
    gap_strategies = {}
    for gap in list(must_have_gaps)[:5]:
        gap_strategies[gap] = _get_gap_addressing_strategy(gap, resume_skills)
    
    return {
        "must_have_matches": list(must_have_matches),
        "must_have_gaps": list(must_have_gaps),
        "nice_to_have_matches": list(nice_to_have_matches),
        "nice_to_have_gaps": list(nice_to_have_gaps),
        "match_percentage": len(must_have_matches) / max(1, len(jd_must_have)) * 100,
        "gap_strategies": gap_strategies,
        "strengths_to_emphasize": list(must_have_matches)[:5],
        "transferable_skills": _find_transferable_skills(resume_skills, must_have_gaps)
    }


def _get_gap_addressing_strategy(gap: str, resume_skills: Set[str]) -> str:
    """Get strategy to address a specific skill gap."""
    strategies = {
        "default": f"Acknowledge you're developing {gap} skills while highlighting related experience",
        "learning": f"Express genuine interest in deepening {gap} expertise through this role",
        "transferable": f"Connect your existing skills to {gap} through similar problem-solving approaches"
    }
    
    # Check for related skills
    related_found = [s for s in resume_skills if any(word in s for word in gap.split())]
    if related_found:
        return f"Leverage your {related_found[0]} experience as foundation for {gap}"
    
    return strategies["learning"]


def _find_transferable_skills(resume_skills: Set[str], gaps: Set[str]) -> List[str]:
    """Find transferable skills that could address gaps."""
    transferable = []
    
    skill_families = {
        "python": ["programming", "scripting", "automation"],
        "sql": ["database", "data", "query"],
        "machine learning": ["ai", "ml", "modeling", "statistics"],
        "aws": ["cloud", "infrastructure", "devops"],
        "docker": ["containerization", "kubernetes", "devops"],
        "spark": ["big data", "distributed", "hadoop"]
    }
    
    for skill in resume_skills:
        for gap in gaps:
            skill_lower = skill.lower()
            gap_lower = gap.lower()
            
            # Check if in same family
            for family_key, family_terms in skill_families.items():
                if any(t in skill_lower for t in [family_key] + family_terms):
                    if any(t in gap_lower for t in [family_key] + family_terms):
                        transferable.append(f"{skill} → {gap}")
    
    return transferable[:5]


# ============================================================
# 🚨 RED FLAG DETECTION
# ============================================================

async def detect_red_flags(
    resume_text: str,
    resume_highlights: Dict[str, Any],
    model: str
) -> Dict[str, Any]:
    """Detect potential red flags that need proactive addressing."""
    
    red_flags = []
    addressing_strategies = {}
    
    # Check for employment gaps (simplified heuristic)
    years_mentioned = re.findall(r'20\d{2}', resume_text)
    if years_mentioned:
        years = sorted(set(int(y) for y in years_mentioned))
        for i in range(len(years) - 1):
            if years[i+1] - years[i] > 1:
                gap = f"Potential gap between {years[i]} and {years[i+1]}"
                red_flags.append(gap)
                addressing_strategies[gap] = "Be prepared to explain this period positively (learning, personal project, etc.)"
    
    # Check for short tenures
    companies = resume_highlights.get("companies_worked", [])
    if len(companies) > 3:
        red_flags.append("Multiple companies - may raise job-hopping concerns")
        addressing_strategies["job_hopping"] = "Frame as intentional career growth and diverse experience"
    
    # Check for career change signals
    roles = resume_highlights.get("roles", [])
    if roles:
        role_types = set(r.lower().split()[0] for r in roles if r)
        if len(role_types) > 2:
            red_flags.append("Diverse role types - may raise focus concerns")
            addressing_strategies["career_change"] = "Present as intentional breadth-building for current goals"
    
    return {
        "red_flags": red_flags,
        "addressing_strategies": addressing_strategies,
        "proactive_topics": list(addressing_strategies.keys())
    }


# ============================================================
# 🎭 PERSONAL BRAND EXTRACTOR
# ============================================================

async def extract_personal_brand(
    resume_highlights: Dict[str, Any],
    model: str
) -> Dict[str, Any]:
    """Extract candidate's personal brand/theme for consistency."""
    
    achievements = resume_highlights.get("top_achievements", [])
    skills = resume_highlights.get("technical_skills", [])
    leadership = resume_highlights.get("leadership_examples", [])
    
    # Identify recurring themes
    themes = []
    
    # Technical depth theme
    if len(skills) > 5:
        themes.append("technical depth and expertise")
    
    # Leadership theme
    if leadership:
        themes.append("technical leadership and ownership")
    
    # Impact theme
    if any("impact" in str(a).lower() or "improved" in str(a).lower() for a in achievements):
        themes.append("measurable impact and results")
    
    # Innovation theme
    if any("built" in str(a).lower() or "created" in str(a).lower() or "designed" in str(a).lower() for a in achievements):
        themes.append("building and creating")
    
    # Primary brand statement
    primary_theme = themes[0] if themes else "technical excellence"
    
    return {
        "primary_theme": primary_theme,
        "supporting_themes": themes[1:3] if len(themes) > 1 else [],
        "brand_statement": f"A professional known for {primary_theme}",
        "consistency_keywords": themes,
        "differentiators": leadership[:2] if leadership else achievements[:2]
    }


# ============================================================
# 🔮 FOLLOW-UP QUESTION PREDICTOR
# ============================================================

def predict_followup_questions(
    question: str,
    answer: str,
    q_type: str,
    q_strategy: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Predict likely follow-up questions based on the answer."""
    
    followups = []
    
    # Get strategy-defined follow-ups
    likely_followups = q_strategy.get("likely_followups", [])
    for f in likely_followups[:3]:
        followups.append({
            "question": f,
            "type": "standard",
            "preparation_tip": "Be ready with a specific example"
        })
    
    # Analyze answer for specific follow-up triggers
    answer_lower = answer.lower()
    
    # If numbers mentioned, expect drilling
    if re.search(r'\d+%?', answer):
        followups.append({
            "question": "How did you measure that? / How did you arrive at that number?",
            "type": "verification",
            "preparation_tip": "Be ready to explain your methodology"
        })
    
    # If project mentioned, expect details
    if "project" in answer_lower or "system" in answer_lower:
        followups.append({
            "question": "Tell me more about the technical architecture",
            "type": "depth",
            "preparation_tip": "Know the technical details of what you mentioned"
        })
    
    # If team mentioned, expect collaboration questions
    if "team" in answer_lower or "collaborated" in answer_lower:
        followups.append({
            "question": "How did you handle disagreements with the team?",
            "type": "behavioral",
            "preparation_tip": "Have a specific conflict resolution example ready"
        })
    
    return followups[:5]


# ============================================================
# 🔒 UTILITIES
# ============================================================

def _tex_safe(s: str) -> str:
    try:
        return secure_tex_input(s)
    except TypeError:
        return secure_tex_input("inline.txt", s)


def _is_responses_only_model(name: str) -> bool:
    if not name:
        return False
    return bool(re.match(r"^(gpt-image|dall[- ]?e|whisper)", name, flags=re.I))


def _get_session_key(jd_text: str, resume_text: str) -> str:
    """Generate session key for consistency tracking."""
    content = (jd_text[:500] + resume_text[:500]).encode()
    return hashlib.md5(content).hexdigest()[:16]


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
    # New optional fields
    interview_stage: Optional[str] = None  # recruiter, technical, hiring_manager, final
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


def _coerce_key_from_ctx(ctx: Dict[str, Any], fallback_path: Optional[Path]) -> str:
    if ctx.get("key"):
        return str(ctx["key"]).strip()
    c, r = (ctx.get("company") or "").strip(), (ctx.get("role") or "").strip()
    if c and r:
        return f"{safe_filename(c)}__{safe_filename(r)}"
    return fallback_path.stem if fallback_path else ""


def _pick_resume_from_ctx(ctx: Dict[str, Any]) -> str:
    h = (ctx.get("humanized") or {})
    o = (ctx.get("optimized") or {})
    for candidate in (h.get("tex"), o.get("tex"), ctx.get("humanized_tex"), ctx.get("resume_tex")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return ""


def _pick_coverletter_from_ctx(ctx: Dict[str, Any]) -> str:
    cl = (ctx.get("cover_letter") or {})
    v = cl.get("tex")
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
        meta = {
            "key": _coerce_key_from_ctx(ctx, path),
            "company": ctx.get("company"),
            "role": ctx.get("role"),
        }
        log_event("talk_context_used", meta)
    return ctx, path


# ============================================================
# 🧩 OPENAI HELPERS
# ============================================================

async def _gen_text_smart(system: str, user: str, model: str) -> str:
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI SDK not installed.")
    if not (getattr(config, "OPENAI_API_KEY", "") or "").strip():
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY missing.")

    requested_model = (model or "").strip() or CHAT_SAFE_DEFAULT

    if _is_responses_only_model(requested_model):
        requested_model = CHAT_SAFE_DEFAULT

    try:
        r = await openai_client.chat.completions.create(
            model=requested_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        log_event("talk_gen_fail", {"error": str(e), "model": requested_model})
        raise


# ============================================================
# 🔎 RESUME & JD ANALYSIS
# ============================================================

async def extract_resume_highlights(resume_text: str, model: str = SUMMARIZER_MODEL) -> Dict[str, Any]:
    """Extract KEY achievements and skills from resume."""
    if not (resume_text or "").strip():
        return {"achievements": [], "skills": [], "experiences": [], "companies_worked": [], "technical_skills": []}

    prompt = f"""Extract the MOST IMPRESSIVE and RELEVANT highlights from this resume.

RESUME:
{resume_text[:5000]}

Return STRICT JSON:
{{
    "top_achievements": ["3-5 most impressive achievements with QUANTIFIED outcomes"],
    "technical_skills": ["key technical skills with context"],
    "leadership_examples": ["ownership/leadership examples"],
    "unique_strengths": ["what makes this candidate unique"],
    "companies_worked": ["company names"],
    "quantified_results": ["any results with numbers/metrics"],
    "roles": ["job titles held"]
}}

Focus on SPECIFIC, IMPRESSIVE, QUANTIFIED achievements.
"""

    try:
        result = await _gen_text_smart("Extract resume highlights as JSON.", prompt, model)
        match = re.search(r"\{[\s\S]*\}", result)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        log_event("resume_highlight_fail", {"error": str(e)})
    
    return {"achievements": [], "skills": [], "experiences": [], "companies_worked": [], "technical_skills": []}


async def extract_jd_requirements(jd_text: str, model: str = SUMMARIZER_MODEL) -> Dict[str, Any]:
    """Extract key requirements from JD."""
    if not (jd_text or "").strip():
        return {"requirements": [], "tech_stack": [], "responsibilities": [], "must_have_skills": [], "nice_to_have_skills": [], "key_responsibilities": []}

    prompt = f"""Extract the KEY requirements from this job description.

JOB DESCRIPTION:
{jd_text[:4000]}

Return STRICT JSON:
{{
    "must_have_skills": ["required technical skills"],
    "nice_to_have_skills": ["preferred skills"],
    "key_responsibilities": ["main job duties"],
    "tech_stack": ["specific technologies mentioned"],
    "team_context": "what team/product this is for",
    "success_metrics": ["how success might be measured"],
    "company_challenges": ["challenges this role addresses"]
}}

Be specific. Extract REAL requirements from the JD.
"""

    try:
        result = await _gen_text_smart("Extract JD requirements as JSON.", prompt, model)
        match = re.search(r"\{[\s\S]*\}", result)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        log_event("jd_extract_fail", {"error": str(e)})
    
    return {"requirements": [], "tech_stack": [], "responsibilities": [], "must_have_skills": [], "nice_to_have_skills": [], "key_responsibilities": []}


# ============================================================
# 💬 ULTIMATE KILLER ANSWER GENERATION
# ============================================================

async def generate_killer_answer(
    jd_text: str,
    resume_text: str,
    question: str,
    company: str,
    role: str,
    model: str,
    cover_letter: str = "",
    resume_highlights: Optional[Dict[str, Any]] = None,
    jd_requirements: Optional[Dict[str, Any]] = None,
    skill_gaps: Optional[Dict[str, Any]] = None,
    personal_brand: Optional[Dict[str, Any]] = None,
    interview_stage: Optional[str] = None,
) -> str:
    """Generate an ULTIMATE KILLER answer with all enhancements."""
    
    # Detect question type and get strategy
    q_type, q_strategy = detect_question_type(question)
    
    # Get company intelligence
    company_intel = get_company_intelligence(company)
    
    # Build context strings
    achievements_str = ""
    if resume_highlights:
        achievements = resume_highlights.get("top_achievements", [])
        if achievements:
            achievements_str = "TOP ACHIEVEMENTS FROM RESUME (use these EXACT facts):\n" + "\n".join(f"• {a}" for a in achievements[:5])
    
    jd_reqs_str = ""
    if jd_requirements:
        skills = jd_requirements.get("must_have_skills", [])
        responsibilities = jd_requirements.get("key_responsibilities", [])
        if skills:
            jd_reqs_str += "REQUIRED SKILLS (mention 2-3): " + ", ".join(skills[:8]) + "\n"
        if responsibilities:
            jd_reqs_str += "KEY RESPONSIBILITIES: " + "; ".join(responsibilities[:5])
    
    # Skill gap context
    gap_context = ""
    if skill_gaps and skill_gaps.get("must_have_gaps"):
        gaps = skill_gaps.get("must_have_gaps", [])[:3]
        strategies = skill_gaps.get("gap_strategies", {})
        gap_context = f"\n⚠️ SKILL GAPS TO ADDRESS HONESTLY (if relevant):\n"
        for gap in gaps:
            strategy = strategies.get(gap, "Show willingness to learn")
            gap_context += f"• {gap}: {strategy}\n"
    
    # Personal brand context
    brand_context = ""
    if personal_brand:
        brand_context = f"\n🎯 PERSONAL BRAND THEME: {personal_brand.get('primary_theme', 'technical excellence')}\n"
        brand_context += f"Consistently emphasize: {', '.join(personal_brand.get('consistency_keywords', []))}\n"
    
    # Interview stage calibration
    stage_context = ""
    if interview_stage:
        stage_intel = company_intel.get("interview_stages", {}).get(interview_stage, "")
        if stage_intel:
            stage_context = f"\n📋 INTERVIEW STAGE: {interview_stage.upper()}\nFocus: {stage_intel}\n"
    
    what_impresses = company_intel.get("what_impresses_them", [])
    avoid_saying = company_intel.get("avoid_saying", [])
    
    # Build the ultimate prompt
    sys_prompt = f"""You are helping a candidate give a KILLER answer that will WIN the interview.

🎯 QUESTION TYPE: {q_type.upper()}
📋 STRATEGY: {q_strategy['strategy']}

REQUIRED STRUCTURE:
{chr(10).join(f"• {s}" for s in q_strategy['structure'])}

🏢 WHAT {company.upper()} VALUES:
{chr(10).join(f"• {v}" for v in company_intel.get('what_they_value', [])[:4])}

✅ WHAT IMPRESSES THEM:
{chr(10).join(f"• {w}" for w in what_impresses[:4])}

❌ AVOID SAYING (company-specific):
{chr(10).join(f"• {a}" for a in avoid_saying[:4])}

💡 OPENING HOOK IDEAS (adapt one to be specific):
{chr(10).join(f"• {h}" for h in q_strategy['hook_templates'][:3])}

⚠️ TRAP WARNINGS:
{chr(10).join(f"• {t}" for t in q_strategy.get('trap_warnings', [])[:3])}
{stage_context}
{brand_context}
{gap_context}

📝 ANSWER REQUIREMENTS:
1. HOOK: First sentence must grab attention - NO generic openings
2. GROUNDING: Use ONLY facts from the resume - do NOT invent achievements
3. SPECIFICITY: Include EXACT numbers, tools, company names from resume
4. RELEVANCE: Map directly to THIS company's needs using JD keywords
5. FORWARD: Show what you'll CONTRIBUTE, not just what you've done
6. CONFIDENCE: Sound natural and confident - no hedging or pleading
7. LENGTH: 2-3 paragraphs, 120-180 words total

🚫 ABSOLUTE RULES - NEVER VIOLATE:
- Do NOT invent achievements, metrics, or company names not in resume
- Do NOT use clichés: "passionate", "team player", "hard worker", "excited"
- Do NOT start with: "I am writing...", "Thank you for...", "I believe..."
- Do NOT mention GPA, graduation dates, or academic achievements
- Do NOT sound desperate: "grateful for opportunity", "hope you consider"
- Do NOT be generic - EVERY sentence must be specific to THIS role/company
- Do NOT use hedging: "I think", "maybe", "perhaps"

✨ TONE: Confident professional who knows their worth and has done their research.
"""

    user_prompt = f"""Generate a KILLER answer for this question:

QUESTION: {question}

COMPANY: {company}
ROLE: {role}

{achievements_str}

{jd_reqs_str}

JOB DESCRIPTION (key excerpts):
{jd_text[:3000]}

RESUME (ONLY use facts from here - do not invent):
{resume_text[:3000]}

{f'COVER LETTER (additional context):{chr(10)}{cover_letter[:1500]}' if cover_letter else ''}

Remember:
- Open with a HOOK that grabs attention
- Use ONLY facts from the resume
- Be SPECIFIC with examples and outcomes
- Show you understand {company}'s unique needs
- Sound CONFIDENT but not arrogant
- 2-3 paragraphs, 120-180 words
"""

    start = time.time()
    answer = await _gen_text_smart(sys_prompt, user_prompt, model=model)
    latency = round(time.time() - start, 2)
    
    log_event("killer_answer_generated", {
        "question_type": q_type,
        "company": company,
        "latency": latency,
        "words": len(answer.split()),
        "interview_stage": interview_stage
    })

    return _tex_safe(answer)


# ============================================================
# ✨ HUMANIZE (with answer-specific instructions)
# ============================================================

async def humanize_answer(answer_text: str, tone: str, q_type: str) -> Tuple[str, bool]:
    """Refine the answer while preserving killer elements."""
    api_base = (getattr(config, "API_BASE_URL", "") or "").rstrip("/") or "http://127.0.0.1:8000"
    url = f"{api_base}/api/superhuman/rewrite"
    
    instructions = (
        f"Rewrite this {q_type} interview answer. "
        "PRESERVE: the opening hook, specific achievements, numbers, company names, and confident tone. "
        "IMPROVE: natural flow, remove any AI-sounding phrases, make it sound like a real human. "
        "KEEP: 2-3 paragraphs, 120-180 words total. "
        "DO NOT: add clichés, make it generic, remove specific details, or change facts."
    )
    
    payload = {
        "text": instructions + "\n\n" + answer_text,
        "mode": "interview_answer",
        "tone": tone,
        "latex_safe": True,
    }

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        rewritten = data.get("rewritten") or answer_text
        was_humanized = isinstance(rewritten, str) and rewritten.strip() != answer_text.strip()
        return _tex_safe(rewritten), was_humanized
    except Exception as e:
        log_event("talk_humanize_fail", {"error": str(e)})
        return answer_text, False


# ============================================================
# 🔄 ANSWER IMPROVEMENT LOOP
# ============================================================

async def improve_answer_if_needed(
    answer: str,
    quality_score: Dict[str, Any],
    question: str,
    company: str,
    role: str,
    jd_requirements: Dict[str, Any],
    resume_highlights: Dict[str, Any],
    model: str,
    max_iterations: int = 2
) -> str:
    """Improve answer if quality score is below threshold."""
    
    if quality_score.get("overall_score", 0) >= 7.5:
        return answer  # Good enough
    
    if max_iterations <= 0:
        return answer  # Gave up
    
    feedback = quality_score.get("feedback", [])
    if not feedback:
        return answer
    
    improvement_prompt = f"""Improve this interview answer based on the following feedback:

CURRENT ANSWER:
{answer}

FEEDBACK TO ADDRESS:
{chr(10).join(f"• {f}" for f in feedback)}

QUESTION: {question}
COMPANY: {company}
ROLE: {role}

Requirements:
- Fix the specific issues mentioned in feedback
- Keep the same length (120-180 words)
- Maintain confidence and specificity
- Do NOT add clichés or generic phrases
- Do NOT invent new achievements

Return only the improved answer, nothing else.
"""

    try:
        improved = await _gen_text_smart(
            "You are improving an interview answer based on specific feedback.",
            improvement_prompt,
            model
        )
        
        # Re-score
        scorer = AnswerQualityScorer()
        new_score = scorer.score_answer(
            improved, question, "generic", company, jd_requirements, resume_highlights
        )
        
        if new_score.get("overall_score", 0) > quality_score.get("overall_score", 0):
            log_event("answer_improved", {
                "old_score": quality_score.get("overall_score"),
                "new_score": new_score.get("overall_score")
            })
            return improved
        
    except Exception as e:
        log_event("answer_improvement_fail", {"error": str(e)})
    
    return answer


# ============================================================
# 🟢 HEALTH CHECK
# ============================================================

@router.get("/ping")
async def ping():
    now = datetime.now(tz=timezone.utc)
    return {"ok": True, "service": "talk", "version": "4.0.0", "epoch": time.time(), "iso": now.isoformat()}


# ============================================================
# 🚀 MAIN ENDPOINT - ULTIMATE VERSION
# ============================================================

@router.post("/answer")
@router.post("")
async def talk_to_hirex(req: TalkReq):
    """
    Generate an ULTIMATE, interview-winning answer.
    
    Features:
    - Question type detection with tailored strategy
    - Company-specific intelligence (values, culture, what impresses)
    - Anti-hallucination grounding (verify claims against resume)
    - Answer quality scoring with feedback
    - Skill gap analysis with addressing strategies
    - Follow-up question prediction
    - Personal brand consistency
    - Interview stage calibration
    - Automatic answer improvement loop
    """
    
    # Load context if needed
    jd_text = (req.jd_text or "").strip()
    resume_tex = (req.resume_tex or "").strip()
    cover_letter_tex = ""
    used_key = ""
    used_company = ""
    used_role = ""

    if (not jd_text) or (not resume_tex and not (req.resume_plain or "").strip()):
        ctx, ctx_path = _load_context(req)
        if ctx:
            jd_text = jd_text or (ctx.get("jd_text") or "")
            resume_tex = resume_tex or _pick_resume_from_ctx(ctx)
            cover_letter_tex = _pick_coverletter_from_ctx(ctx)
            used_key = _coerce_key_from_ctx(ctx, ctx_path)
            used_company = (ctx.get("company") or "").strip()
            used_role = (ctx.get("role") or "").strip()

    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="Job Description missing.")
    if not (resume_tex or (req.resume_plain or "").strip()):
        raise HTTPException(status_code=400, detail="Resume text missing.")

    resume_text = resume_tex or req.resume_plain or ""
    model = (req.model or ANSWER_MODEL).strip() or ANSWER_MODEL
    
    # Extract company and role from JD if not from context
    if not used_company or not used_role:
        try:
            extract_prompt = f"""Extract company and role from this JD.
Return JSON: {{"company": "...", "role": "..."}}
JD: {jd_text[:2000]}"""
            result = await _gen_text_smart("Extract as JSON.", extract_prompt, SUMMARIZER_MODEL)
            match = re.search(r"\{[\s\S]*\}", result)
            if match:
                data = json.loads(match.group(0))
                used_company = used_company or data.get("company", "Company")
                used_role = used_role or data.get("role", "Role")
        except Exception:
            used_company = used_company or "Company"
            used_role = used_role or "Role"

    # Detect question type
    q_type, q_strategy = detect_question_type(req.question)

    # Extract resume highlights and JD requirements
    resume_highlights = await extract_resume_highlights(resume_text, model)
    jd_requirements = await extract_jd_requirements(jd_text, model)

    # Analyze skill gaps
    skill_gaps = await analyze_skill_gaps(resume_highlights, jd_requirements, model)

    # Extract personal brand
    personal_brand = await extract_personal_brand(resume_highlights, model)

    # Detect red flags
    red_flags = await detect_red_flags(resume_text, resume_highlights, model)

    # Generate killer answer
    draft_answer = await generate_killer_answer(
        jd_text=jd_text,
        resume_text=resume_text,
        question=req.question,
        company=used_company,
        role=used_role,
        model=model,
        cover_letter=cover_letter_tex,
        resume_highlights=resume_highlights,
        jd_requirements=jd_requirements,
        skill_gaps=skill_gaps,
        personal_brand=personal_brand,
        interview_stage=req.interview_stage,
    )

    # Score answer quality
    scorer = AnswerQualityScorer()
    quality_score = scorer.score_answer(
        draft_answer,
        req.question,
        q_type,
        used_company,
        jd_requirements,
        resume_highlights
    )

    # Improve if score is low
    if quality_score.get("overall_score", 0) < 7.5:
        draft_answer = await improve_answer_if_needed(
            draft_answer,
            quality_score,
            req.question,
            used_company,
            used_role,
            jd_requirements,
            resume_highlights,
            model
        )
        # Re-score
        quality_score = scorer.score_answer(
            draft_answer, req.question, q_type, used_company, jd_requirements, resume_highlights
        )

    # Humanize if requested
    if req.humanize:
        final_answer, was_humanized = await humanize_answer(draft_answer, req.tone, q_type)
    else:
        final_answer, was_humanized = draft_answer, False

    # Predict follow-up questions
    followup_predictions = []
    if req.include_followups:
        followup_predictions = predict_followup_questions(req.question, final_answer, q_type, q_strategy)

    # Get company intel summary
    company_intel = get_company_intelligence(used_company)

    # Log
    log_event("talk_ultimate_answer", {
        "question": req.question[:100],
        "question_type": q_type,
        "company": used_company,
        "role": used_role,
        "humanized": was_humanized,
        "quality_score": quality_score.get("overall_score"),
        "quality_grade": quality_score.get("grade"),
        "interview_stage": req.interview_stage
    })

    # Build response (same interface as before, with additional fields)
    response = {
        # Original fields (backward compatible)
        "question": req.question.strip(),
        "question_type": q_type,
        "strategy_used": q_strategy["strategy"],
        "draft_answer": draft_answer,
        "final_text": final_answer,
        "answer": final_answer,
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
            "key": used_key,
            "company": used_company,
            "role": used_role,
            "has_cover_letter": bool(cover_letter_tex),
        },
        
        # NEW: Quality scoring
        "quality": {
            "overall_score": quality_score.get("overall_score", 0),
            "grade": quality_score.get("grade", "N/A"),
            "pass": quality_score.get("pass", False),
            "dimension_scores": quality_score.get("dimension_scores", {}),
            "feedback": quality_score.get("feedback", []),
        } if req.include_quality_score else None,
        
        # NEW: Follow-up predictions
        "predicted_followups": followup_predictions if req.include_followups else None,
        
        # NEW: Skill gap analysis
        "skill_analysis": {
            "match_percentage": skill_gaps.get("match_percentage", 0),
            "strengths_to_emphasize": skill_gaps.get("strengths_to_emphasize", [])[:5],
            "gaps_to_address": skill_gaps.get("must_have_gaps", [])[:3],
            "gap_strategies": skill_gaps.get("gap_strategies", {}),
            "transferable_skills": skill_gaps.get("transferable_skills", [])[:3],
        },
        
        # NEW: Personal brand
        "personal_brand": {
            "primary_theme": personal_brand.get("primary_theme", ""),
            "brand_statement": personal_brand.get("brand_statement", ""),
            "consistency_keywords": personal_brand.get("consistency_keywords", [])[:3],
        },
        
        # NEW: Red flags detected
        "red_flags": {
            "detected": red_flags.get("red_flags", []),
            "addressing_strategies": red_flags.get("addressing_strategies", {}),
        } if red_flags.get("red_flags") else None,
        
        # NEW: Interview stage context
        "interview_stage": {
            "stage": req.interview_stage,
            "focus": company_intel.get("interview_stages", {}).get(req.interview_stage, ""),
        } if req.interview_stage else None,
        
        # NEW: Trap warnings for this question type
        "trap_warnings": q_strategy.get("trap_warnings", [])[:3],
        
        # NEW: Salary intelligence (if salary question)
        "salary_intel": company_intel.get("salary_range") if q_type == "salary" else None,
        
        # NEW: Common questions for this company
        "company_common_questions": company_intel.get("common_questions", [])[:5],
        
        # NEW: Why not competitors (if "why this company" question)
        "competitor_differentiation": {
            "competitors": company_intel.get("competitors", []),
            "why_not_them": company_intel.get("why_not_competitors", ""),
        } if q_type == "why_this_company" else None,
    }
    
    # Clean up None values for cleaner response
    response = {k: v for k, v in response.items() if v is not None}
    
    return response


# ============================================================
# 📚 ADDITIONAL ENDPOINTS
# ============================================================

@router.get("/company-intel/{company}")
async def get_company_intel(company: str):
    """Get company intelligence for preparation."""
    intel = get_company_intelligence(company)
    return {
        "company": company,
        "found": intel != DEFAULT_COMPANY_INTELLIGENCE,
        "intelligence": intel
    }


@router.get("/question-types")
async def list_question_types():
    """List all supported question types and their strategies."""
    return {
        "question_types": {
            q_type: {
                "strategy": config["strategy"],
                "structure": config["structure"],
                "hook_templates": config["hook_templates"][:2],
                "trap_warnings": config.get("trap_warnings", [])[:2],
                "likely_followups": config.get("likely_followups", [])[:3]
            }
            for q_type, config in QUESTION_STRATEGIES.items()
        }
    }


@router.post("/analyze-gaps")
async def analyze_gaps_endpoint(
    jd_text: str,
    resume_text: str,
    model: str = SUMMARIZER_MODEL
):
    """Analyze skill gaps between resume and JD."""
    resume_highlights = await extract_resume_highlights(resume_text, model)
    jd_requirements = await extract_jd_requirements(jd_text, model)
    skill_gaps = await analyze_skill_gaps(resume_highlights, jd_requirements, model)
    
    return {
        "resume_skills": resume_highlights.get("technical_skills", []),
        "jd_requirements": jd_requirements.get("must_have_skills", []),
        "analysis": skill_gaps
    }


@router.post("/score-answer")
async def score_answer_endpoint(
    answer: str,
    question: str,
    company: str,
    jd_text: str = "",
    resume_text: str = ""
):
    """Score an answer and get improvement feedback."""
    q_type, _ = detect_question_type(question)
    
    jd_requirements = {}
    resume_highlights = {}
    
    if jd_text:
        jd_requirements = await extract_jd_requirements(jd_text, SUMMARIZER_MODEL)
    if resume_text:
        resume_highlights = await extract_resume_highlights(resume_text, SUMMARIZER_MODEL)
    
    scorer = AnswerQualityScorer()
    score = scorer.score_answer(
        answer, question, q_type, company, jd_requirements, resume_highlights
    )
    
    return {
        "question": question,
        "question_type": q_type,
        "company": company,
        "score": score
    }


@router.post("/predict-followups")
async def predict_followups_endpoint(
    question: str,
    answer: str
):
    """Predict likely follow-up questions."""
    q_type, q_strategy = detect_question_type(question)
    followups = predict_followup_questions(question, answer, q_type, q_strategy)
    
    return {
        "question": question,
        "question_type": q_type,
        "predicted_followups": followups
    }


@router.post("/extract-brand")
async def extract_brand_endpoint(
    resume_text: str,
    model: str = SUMMARIZER_MODEL
):
    """Extract personal brand from resume."""
    resume_highlights = await extract_resume_highlights(resume_text, model)
    personal_brand = await extract_personal_brand(resume_highlights, model)
    
    return {
        "personal_brand": personal_brand,
        "resume_highlights": resume_highlights
    }


@router.post("/detect-red-flags")
async def detect_red_flags_endpoint(
    resume_text: str,
    model: str = SUMMARIZER_MODEL
):
    """Detect potential red flags in resume."""
    resume_highlights = await extract_resume_highlights(resume_text, model)
    red_flags = await detect_red_flags(resume_text, resume_highlights, model)
    
    return {
        "red_flags": red_flags
    }