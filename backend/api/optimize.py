"""
Resume optimizer API (FastAPI) — ENHANCED VERSION

IMPROVEMENTS:
- Action verb diversity tracking (no repetition)
- Result phrases without numbers
- Technical depth indicators
- Skill progression across experience blocks
- Believability constraints for intern-level
- Bullet structure templates
- Cross-bullet coherence
- Industry-specific vocabulary
- Enhanced company context with progression
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
import httpx
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
# 💪 ACTION VERB MANAGEMENT - Diversity & Strength
# ============================================================

ACTION_VERBS = {
    "development": [
        "Architected", "Engineered", "Developed", "Built", "Implemented",
        "Constructed", "Designed", "Created", "Established", "Formulated"
    ],
    "research": [
        "Investigated", "Explored", "Analyzed", "Evaluated", "Validated",
        "Examined", "Studied", "Researched", "Assessed", "Characterized"
    ],
    "optimization": [
        "Optimized", "Enhanced", "Streamlined", "Accelerated", "Refined",
        "Improved", "Strengthened", "Advanced", "Elevated", "Augmented"
    ],
    "data_work": [
        "Processed", "Transformed", "Aggregated", "Curated", "Cleaned",
        "Structured", "Organized", "Consolidated", "Standardized", "Normalized"
    ],
    "ml_training": [
        "Trained", "Fine-tuned", "Calibrated", "Tuned", "Configured",
        "Parameterized", "Adapted", "Specialized", "Customized", "Fitted"
    ],
    "deployment": [
        "Deployed", "Launched", "Released", "Shipped", "Delivered",
        "Productionized", "Operationalized", "Integrated", "Provisioned", "Staged"
    ],
    "analysis": [
        "Analyzed", "Diagnosed", "Identified", "Discovered", "Uncovered",
        "Detected", "Recognized", "Profiled", "Mapped", "Quantified"
    ],
    "collaboration": [
        "Collaborated", "Partnered", "Coordinated", "Facilitated", "Supported",
        "Contributed", "Assisted", "Engaged", "Interfaced", "Liaised"
    ],
    "automation": [
        "Automated", "Systematized", "Scripted", "Programmed", "Orchestrated",
        "Scheduled", "Templated", "Codified", "Mechanized", "Streamlined"
    ],
    "documentation": [
        "Documented", "Recorded", "Cataloged", "Annotated", "Detailed",
        "Specified", "Outlined", "Summarized", "Reported", "Communicated"
    ]
}

# Session-level tracking to avoid verb repetition
_used_verbs_in_session: Set[str] = set()


def reset_verb_tracking():
    """Reset verb tracking for new resume optimization."""
    global _used_verbs_in_session
    _used_verbs_in_session.clear()


def get_diverse_verb(category: str, fallback: str = "Developed") -> str:
    """Get a verb that hasn't been used yet in this session."""
    global _used_verbs_in_session
    
    verbs = ACTION_VERBS.get(category, ACTION_VERBS["development"])
    available = [v for v in verbs if v.lower() not in _used_verbs_in_session]
    
    if not available:
        # Reset category if exhausted
        for v in verbs:
            _used_verbs_in_session.discard(v.lower())
        available = verbs
    
    chosen = random.choice(available)
    _used_verbs_in_session.add(chosen.lower())
    return chosen


def get_verb_categories_for_context(company_type: str) -> List[str]:
    """Get appropriate verb categories based on company type."""
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
        "attaining competitive benchmark results against established baselines"
    ],
    "efficiency": [
        "enabling faster experimentation and iteration cycles",
        "streamlining the end-to-end development workflow significantly",
        "reducing computational overhead while maintaining output quality",
        "accelerating model training and evaluation throughput",
        "improving overall resource utilization and pipeline efficiency"
    ],
    "quality": [
        "ensuring high-quality and reproducible model outputs",
        "maintaining rigorous quality standards throughout development",
        "achieving consistent and reliable experimental results",
        "delivering enterprise-grade code quality and documentation",
        "meeting stringent production readiness requirements"
    ],
    "scalability": [
        "supporting seamless scaling to larger datasets",
        "enabling distributed processing capabilities for production workloads",
        "facilitating efficient handling of increased data volumes",
        "ensuring system robustness under production-scale demands",
        "accommodating future growth and extensibility requirements"
    ],
    "insight": [
        "uncovering actionable insights from complex data patterns",
        "revealing previously hidden correlations and trends",
        "generating valuable intelligence for downstream applications",
        "providing data-driven recommendations for model improvements",
        "enabling informed decision-making through rigorous analysis"
    ],
    "collaboration": [
        "facilitating cross-functional collaboration and knowledge sharing",
        "enabling seamless integration with existing team workflows",
        "supporting reproducibility and handoff to other team members",
        "improving documentation and codebase maintainability",
        "establishing reusable components for future projects"
    ]
}

_used_result_phrases: Set[str] = set()


def reset_result_phrase_tracking():
    """Reset result phrase tracking."""
    global _used_result_phrases
    _used_result_phrases.clear()


def get_result_phrase(category: str) -> str:
    """Get a result phrase that hasn't been used."""
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
        "conducting systematic Ablation Studies to validate design choices"
    ],
    "dl_techniques": [
        "incorporating Batch Normalization and Dropout for regularization",
        "implementing Learning Rate Scheduling with warm restarts",
        "utilizing Gradient Clipping to stabilize training dynamics",
        "applying Transfer Learning with frozen backbone and fine-tuned heads",
        "employing Attention Mechanisms for improved feature representation",
        "implementing residual connections for gradient flow optimization"
    ],
    "data_techniques": [
        "implementing comprehensive Data Preprocessing pipelines with validation",
        "applying Dimensionality Reduction for efficient feature representation",
        "utilizing robust Outlier Detection and handling strategies",
        "implementing Missing Value Imputation with multiple strategies",
        "applying class balancing techniques for imbalanced datasets",
        "conducting thorough Exploratory Data Analysis for insight generation"
    ],
    "mlops_techniques": [
        "implementing Model Versioning with comprehensive experiment tracking",
        "establishing CI/CD pipelines for automated model validation",
        "utilizing containerization with Docker for reproducible deployments",
        "implementing Feature Store patterns for consistent feature serving",
        "establishing Model Monitoring dashboards for production oversight",
        "applying infrastructure-as-code practices for environment management"
    ],
    "evaluation_techniques": [
        "conducting Precision-Recall analysis for classification performance",
        "implementing comprehensive error analysis and failure mode identification",
        "utilizing statistical significance testing for model comparisons",
        "applying Confusion Matrix analysis for multi-class evaluation",
        "implementing custom evaluation metrics aligned with business objectives",
        "conducting systematic bias and fairness audits"
    ]
}


def get_technical_depth_phrase(category: str) -> str:
    """Get a technical depth phrase for credibility."""
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
        "complexity": "foundational components"
    },
    "mid": {
        "scope": ["developed", "implemented", "designed", "built"],
        "tasks": ["model development", "pipeline creation", "experiment execution", "performance analysis"],
        "autonomy": "with mentorship from team leads",
        "complexity": "core system components"
    },
    "late": {
        "scope": ["led", "architected", "spearheaded", "owned"],
        "tasks": ["end-to-end pipeline", "model optimization", "deployment preparation", "technical documentation"],
        "autonomy": "independently with periodic reviews",
        "complexity": "production-ready solutions"
    }
}


def get_progression_context(block_index: int, total_blocks: int = 4) -> Dict[str, Any]:
    """Get appropriate progression context based on position in experience."""
    if block_index == 0:
        return INTERN_PROGRESSION["late"]  # Most recent = most advanced
    elif block_index == total_blocks - 1:
        return INTERN_PROGRESSION["early"]  # Oldest = most basic
    else:
        return INTERN_PROGRESSION["mid"]


# ============================================================
# 🏭 BELIEVABILITY CONSTRAINTS
# ============================================================

BELIEVABILITY_RULES = {
    "intern_appropriate": [
        "Focus on learning, contribution, and growth",
        "Avoid claiming sole ownership of major systems",
        "Use collaborative language when appropriate",
        "Mention working with senior engineers or mentors",
        "Focus on specific components rather than entire systems"
    ],
    "scope_indicators": {
        "small": ["component", "module", "feature", "function", "utility"],
        "medium": ["pipeline", "workflow", "system", "service", "framework"],
        "large": ["platform", "infrastructure", "architecture", "ecosystem"]
    },
    "collaboration_phrases": [
        "in collaboration with senior engineers",
        "as part of a cross-functional team",
        "working closely with research mentors",
        "under guidance of technical leads",
        "contributing to team-wide initiatives"
    ]
}


def get_believability_phrase(scope: str = "medium") -> str:
    """Get a believability-enhancing phrase."""
    if random.random() < 0.3:  # 30% chance to add collaboration context
        return random.choice(BELIEVABILITY_RULES["collaboration_phrases"])
    return ""


# ============================================================
# 📝 BULLET STRUCTURE TEMPLATES
# ============================================================

BULLET_TEMPLATES = {
    "action_object_method_result": "{verb} {object} using {method}, {result}",
    "action_method_object_result": "{verb} {method}-based {object}, {result}",
    "action_object_result_method": "{verb} {object} {result} through {method}",
    "collaborative_action": "{verb} {object} in collaboration with {team}, {result}",
}


def get_bullet_template() -> str:
    """Get a random bullet template for variety."""
    templates = list(BULLET_TEMPLATES.values())
    return random.choice(templates)


# ============================================================
# 🔠 PROPER CAPITALIZATION MAP
# ============================================================

CAPITALIZATION_MAP: Dict[str, str] = {
    # Programming Languages
    "python": "Python", "java": "Java", "javascript": "JavaScript", "typescript": "TypeScript",
    "c++": "C++", "c#": "C#", "go": "Go", "rust": "Rust", "scala": "Scala", "kotlin": "Kotlin",
    "swift": "Swift", "ruby": "Ruby", "php": "PHP", "r": "R", "matlab": "MATLAB", "sql": "SQL",
    "bash": "Bash", "shell": "Shell", "perl": "Perl", "lua": "Lua", "julia": "Julia",

    # ML/AI Frameworks
    "pytorch": "PyTorch", "tensorflow": "TensorFlow", "keras": "Keras", "scikit-learn": "Scikit-learn",
    "sklearn": "Scikit-learn", "pandas": "Pandas", "numpy": "NumPy", "scipy": "SciPy",
    "matplotlib": "Matplotlib", "seaborn": "Seaborn", "plotly": "Plotly", "opencv": "OpenCV",
    "hugging face": "Hugging Face", "huggingface": "Hugging Face", "transformers": "Transformers",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
    "spacy": "SpaCy", "nltk": "NLTK", "gensim": "Gensim", "fastai": "FastAI", 
    "jax": "JAX", "flax": "Flax",

    # Cloud & DevOps
    "aws": "AWS", "gcp": "GCP", "azure": "Azure", "docker": "Docker", "kubernetes": "Kubernetes",
    "k8s": "K8s", "jenkins": "Jenkins", "terraform": "Terraform", "ansible": "Ansible",
    "circleci": "CircleCI", "github actions": "GitHub Actions", "gitlab": "GitLab",
    "ec2": "EC2", "s3": "S3", "lambda": "Lambda", "sagemaker": "SageMaker", "emr": "EMR",
    "bigquery": "BigQuery", "redshift": "Redshift", "snowflake": "Snowflake",

    # Databases
    "mysql": "MySQL", "postgresql": "PostgreSQL", "postgres": "PostgreSQL", "mongodb": "MongoDB",
    "redis": "Redis", "elasticsearch": "Elasticsearch", "cassandra": "Cassandra",
    "dynamodb": "DynamoDB", "sqlite": "SQLite", "oracle": "Oracle", "neo4j": "Neo4j",

    # Tools & Platforms
    "git": "Git", "github": "GitHub", "linux": "Linux", "unix": "Unix", "windows": "Windows",
    "jupyter": "Jupyter", "vscode": "VS Code", "intellij": "IntelliJ", "vim": "Vim",
    "mlflow": "MLflow", "wandb": "W&B", "weights & biases": "Weights & Biases",
    "airflow": "Airflow", "kafka": "Kafka", "spark": "Spark", "hadoop": "Hadoop",
    "databricks": "Databricks", "dbt": "Dbt", "prefect": "Prefect", "dagster": "Dagster",
    "grafana": "Grafana", "prometheus": "Prometheus", "datadog": "Datadog",

    # Web Frameworks
    "flask": "Flask", "django": "Django", "fastapi": "FastAPI", "express": "Express",
    "react": "React", "angular": "Angular", "vue": "Vue", "nextjs": "Next.js", "next.js": "Next.js",
    "nodejs": "Node.js", "node.js": "Node.js", "spring": "Spring", "rails": "Rails",

    # ML/AI Concepts
    "ml": "ML", "ai": "AI", "dl": "DL", "nlp": "NLP", "cv": "CV", "rl": "RL",
    "machine learning": "Machine Learning", "deep learning": "Deep Learning",
    "natural language processing": "Natural Language Processing",
    "computer vision": "Computer Vision", "reinforcement learning": "Reinforcement Learning",
    "neural network": "Neural Network", "neural networks": "Neural Networks",
    "cnn": "CNN", "rnn": "RNN", "lstm": "LSTM", "gru": "GRU", "gan": "GAN", "vae": "VAE",
    "bert": "BERT", "gpt": "GPT", "llm": "LLM", "llms": "LLMs",
    "transformer": "Transformer", "transformers": "Transformers",
    "attention mechanism": "Attention Mechanism", "self-attention": "Self-Attention",
    "fine-tuning": "Fine-Tuning", "transfer learning": "Transfer Learning",
    "feature engineering": "Feature Engineering", "hyperparameter tuning": "Hyperparameter Tuning",
    "cross-validation": "Cross-Validation", "gradient descent": "Gradient Descent",
    "backpropagation": "Backpropagation", "batch normalization": "Batch Normalization",
    "dropout": "Dropout", "regularization": "Regularization",
    "supervised learning": "Supervised Learning", "unsupervised learning": "Unsupervised Learning",
    "semi-supervised learning": "Semi-Supervised Learning",
    "classification": "Classification", "regression": "Regression", "clustering": "Clustering",
    "dimensionality reduction": "Dimensionality Reduction", "pca": "PCA", "t-sne": "T-SNE",
    "random forest": "Random Forest", "decision tree": "Decision Tree",
    "support vector machine": "Support Vector Machine", "svm": "SVM",
    "k-nearest neighbors": "K-Nearest Neighbors", "knn": "KNN",
    "naive bayes": "Naive Bayes", "logistic regression": "Logistic Regression",
    "linear regression": "Linear Regression", "gradient boosting": "Gradient Boosting",
    "ensemble methods": "Ensemble Methods", "bagging": "Bagging", "boosting": "Boosting",
    "automl": "AutoML", "mlops": "MLOps", "devops": "DevOps", "ci/cd": "CI/CD",
    "etl": "ETL", "elt": "ELT", "api": "API", "rest": "REST", "graphql": "GraphQL",
    "microservices": "Microservices", "serverless": "Serverless",
    "rag": "RAG", "retrieval-augmented generation": "Retrieval-Augmented Generation",
    "vector database": "Vector Database", "embedding": "Embedding", "embeddings": "Embeddings",
    "prompt engineering": "Prompt Engineering", "langchain": "LangChain",
    "llamaindex": "LlamaIndex", "openai": "OpenAI", "anthropic": "Anthropic",
    "chatgpt": "ChatGPT", "claude": "Claude", "gemini": "Gemini",

    # Data Science
    "data science": "Data Science", "data engineering": "Data Engineering",
    "data analysis": "Data Analysis", "data visualization": "Data Visualization",
    "data pipeline": "Data Pipeline", "data warehouse": "Data Warehouse",
    "data lake": "Data Lake", "data mining": "Data Mining",
    "big data": "Big Data", "analytics": "Analytics",
    "business intelligence": "Business Intelligence", "bi": "BI",
    "a/b testing": "A/B Testing", "ab testing": "A/B Testing", "a-b testing": "A/B Testing",
    "statistical analysis": "Statistical Analysis", "hypothesis testing": "Hypothesis Testing",

    # Company-core / systems topics
    "recommender systems": "Recommender Systems",
    "recommendation systems": "Recommendation Systems",
    "search & ranking": "Search & Ranking",
    "search and ranking": "Search And Ranking",
    "ranking": "Ranking",
    "experimentation": "Experimentation",
    "online experimentation": "Online Experimentation",
    "offline evaluation": "Offline Evaluation",
    "distributed systems": "Distributed Systems",
    "system design": "System Design",
    "large-scale data": "Large-Scale Data",
    "stream processing": "Stream Processing",
    "batch processing": "Batch Processing",
    "feature store": "Feature Store",
    "data modeling": "Data Modeling",

    # Other
    "agile": "Agile", "scrum": "Scrum", "jira": "Jira", "confluence": "Confluence",
    "slack": "Slack", "notion": "Notion", "trello": "Trello",
    "json": "JSON", "xml": "XML", "yaml": "YAML", "csv": "CSV",
    "html": "HTML", "css": "CSS", "sass": "SASS", "less": "LESS",
    "oauth": "OAuth", "jwt": "JWT", "ssl": "SSL", "tls": "TLS", "https": "HTTPS",
    "tcp": "TCP", "udp": "UDP", "http": "HTTP", "websocket": "WebSocket",
    "gpu": "GPU", "cpu": "CPU", "tpu": "TPU", "cuda": "CUDA", "cudnn": "CuDNN",
    "ios": "iOS", "macos": "MacOS",
}


def fix_capitalization(text: str) -> str:
    """Fix capitalization of technical terms while preserving sentence structure."""
    if not text:
        return text

    result = text
    sorted_terms = sorted(CAPITALIZATION_MAP.keys(), key=len, reverse=True)
    for term in sorted_terms:
        correct = CAPITALIZATION_MAP[term]
        pattern = rf"\b{re.escape(term)}\b"
        result = re.sub(pattern, correct, result, flags=re.IGNORECASE)
    return result


def _ensure_first_letter_capital(s: str) -> str:
    """Force first character to uppercase if it's a lowercase letter."""
    s = (s or "").strip()
    if not s:
        return s
    if s[0].isalpha() and s[0].islower():
        return s[0].upper() + s[1:]
    return s


def fix_skill_capitalization(skill: str) -> str:
    """Fix capitalization for a single skill term + enforce first-letter-capital."""
    skill = (skill or "").strip()
    if not skill:
        return ""

    skill_lower = skill.lower()
    if skill_lower in CAPITALIZATION_MAP:
        return CAPITALIZATION_MAP[skill_lower]

    out = fix_capitalization(skill)
    out = _ensure_first_letter_capital(out)
    return out


# ============================================================
# 🏢 ENHANCED Company Context with Progression & Vocabulary
# ============================================================

COMPANY_CONTEXTS = {
    "ayar labs": {
        "type": "industry_internship",
        "domain": "ML/AI in Semiconductor Industry",
        "context": "Silicon photonics company where ML/AI is applied across semiconductor workflow for yield prediction, process optimization, and quality assurance.",
        "technical_vocabulary": [
            "yield prediction", "process optimization", "wafer inspection",
            "defect classification", "signal integrity", "test data analysis",
            "equipment health monitoring", "production forecasting", "quality metrics"
        ],
        "ml_projects": [
            "ML-based yield prediction using manufacturing sensor data and process parameters",
            "Predictive maintenance system for semiconductor fabrication equipment health",
            "Automated defect classification pipeline for wafer inspection quality control",
            "Time-series forecasting model for production capacity planning optimization",
            "Feature engineering framework for high-dimensional semiconductor test data"
        ],
        "believable_tasks": [
            "Data Preprocessing", "Feature Engineering", "Model Training", "Hyperparameter Tuning",
            "Cross-Validation", "Model Evaluation", "Pipeline Development", "Data Visualization",
            "Statistical Analysis", "Experiment Tracking", "Model Deployment", "Batch Inference"
        ],
        "progression_tasks": {
            "early": ["data cleaning", "EDA", "baseline models", "documentation"],
            "mid": ["feature engineering", "model development", "pipeline creation"],
            "late": ["model optimization", "deployment prep", "production integration"]
        }
    },
    "indian institute of technology indore": {
        "type": "research_internship",
        "domain": "ML/AI Research",
        "context": "Premier research institution conducting cutting-edge ML/AI research with focus on novel architectures and optimization methods.",
        "technical_vocabulary": [
            "state-of-the-art", "baseline comparison", "ablation study",
            "benchmark evaluation", "novel architecture", "convergence analysis",
            "generalization capability", "theoretical foundation", "empirical validation"
        ],
        "ml_projects": [
            "Research on efficient Neural Network architectures for resource-constrained deployment",
            "Investigation of Transfer Learning techniques for cross-domain adaptation",
            "Development of novel Attention Mechanisms for improved sequence modeling",
            "Empirical study of optimization algorithms for Deep Learning convergence",
            "Research on model compression and knowledge distillation techniques"
        ],
        "believable_tasks": [
            "Literature Review", "Baseline Implementation", "Experiment Design", "Ablation Studies",
            "Benchmark Evaluation", "Result Analysis", "Technical Writing", "Paper Reproduction"
        ],
        "progression_tasks": {
            "early": ["literature survey", "baseline reproduction", "data preparation"],
            "mid": ["experiment design", "systematic evaluation", "ablation studies"],
            "late": ["novel contributions", "paper writing", "presentation"]
        }
    },
    "iit indore": {
        "type": "research_internship",
        "domain": "ML/AI Research",
        "context": "Premier research institution conducting ML/AI research.",
        "technical_vocabulary": [
            "state-of-the-art", "baseline comparison", "ablation study",
            "benchmark evaluation", "novel architecture"
        ],
        "ml_projects": [
            "Novel Neural Network architecture research",
            "Transfer Learning and Domain Adaptation studies",
            "Attention mechanisms and Transformer variants"
        ],
        "believable_tasks": [
            "Literature Review", "Baseline Implementation", "Experiment Design", "Ablation Studies"
        ],
        "progression_tasks": {
            "early": ["literature survey", "baseline reproduction"],
            "mid": ["experiment design", "evaluation"],
            "late": ["novel contributions", "documentation"]
        }
    },
    "national institute of technology jaipur": {
        "type": "research_internship",
        "domain": "Applied ML Research",
        "context": "Engineering institution focusing on practical ML applications to real-world problems.",
        "technical_vocabulary": [
            "practical application", "real-world dataset", "engineering constraints",
            "system integration", "performance benchmarking", "scalability analysis",
            "robustness testing", "deployment considerations"
        ],
        "ml_projects": [
            "Applied Machine Learning for time-series forecasting in engineering systems",
            "Development of anomaly detection methods for industrial monitoring applications",
            "Classification system for pattern recognition in sensor data streams",
            "Ensemble methods research for improved prediction reliability",
            "Feature selection study for high-dimensional engineering datasets"
        ],
        "believable_tasks": [
            "Data Collection", "Data Cleaning", "Exploratory Analysis", "Feature Extraction",
            "Model Selection", "Training Pipelines", "Error Analysis", "Documentation"
        ],
        "progression_tasks": {
            "early": ["data collection", "preprocessing", "EDA"],
            "mid": ["model implementation", "evaluation", "iteration"],
            "late": ["optimization", "documentation", "handoff"]
        }
    },
    "nit jaipur": {
        "type": "research_internship",
        "domain": "Applied ML Research",
        "context": "Engineering institution focusing on practical ML applications.",
        "technical_vocabulary": [
            "practical application", "real-world dataset", "engineering constraints"
        ],
        "ml_projects": [
            "Time-series analysis and forecasting models",
            "Anomaly detection for industrial systems",
            "Regression models for engineering tasks"
        ],
        "believable_tasks": [
            "Data Collection", "Data Cleaning", "Exploratory Analysis", "Feature Extraction"
        ],
        "progression_tasks": {
            "early": ["data collection", "preprocessing"],
            "mid": ["model implementation", "evaluation"],
            "late": ["optimization", "documentation"]
        }
    }
}


def get_company_context(company_name: str) -> Dict[str, Any]:
    name_lower = (company_name or "").lower().strip()
    for key, ctx in COMPANY_CONTEXTS.items():
        if key in name_lower or name_lower in key:
            return ctx
    return {
        "type": "internship",
        "domain": "ML/AI",
        "context": "Technical internship applying Machine Learning and Data Science.",
        "technical_vocabulary": ["model development", "data analysis", "pipeline"],
        "ml_projects": ["ML Model Development", "Data Pipeline Creation"],
        "believable_tasks": ["Model Development", "Data Analysis", "Testing", "Documentation"],
        "progression_tasks": {
            "early": ["learning", "documentation"],
            "mid": ["implementation", "testing"],
            "late": ["optimization", "delivery"]
        }
    }


# ============================================================
# 🏢 Company Core Expectations (target employer)
# ============================================================

COMPANY_CORE_FALLBACKS: Dict[str, Dict[str, Any]] = {
    "netflix": {
        "core_areas": ["Recommender Systems", "Search & Ranking", "Experimentation", "Large-Scale Data"],
        "core_keywords": ["Recommender Systems", "Search & Ranking", "A/B Testing", "Spark", "Scala", "Data Pipelines", "Offline Evaluation"],
    },
    "google": {
        "core_areas": ["System Design", "Scalability", "Distributed Systems", "Experimentation"],
        "core_keywords": ["System Design", "Distributed Systems", "Scalability", "A/B Testing", "Data Structures", "Algorithms"],
    },
    "meta": {
        "core_areas": ["Experimentation", "Ranking", "Large-Scale Data", "Distributed Systems"],
        "core_keywords": ["A/B Testing", "Ranking", "Distributed Systems", "Spark", "Data Pipelines", "Online Experimentation"],
    },
    "amazon": {
        "core_areas": ["Scalability", "Distributed Systems", "Operational Excellence", "Data-driven Decisions"],
        "core_keywords": ["Distributed Systems", "Scalability", "System Design", "Monitoring", "Data Pipelines"],
    },
    "microsoft": {
        "core_areas": ["Cloud Computing", "Distributed Systems", "AI/ML", "Product Development"],
        "core_keywords": ["Azure", "Distributed Systems", "Machine Learning", "System Design", "Cloud Architecture"],
    },
    "apple": {
        "core_areas": ["Privacy-Preserving ML", "On-Device ML", "User Experience", "System Optimization"],
        "core_keywords": ["On-Device ML", "Privacy", "Core ML", "System Optimization", "User Experience"],
    },
}

_company_core_cache: Dict[str, Dict[str, Any]] = {}


async def extract_company_core_requirements(
    target_company: str,
    target_role: str,
    jd_text: str,
) -> Dict[str, Any]:
    """Uses ChatGPT API to infer company expectations NOT always in the JD."""
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

    json_schema = (
        '{\n'
        '  "core_areas": ["..."],\n'
        '  "core_keywords": ["..."],\n'
        '  "notes": "1-2 sentence justification"\n'
        '}'
    )

    prompt = (
        "You are building an ATS-focused resume optimizer.\n"
        "Infer the KEY COMPANY EXPECTATIONS for the target employer that are often NOT explicitly stated in the JD.\n\n"
        f"Target Company: {target_company}\n"
        f"Target Role: {target_role}\n\n"
        "Rules:\n"
        "- Return STRICT JSON ONLY in this format:\n"
        f"{json_schema}\n"
        "- core_areas: 3-6 high-level areas (2-4 words each).\n"
        "- core_keywords: 8-14 resume-friendly skills/topics/tools commonly expected.\n"
        "- Do NOT invent proprietary internal tool names.\n"
        "- Keep tokens short (1-4 words).\n\n"
        "JD (for context only; do not restrict to it):\n"
        f"{jd_text[:2500]}"
    )

    try:
        data = await gpt_json(prompt, temperature=0.0)
        core_areas = data.get("core_areas", []) or []
        core_kw = data.get("core_keywords", []) or []
        notes = (data.get("notes", "") or "").strip()

        def _clean_list(lst: Iterable[Any]) -> List[str]:
            out_list: List[str] = []
            seen: Set[str] = set()
            for x in lst:
                s = re.sub(r"[^\w\-\+\.#\/ \(\)&]", "", str(x)).strip()
                s = re.sub(r"\s+", " ", s)
                if not s:
                    continue
                s = fix_skill_capitalization(s)
                key = s.lower()
                if key not in seen:
                    seen.add(key)
                    out_list.append(s)
            return out_list

        core_areas = _clean_list(core_areas)[:8]
        core_kw = _clean_list(core_kw)[:18]

        if not core_kw:
            fb = COMPANY_CORE_FALLBACKS.get(ckey, {})
            core_areas = fb.get("core_areas", core_areas) or core_areas
            core_kw = fb.get("core_keywords", core_kw) or core_kw

        out = {"core_areas": core_areas, "core_keywords": core_kw, "notes": notes}
        _company_core_cache[cache_key] = out
        log_event(f"🏢 [COMPANY CORE] {target_company} areas={len(core_areas)} keywords={len(core_kw)}")
        return out

    except Exception as e:
        log_event(f"⚠️ [COMPANY CORE] Failed: {e}")
        fb = COMPANY_CORE_FALLBACKS.get(ckey, {})
        out = {
            "core_areas": fb.get("core_areas", ["System Design", "Experimentation", "Distributed Systems"]),
            "core_keywords": fb.get("core_keywords", ["System Design", "Distributed Systems", "A/B Testing", "Data Pipelines", "Scalability"]),
            "notes": "Fallback company-core profile used due to API failure.",
        }
        _company_core_cache[cache_key] = out
        return out


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

_FALLBACK_TAG_RE = re.compile(r"^\[LOCAL-FALLBACK:[^\]]+\]\s*", re.IGNORECASE)


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
# 🚫 No-quantification guardrails
# ============================================================

_NUMBER_WORDS_RE = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|dozen|couple|hundred|thousand|million|billion|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
    re.IGNORECASE,
)

_QUANT_UNITS_RE = re.compile(
    r"\b(day|days|week|weeks|month|months|year|years|quarter|quarters|"
    r"hrs?|hours?|mins?|minutes?|seconds?|sec|sprint|sprints|percent|percentage|%)\b",
    re.IGNORECASE,
)

_QUANT_SYMBOLS_RE = re.compile(r"[\d%$£€¥]")
_MULTIPLIER_RE = re.compile(r"\b\d+\s*[xX]\b")


def _has_any_quantification(text: str) -> bool:
    if not text:
        return False
    t = strip_all_macros_keep_text(text)
    return bool(_QUANT_SYMBOLS_RE.search(t) or _MULTIPLIER_RE.search(t) or
                _NUMBER_WORDS_RE.search(t) or _QUANT_UNITS_RE.search(t))


def _strip_quantification(text: str) -> str:
    if not text:
        return ""
    t = strip_all_macros_keep_text(text)
    t = re.sub(r"[$£€¥]\s*\d+(\.\d+)?", "", t)
    t = re.sub(r"\b\d+(\.\d+)?\b", "", t)
    t = t.replace("%", "")
    t = re.sub(r"\b[xX]\b", "", t)
    t = _NUMBER_WORDS_RE.sub("", t)
    t = _QUANT_UNITS_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.;:])", r"\1", t).strip()
    if t and t[-1] not in ".!?":
        t += "."
    return t


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
    """Truncate if too long; otherwise leave."""
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
# 🧠 JD Analysis
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

Be comprehensive. Extract EVERY technical term that belongs on a resume.
"""

    try:
        data = await gpt_json(prompt, temperature=0.0)
        must_have = [fix_skill_capitalization(k) for k in data.get("must_have", [])]
        should_have = [fix_skill_capitalization(k) for k in data.get("should_have", [])]
        nice_to_have = [fix_skill_capitalization(k) for k in data.get("nice_to_have", [])]
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
            "must_have": must_have,
            "should_have": should_have,
            "nice_to_have": nice_to_have,
            "all_keywords": all_keywords,
            "responsibilities": responsibilities,
            "domain": domain,
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
        out: List[str] = []
        seen: Set[str] = set()
        for c in courses:
            c = fix_capitalization(re.sub(r"\s+", " ", str(c)).strip())
            c = _ensure_first_letter_capital(c)
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
        c = fix_capitalization(re.sub(r"\s+", " ", str(c)).strip())
        c = _ensure_first_letter_capital(c)
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
        s = fix_skill_capitalization(str(s).strip())
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
# 📝 ENHANCED Bullet Generation with All Improvements
# ============================================================

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
    total_blocks: int = 4,
    num_bullets: int = 3,
) -> Tuple[List[str], Set[str]]:
    """
    Generate ENHANCED resume bullets with:
    - Action verb diversity
    - Technical depth indicators
    - Result phrases without numbers
    - Skill progression
    - Believability constraints
    """
    exp_context = get_company_context(experience_company)
    progression = get_progression_context(block_index, total_blocks)
    
    # Get diverse verb categories for this company type
    verb_categories = get_verb_categories_for_context(exp_context.get("type", "internship"))
    suggested_verbs = [get_diverse_verb(cat) for cat in verb_categories[:3]]
    
    # Get technical depth phrase
    tech_depth = get_technical_depth_phrase("ml_techniques")
    
    # Get result phrase
    result_phrase = get_result_phrase("performance")
    
    # Get believability phrase
    believability = get_believability_phrase()

    available_must = [fix_skill_capitalization(k) for k in must_use_keywords if k.lower() not in used_keywords][:6]
    available_should = [fix_skill_capitalization(k) for k in should_use_keywords if k.lower() not in used_keywords][:4]

    core_pool = [fix_skill_capitalization(k) for k in (company_core_keywords or [])]
    core_pool = [k for k in core_pool if k.lower() not in used_keywords][:6]

    keywords_for_block = core_pool[:3] + available_must + available_should
    keywords_for_block = [k for k in keywords_for_block if k]

    keywords_str = ", ".join(keywords_for_block[:10]) if keywords_for_block else "Python, Machine Learning"
    resp_str = "; ".join(responsibilities[:3]) if responsibilities else "Model Development; Evaluation; Deployment"

    core_focus_str = ", ".join(core_pool[:4]) if core_pool else ""
    core_rule = (
        f"- Naturally include target-company core areas: {core_focus_str}\n"
        if core_focus_str else ""
    )

    # Get company-specific vocabulary
    tech_vocab = exp_context.get("technical_vocabulary", [])
    vocab_str = ", ".join(tech_vocab[:5]) if tech_vocab else ""

    prompt = f"""Write EXACTLY {num_bullets} HIGHLY CREDIBLE resume bullet points for an INTERN at "{experience_company}",
tailored for applying to "{target_company}" ({target_role}).

🎯 CRITICAL REQUIREMENTS FOR CREDIBILITY:

1. LENGTH: Each bullet MUST be EXACTLY 18-22 words (count carefully!)

2. SKILLS TO USE: Each bullet MUST naturally integrate 2-3 skills from: {keywords_str}

3. ACTION VERBS: Use these strong, varied verbs (one per bullet, no repetition):
   - Suggested: {', '.join(suggested_verbs)}
   - Avoid: "Worked on", "Helped with", "Was responsible for"

4. TECHNICAL DEPTH: Show HOW you did things, not just WHAT:
   - Example technique reference: "{tech_depth}"
   - Include specific methodologies, not vague claims

5. RESULT LANGUAGE (NO NUMBERS): End with impact phrases like:
   - "{result_phrase}"
   - Avoid any digits, percentages, or quantified metrics

6. BELIEVABILITY FOR INTERN LEVEL:
   - Scope: {progression['scope'][0]} / {progression['scope'][1]} level work
   - Autonomy: {progression['autonomy']}
   - Task complexity: {progression['complexity']}
   {f'- Collaboration: {believability}' if believability else ''}

7. DOMAIN VOCABULARY: Use industry terms naturally:
   - Company domain: {exp_context['domain']}
   - Relevant terms: {vocab_str}

{core_rule}

EXPERIENCE CONTEXT:
- Company Type: {exp_context['type']}
- Domain: {exp_context['domain']}

JOB RESPONSIBILITIES TO ALIGN WITH:
{resp_str}

GOOD BULLET EXAMPLES (18-22 words, technical depth, no numbers):
- "Engineered PyTorch classification pipeline with stratified Cross-Validation and Hyperparameter Tuning, achieving robust generalization across imbalanced semiconductor datasets."
- "Developed automated Feature Engineering framework using Pandas and Scikit-learn, enabling consistent data transformation for downstream Machine Learning models."
- "Implemented TensorFlow model training workflow with experiment tracking via MLflow, facilitating reproducible research and systematic performance comparison."

BAD BULLET EXAMPLES (avoid these):
- "Worked on machine learning projects." (too vague, no depth)
- "Improved model accuracy by 25%." (has numbers)
- "Responsible for data analysis tasks." (passive, no action verb)

Return STRICT JSON with EXACTLY {num_bullets} bullets:
{{"bullets": ["bullet1 (18-22 words)", "bullet2 (18-22 words)", "bullet3 (18-22 words)"]}}
"""

    try:
        data = await gpt_json(prompt, temperature=0.25)
        bullets = data.get("bullets", []) or []

        cleaned: List[str] = []
        newly_used: Set[str] = set()

        for b in bullets[:num_bullets]:
            b = str(b).strip()
            b = fix_capitalization(b)
            if _has_any_quantification(b):
                b = _strip_quantification(b)

            b = adjust_bullet_length(b)
            b = latex_escape_text(b)

            if b:
                cleaned.append(b)
                for kw in keywords_for_block:
                    if kw.lower() in b.lower():
                        newly_used.add(kw.lower())

        # Enhanced fallback bullets with proper structure
        while len(cleaned) < num_bullets:
            idx = len(cleaned)
            verb = get_diverse_verb(verb_categories[idx % len(verb_categories)])
            kw1 = fix_skill_capitalization(keywords_for_block[idx % max(1, len(keywords_for_block))]) if keywords_for_block else "Python"
            kw2 = fix_skill_capitalization(keywords_for_block[(idx + 1) % max(1, len(keywords_for_block))]) if len(keywords_for_block) > 1 else "Machine Learning"
            
            fallback = (
                f"{verb} {kw1}-based analytical workflow with {kw2} integration, "
                f"enabling systematic evaluation and improved reproducibility for research deliverables."
            )
            fallback = fix_capitalization(fallback)
            cleaned.append(latex_escape_text(fallback))
            newly_used.add(kw1.lower())
            newly_used.add(kw2.lower())

        return cleaned[:num_bullets], newly_used

    except Exception as e:
        log_event(f"⚠️ [BULLETS] Generation failed for {experience_company}: {e}")
        fallback = "Contributed to Machine Learning model development and Data Pipeline implementation supporting research objectives with reliable engineering practices."
        return [latex_escape_text(fallback)] * num_bullets, set()


# ============================================================
# 🔄 Experience Rewriter
# ============================================================

async def rewrite_experience_with_skill_alignment(
    tex_content: str,
    jd_text: str,
    jd_info: Dict[str, Any],
    target_company: str,
    target_role: str,
    company_core_keywords: List[str],
) -> Tuple[str, Set[str]]:
    # Reset tracking for new resume
    reset_verb_tracking()
    reset_result_phrase_tracking()
    
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

            # Candidate experience company
            if block_index == 0:
                exp_company = "Ayar Labs"
            elif block_index == 1:
                exp_company = "Indian Institute of Technology Indore"
            elif block_index == 2:
                exp_company = "National Institute of Technology Jaipur"
            else:
                exp_company = "Indian Institute of Technology Indore"

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
                total_blocks=num_blocks,
                num_bullets=3,
            )

            exp_used_keywords.update(newly_used)

            new_block = s_tag + "\n"
            for bullet in new_bullets:
                new_block += f"    \\resumeItem{{{bullet}}}\n"
            new_block += "  " + e_tag

            rebuilt.append(new_block)
            block_index += 1
            i = b + len(e_tag)

        out.append("".join(rebuilt))
        pos = m.end()

    out.append(tex_content[pos:])

    must_covered = len([k for k in must_have if k.lower() in exp_used_keywords])
    log_event(f"📊 [EXP COVERAGE] Must-have: {must_covered}/{len(must_have)}")

    return "".join(out), exp_used_keywords


# ============================================================
# ✨ Humanize using API
# ============================================================

async def humanize_experience_bullets(tex_content: str) -> str:
    log_event("🟨 [HUMANIZE] Starting via superhuman API")

    async def _humanize_block(block: str) -> str:
        items = find_resume_items(block)
        if not items:
            return block

        plain_texts: List[str] = []
        for (_s, open_b, close_b, _e) in items:
            inner = block[open_b + 1:close_b]
            txt = strip_all_macros_keep_text(inner)
            plain_texts.append(txt[:1000].strip())

        async def rewrite_one(text: str, idx: int) -> str:
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
                        rew = _FALLBACK_TAG_RE.sub("", rew).replace("\n", " ").strip()
                        if rew:
                            rew = fix_capitalization(rew)
                            if _has_any_quantification(rew):
                                rew = _strip_quantification(rew)
                            rew = adjust_bullet_length(rew)
                            return latex_escape_text(rew)
                except Exception:
                    await asyncio.sleep(0.4)

            text2 = fix_capitalization(text)
            if _has_any_quantification(text2):
                text2 = _strip_quantification(text2)
            return latex_escape_text(text2)

        sem = asyncio.Semaphore(5)

        async def lim(i: int, t: str) -> str:
            async with sem:
                return await rewrite_one(t, i)

        humanized = await asyncio.gather(*[lim(i, t) for i, t in enumerate(plain_texts, 1)])
        return replace_resume_items(block, humanized)

    for sec_name in ["Experience", "Projects"]:
        pat = section_rx(sec_name)
        out: List[str] = []
        pos = 0
        for m in pat.finditer(tex_content):
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
                block = section[a:b]
                block = await _humanize_block(block)
                rebuilt.append(block)
                rebuilt.append(section[b:b + len(e_tag)])
                i = b + len(e_tag)
            out.append("".join(rebuilt))
            pos = m.end()
        out.append(tex_content[pos:])
        tex_content = "".join(out)

    return tex_content


# ============================================================
# 📄 PDF Helpers
# ============================================================

def _pdf_page_count(pdf_bytes: Optional[bytes]) -> int:
    if not pdf_bytes:
        return 0
    return len(re.findall(rb"/Type\s*/Page\b", pdf_bytes))


_EDU_SPLIT_ANCHOR = re.compile(r"(%-----------EDUCATION-----------)|\\section\*?\{\s*Education\s*\}", re.IGNORECASE)


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


def remove_last_bullet_from_sections(tex_content: str, sections: Tuple[str, ...] = ("Projects", "Experience")) -> Tuple[str, bool]:
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
    return {"ratio": len(present) / total, "present": sorted(present), "missing": sorted(missing), "total": total}


# ============================================================
# 🚀 Main Optimizer
# ============================================================

async def optimize_resume(
    base_tex: str,
    jd_text: str,
    target_company: str,
    target_role: str,
    extra_keywords: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    log_event("🟨 [OPTIMIZE] Starting ENHANCED optimization with credibility features")

    # 1) JD keywords
    jd_info = await extract_keywords_with_priority(jd_text)

    # 2) Company-core expectations
    company_core = await extract_company_core_requirements(target_company, target_role, jd_text)
    core_keywords = [fix_skill_capitalization(k) for k in (company_core.get("core_keywords", []) or [])]

    jd_info["company_core_keywords"] = core_keywords
    all_keywords = list(jd_info.get("all_keywords", []) or [])
    for k in core_keywords:
        if k and k.lower() not in [x.lower() for x in all_keywords]:
            all_keywords.append(k)

    # 3) Extra keywords
    extra_list: List[str] = []
    if extra_keywords:
        for token in re.split(r"[,\n;]+", extra_keywords):
            t = fix_skill_capitalization(token.strip())
            if t and t.lower() not in [x.lower() for x in extra_list]:
                extra_list.append(t)
    if extra_list:
        jd_info["extra_keywords"] = extra_list
        for k in extra_list:
            if k.lower() not in [x.lower() for x in all_keywords]:
                all_keywords.append(k)
    else:
        jd_info["extra_keywords"] = []

    log_event(f"📊 [KEYWORDS] JD={len(jd_info.get('all_keywords', []))} + CORE={len(core_keywords)} + EXTRA={len(extra_list)} → TOTAL={len(all_keywords)}")

    # 4) Coursework
    courses = await extract_coursework_gpt(jd_text, max_courses=24)

    # 5) Split preamble/body
    preamble, body = _split_preamble_body(base_tex)

    # 6) Coursework replace
    body = replace_relevant_coursework_distinct(body, courses, max_per_line=8)
    log_event("✅ [COURSEWORK] Updated")

    # 7) Rewrite experience with enhanced bullet generation
    body, exp_used_keywords = await rewrite_experience_with_skill_alignment(
        body,
        jd_text,
        jd_info,
        target_company=target_company,
        target_role=target_role,
        company_core_keywords=core_keywords,
    )
    log_event(f"✅ [EXPERIENCE] {len(exp_used_keywords)} keywords used with enhanced credibility")

    # 8) Skills section
    skills_list: List[str] = [fix_skill_capitalization(k) for k in exp_used_keywords]

    for kw in jd_info.get("must_have", []) or []:
        kw_fixed = fix_skill_capitalization(kw)
        if kw_fixed.lower() not in [s.lower() for s in skills_list]:
            skills_list.append(kw_fixed)

    for kw in jd_info.get("nice_to_have", []) or []:
        kw_fixed = fix_skill_capitalization(kw)
        if kw_fixed.lower() not in [s.lower() for s in skills_list]:
            skills_list.append(kw_fixed)

    for kw in core_keywords:
        kw_fixed = fix_skill_capitalization(kw)
        if kw_fixed.lower() not in [s.lower() for s in skills_list]:
            skills_list.append(kw_fixed)

    for kw in extra_list:
        kw_fixed = fix_skill_capitalization(kw)
        if kw_fixed.lower() not in [s.lower() for s in skills_list]:
            skills_list.append(kw_fixed)

    body = await replace_skills_section(body, skills_list)
    log_event(f"✅ [SKILLS] {len(skills_list)} skills")

    # 9) Merge back
    final_tex = _merge_tex(preamble, body)

    # 10) Coverage
    coverage = compute_coverage(final_tex, all_keywords)
    log_event(f"📊 [COVERAGE] {coverage['ratio']:.1%}")

    return final_tex, {
        "jd_info": jd_info,
        "company_core": company_core,
        "all_keywords": all_keywords,
        "coverage": coverage,
        "exp_used_keywords": list(exp_used_keywords),
        "skills_list": skills_list,
    }


# ============================================================
# 🚀 API Endpoint
# ============================================================

@router.post("/")
@router.post("/run")
@router.post("/submit")
async def optimize_endpoint(
    jd_text: str = Form(...),
    use_humanize: bool = Form(True),
    base_resume_tex: Optional[UploadFile] = File(None),
    extra_keywords: Optional[str] = Form(None),
):
    try:
        use_humanize = True if getattr(config, "HUMANIZE_DEFAULT_ON", True) else use_humanize

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
            raw_tex,
            jd_text,
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

        # HUMANIZED PDF
        pdf_bytes_humanized: Optional[bytes] = None
        humanized_tex: Optional[str] = None
        did_humanize = False
        coverage = info["coverage"]

        if use_humanize:
            log_event("🟨 [HUMANIZE] Starting via API")
            did_humanize = True

            humanized_tex = await humanize_experience_bullets(optimized_tex_final)

            humanized_rendered = render_final_tex(humanized_tex)
            try:
                pdf_bytes_humanized = compile_latex_safely(humanized_rendered)
            except Exception as e:
                log_event(f"⚠️ [COMPILE] Humanized failed: {e}")
                pdf_bytes_humanized = None

            if pdf_bytes_humanized:
                h_pages = _pdf_page_count(pdf_bytes_humanized)
                h_trim = 0
                while h_pages > 1 and h_trim < MAX_TRIMS:
                    humanized_tex, removed = remove_one_achievement_bullet(humanized_tex)
                    if not removed:
                        humanized_tex, removed = remove_last_bullet_from_sections(humanized_tex, ("Projects", "Experience"))
                    if not removed:
                        break
                    h_trim += 1
                    humanized_rendered = render_final_tex(humanized_tex)
                    pdf_bytes_humanized = compile_latex_safely(humanized_rendered)
                    h_pages = _pdf_page_count(pdf_bytes_humanized)

        # Save files
        paths = build_output_paths(target_company, target_role)
        opt_path = paths["optimized"]
        hum_path = paths["humanized"]
        saved_paths: List[str] = []

        if pdf_bytes_optimized:
            opt_path.parent.mkdir(parents=True, exist_ok=True)
            opt_path.write_bytes(pdf_bytes_optimized)
            saved_paths.append(str(opt_path))
            log_event(f"💾 [SAVE] Optimized → {opt_path}")

        if did_humanize and pdf_bytes_humanized:
            hum_path.parent.mkdir(parents=True, exist_ok=True)
            hum_path.write_bytes(pdf_bytes_humanized)
            saved_paths.append(str(hum_path))
            log_event(f"💾 [SAVE] Humanized → {hum_path}")

        return JSONResponse({
            "company_name": target_company,
            "role": target_role,
            "eligibility": {
                "score": coverage["ratio"],
                "present": coverage["present"],
                "missing": coverage["missing"],
                "total": coverage["total"],
                "verdict": "Strong fit" if coverage["ratio"] >= 0.7 else "Good fit" if coverage["ratio"] >= 0.5 else "Needs improvement",
            },
            "company_core": info.get("company_core", {}),
            "optimized": {
                "tex": render_final_tex(optimized_tex_final),
                "pdf_b64": base64.b64encode(pdf_bytes_optimized or b"").decode("ascii"),
                "filename": str(opt_path) if pdf_bytes_optimized else "",
            },
            "humanized": {
                "tex": render_final_tex(humanized_tex) if (did_humanize and humanized_tex) else "",
                "pdf_b64": base64.b64encode(pdf_bytes_humanized or b"").decode("ascii") if (did_humanize and pdf_bytes_humanized) else "",
                "filename": str(hum_path) if (did_humanize and pdf_bytes_humanized) else "",
            },
            "tex_string": render_final_tex(optimized_tex_final),
            "pdf_base64": base64.b64encode(pdf_bytes_optimized or b"").decode("ascii"),
            "pdf_base64_humanized": base64.b64encode(pdf_bytes_humanized or b"").decode("ascii") if (did_humanize and pdf_bytes_humanized) else None,
            "saved_paths": saved_paths,
            "coverage_ratio": coverage["ratio"],
            "coverage_present": coverage["present"],
            "coverage_missing": coverage["missing"],
            "coverage_history": [],
            "did_humanize": did_humanize,
            "extra_keywords": info.get("jd_info", {}).get("extra_keywords", []),
            "skills_list": info.get("skills_list", []),
        })

    except Exception as e:
        log_event(f"💥 [PIPELINE] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))