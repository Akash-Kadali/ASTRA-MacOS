"""
Resume optimizer API (FastAPI)

NEW IN v1.0.0:
- ✨ TECHNOLOGY SPECIFICITY: Maps generic terms to specific implementations
  (LLM → Llama 3.1/GPT-4, NLP → BERT/RoBERTa, etc.)
- ✨ FRESHER-APPROPRIATE SCOPE: Believable intern/junior achievements
- ✨ REALISTIC METRICS: Intern-level numbers and project scales
- ✨ TOOL CHAIN DETAILS: Specific tool versions and configurations
- ✨ CONTEXTUAL TECHNOLOGY SELECTION: Picks technologies that fit together
- ✨ PROGRESSIVE COMPLEXITY: Earlier internships = simpler tech, later = advanced
- ✨ IMPLEMENTATION DETAILS: Shows HOW you used the technology, not just WHAT
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

TECHNOLOGY_SPECIFICITY_MAP = {
    # LLMs
    "llm": ["Llama 3.1", "GPT-4", "Claude Sonnet", "Mistral", "Gemini Pro", "GPT-3.5"],
    "large language model": ["Llama", "GPT-4", "Claude", "Mistral", "Gemini"],
    "language model": ["BERT", "RoBERTa", "T5", "DistilBERT", "ELECTRA"],
    "transformer": ["BERT", "GPT-2", "T5", "BART", "RoBERTa", "DeBERTa"],
    
    # NLP
    "nlp": ["BERT", "RoBERTa", "spaCy", "NLTK", "Sentence-BERT"],
    "natural language processing": ["BERT for token classification", "RoBERTa for sentiment analysis", "T5 for text generation"],
    "text classification": ["BERT with CLS token pooling", "RoBERTa with attention pooling", "DistilBERT"],
    "sentiment analysis": ["VADER", "RoBERTa fine-tuned", "TextBlob"],
    "named entity recognition": ["spaCy NER", "BERT-NER with CRF", "Flair embeddings"],
    
    # Agentic AI
    "agentic ai": ["LangChain with ReAct", "AutoGPT", "LlamaIndex", "CrewAI"],
    "ai agent": ["LangChain agent", "function-calling with GPT-4", "ReAct prompting"],
    "multi-agent": ["LangGraph", "CrewAI", "custom agent orchestration"],
    "agent": ["LangChain Agent", "OpenAI function calling", "custom tool-using agent"],
    
    # ML Frameworks - NO VERSIONS
    "pytorch": ["PyTorch with CUDA", "PyTorch Lightning", "torch.nn modules"],
    "tensorflow": ["TensorFlow with Keras", "tf.data pipelines", "TensorBoard"],
    "keras": ["Keras with mixed precision", "custom Keras layers", "Keras callbacks"],
    "scikit-learn": ["scikit-learn Pipeline", "GridSearchCV", "custom transformers"],
    
    # Deep Learning
    "deep learning": ["CNN with ResNet backbone", "LSTM with attention", "Transformer encoder-decoder"],
    "neural network": ["MLP with dropout", "ResNet transfer learning", "custom PyTorch architecture"],
    "cnn": ["ResNet pretrained", "EfficientNet", "custom CNN"],
    "rnn": ["bidirectional LSTM", "GRU with attention", "RNN with gradient clipping"],
    "lstm": ["bidirectional LSTM", "LSTM with peephole", "stacked LSTM"],
    
    # Cloud & Infrastructure - NO VERSIONS
    "aws": ["AWS SageMaker", "Lambda", "S3", "EC2"],
    "kubernetes": ["Kubernetes with Helm", "K8s CronJobs", "KServe"],
    "docker": ["Docker multi-stage builds", "Docker Compose", "custom Dockerfile"],
    "cloud": ["AWS SageMaker", "GCP Vertex AI", "Azure ML"],
    
    # Data Engineering
    "data pipeline": ["Airflow DAGs", "Prefect workflows", "ETL with Pandas"],
    "etl": ["Apache Airflow", "dbt", "Spark ETL"],
    "data processing": ["Pandas with Dask", "PySpark", "NumPy vectorization"],
    "feature engineering": ["scikit-learn FeatureUnion", "custom extractors", "automated selection"],
    
    # MLOps
    "mlops": ["MLflow tracking", "DVC versioning", "Kubeflow Pipelines"],
    "model deployment": ["FastAPI with Docker", "TorchServe", "SageMaker endpoints"],
    "ci/cd": ["GitHub Actions", "Jenkins pipeline", "GitLab CI"],
    "monitoring": ["Prometheus with Grafana", "model drift detection", "CloudWatch"],
    
    # Databases - NO VERSIONS
    "database": ["PostgreSQL with pgvector", "MongoDB", "Redis"],
    "sql": ["PostgreSQL", "MySQL", "SQLite"],
    "nosql": ["MongoDB", "DynamoDB", "Cassandra"],
    "vector database": ["Pinecone", "Weaviate", "ChromaDB"],
    
    # Tools
    "git": ["Git with feature branching", "GitHub PR workflows", "Git hooks"],
    "jupyter": ["Jupyter with nbconvert", "JupyterLab", "papermill"],
    "wandb": ["Weights & Biases tracking", "W&B Sweeps"],
    "tensorboard": ["TensorBoard profiling", "custom plugins", "embeddings visualization"],
}

# NO VERSIONS in ecosystems either
TECHNOLOGY_ECOSYSTEMS = {
    "pytorch_stack": ["PyTorch", "Hugging Face Transformers", "WandB", "Docker"],
    "tensorflow_stack": ["TensorFlow", "Keras", "TensorBoard", "TFX"],
    "nlp_stack": ["BERT", "spaCy", "Hugging Face", "NLTK"],
    "llm_stack": ["LangChain", "Llama", "ChromaDB", "FastAPI"],
    "mlops_stack": ["MLflow", "Docker", "Kubernetes", "Airflow"],
    "aws_stack": ["SageMaker", "Lambda", "S3", "ECR"],
    "data_stack": ["Pandas", "NumPy", "Matplotlib", "scikit-learn"],
}

_used_specific_technologies: Set[str] = set()


def reset_technology_tracking():
    global _used_specific_technologies
    _used_specific_technologies.clear()


async def get_specific_technology(
    generic_term: str,
    context: str = "",
    already_used: Optional[Set[str]] = None,
    block_index: int = 0,
) -> str:
    """
    Convert generic technology term to specific implementation.
    
    Args:
        generic_term: Generic term like "LLM", "NLP", "AWS"
        context: Context about what the technology is being used for
        already_used: Set of technologies already mentioned (for dedup)
        block_index: Which experience block (0=latest, 3=earliest)
        
    Returns:
        Specific technology string like "Llama 3.1 70B for text generation"
    """
    global _used_specific_technologies
    
    if already_used is None:
        already_used = _used_specific_technologies
    
    generic_lower = generic_term.lower().strip()
    
    # Get candidates from the mapping
    candidates = TECHNOLOGY_SPECIFICITY_MAP.get(generic_lower, [])
    
    if not candidates:
        # If no mapping, return the original with proper capitalization
        return await fix_capitalization_gpt(generic_term)
    
    # Filter out already used technologies
    available = [c for c in candidates if c.lower() not in already_used]
    
    if not available:
        # If all used, just pick randomly from all candidates
        available = candidates
    
    # For fresher resume: earlier internships get simpler tech, later get advanced
    # block_index 0 = most recent (use advanced), 3 = oldest (use simpler)
    complexity_bias = 0.3 if block_index == 0 else (0.5 if block_index <= 1 else 0.7)
    
    # Sort by perceived complexity (versions with numbers = more specific = higher complexity)
    def complexity_score(tech: str) -> float:
        score = 0.5
        if re.search(r'\d+\.\d+', tech):  # Has version number
            score += 0.3
        if len(tech.split()) > 2:  # Longer description
            score += 0.2
        return score
    
    available_sorted = sorted(available, key=complexity_score)
    
    # Select based on complexity bias
    idx = int(len(available_sorted) * complexity_bias)
    idx = max(0, min(idx, len(available_sorted) - 1))
    chosen = available_sorted[idx]
    
    # Add usage context if provided
    if context and random.random() < 0.6:
        # Use GPT to add context naturally
        prompt = f"""Add a brief usage context to this technology mention for a resume bullet.

Technology: {chosen}
Context: {context}

Return STRICT JSON: {{"specific_mention": "technology with context"}}

Examples:
- Input: "BERT-base", Context: "sentiment classification"
  Output: {{"specific_mention": "BERT-base for sentiment classification"}}
- Input: "Llama 3.1 70B", Context: "text generation"
  Output: {{"specific_mention": "Llama 3.1 70B for instruction-tuned generation"}}
- Input: "Kubernetes", Context: "model serving"
  Output: {{"specific_mention": "Kubernetes with KServe for model serving"}}

Keep it concise (under 8 words).
"""
        try:
            data = await gpt_json(prompt, temperature=0.2)
            specific = data.get("specific_mention", chosen)
            chosen = specific
        except Exception:
            pass
    
    already_used.add(chosen.lower())
    log_event(f"🔧 [TECH SPECIFIC] {generic_term} → {chosen}")
    
    return chosen


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


# ============================================================
# 📊 Quantification templates — FIXED (no "\%" SyntaxWarning)
#   ✅ Use "%" here; latex_escape_text() will convert to "\%" later.
# ============================================================

QUANTIFICATION_SENTENCE_ENDINGS = {
    "comparison_hero": [
        "pushing validation accuracy from {start}% to {end}% after systematic tuning",
        "moving the baseline F1 from {start}% to {end}% on the held-out evaluation set",
        "closing the accuracy gap from {start}% to {end}% through targeted data cleaning and retraining",
        "improving the model's precision from {start}% to {end}% by rebalancing the training distribution",
        "lifting recall on the minority class from {start}% to {end}% via focal loss and oversampling",
    ],
    "count_scale": [
        "applied to a corpus of {count} domain-specific examples",
        "validated across a {count}-sample held-out test set spanning three label categories",
        "trained on {count} annotated records curated from internal data collection",
        "benchmarked on a balanced evaluation set of {count} instances",
        "processed {count} raw inputs through the full preprocessing and inference pipeline",
    ],
    "metric_achievement": [
        "achieving {metric_name} of {value} on the official evaluation split",
        "reaching {metric_name} of {value} after five rounds of cross-validated tuning",
        "delivering {metric_name} of {value}, exceeding the project baseline by a clear margin",
        "attaining {metric_name} of {value} on the stratified test partition",
    ],
    "percent_improvement": [
        "reducing inference error by {value} relative to the untuned checkpoint",
        "cutting the false-positive rate by {value} without degrading recall",
        "improving overall pipeline throughput by {value} through batched preprocessing",
        "shrinking average prediction latency by {value} after model pruning",
        "boosting the model's macro-F1 by {value} on the imbalanced evaluation set",
    ],
}


def generate_quantification_ending(
    category: str,
    jd_context: str = "",
    is_fresher: bool = True,
) -> str:
    """
    Generate a natural sentence ending that can be appended after a comma.
    Returns a complete phrase, not a dangling fragment.

    NOTE: Uses "%" (not "\\%") to avoid SyntaxWarning; LaTeX escaping happens later.
    """
    templates = QUANTIFICATION_SENTENCE_ENDINGS.get(
        category, QUANTIFICATION_SENTENCE_ENDINGS["percent_improvement"]
    )
    template = random.choice(templates)

    if category == "comparison_hero":
        start = random.randint(63, 75)
        if start % 2 == 0:
            start += 1

        improvement = random.randint(11, 19) if is_fresher else random.randint(15, 27)
        if improvement % 2 == 0:
            improvement += 1

        end = min(93 if is_fresher else 97, start + improvement)
        if end % 2 == 0:
            end -= 1

        return template.format(start=start, end=end)

    if category == "count_scale":
        if is_fresher:
            base = random.choice([random.randint(1500, 9999), random.randint(11000, 45000)])
        else:
            base = random.choice([random.randint(25000, 99999), random.randint(110000, 450000)])

        if base % 2 == 0:
            base += 1

        count_str = f"{(base // 1000)}K" if base >= 10000 else f"{base:,}"
        return template.format(count=count_str)

    if category == "metric_achievement":
        metric_name = random.choice(["F1 score", "precision", "macro-recall", "weighted F1"])
        lo, hi = (0.73, 0.89) if is_fresher else (0.82, 0.95)

        value = round(random.uniform(lo, hi), 2)
        for _ in range(20):
            v = round(random.uniform(lo, hi), 2)
            if int(v * 100) % 2 != 0:  # odd last digit in percent-form view
                value = v
                break

        return template.format(metric_name=metric_name, value=value)

    if category == "percent_improvement":
        base = random.randint(9, 27) if is_fresher else random.randint(15, 41)
        if base % 2 == 0:
            base += 1

        # Sometimes use decimal
        if random.random() < 0.5:
            dec = round(random.uniform(base, base + 0.9), 1)
            if int(dec * 10) % 2 == 0:
                dec = round(dec + 0.1, 1)
            value_str = f"{dec:.1f}%"
        else:
            value_str = f"{base}%"

        return template.format(value=value_str)

    return template


# Quantified positions: absolute bullet index 0-11
QUANTIFIED_POSITIONS = [0, 3, 4, 7, 11]   # 6 of 12; spread across all blocks
HERO_POSITIONS = [0, 7]                        # First bullet of block 0, second of block 2

# Category map: which metric type each quantified position gets
_QUANT_CATEGORY_MAP = {
    0: "comparison_hero",
    3: "count_scale",
    4: "percent_improvement",
    7: "comparison_hero",
    11: "metric_achievement"
}

def reset_quantification_tracking():
    reset_number_tracking()

def get_quantification_category(bullet_position: int, jd_context: str = "") -> Optional[str]:
    return _QUANT_CATEGORY_MAP.get(bullet_position)


def should_quantify_bullet(bullet_position: int) -> bool:
    return bullet_position in QUANTIFIED_POSITIONS


def _build_bullet_prompt(
    num_bullets: int,
    experience_company: str,
    target_company: str,
    target_role: str,
    specific_tech_str: str,
    realistic_tech_str: str,
    keywords_str: str,
    suggested_verbs: List[str],
    resp_str: str,
    vocab_str: str,
    exp_context: Dict[str, Any],
    progression: Dict[str, Any],
    block_index: int,
    total_blocks: int,
    believability: str,
    core_rule: str,
    quantified_bullets_in_block: List[Tuple[int, str]],
    ideal_bullet_instructions: str,
    dedup_instruction: str,
    jd_text: str,
    result_phrases: List[str],
) -> str:
    """
    Build the GPT prompt for bullet generation.
    Philosophy: ask GPT to write flowing sentences; constraints shape
    the writing rather than dictating assembly order.
    """

    # Build quantification guidance — give GPT the ENDING TEXT to weave in
    quant_guidance_parts = []
    for local_idx, category in quantified_bullets_in_block:
        ending = generate_quantification_ending(category, jd_text, is_fresher=True)
        abs_pos = (block_index * num_bullets) + local_idx
        is_hero = abs_pos in HERO_POSITIONS

        if is_hero:
            quant_guidance_parts.append(
                f"   Bullet {local_idx + 1} (HERO): End the sentence with something like:\n"
                f"   \"{ending}\"\n"
                f"   — modify the numbers slightly (keep odd last digits), weave this ending naturally into your sentence."
            )
        else:
            quant_guidance_parts.append(
                f"   Bullet {local_idx + 1}: Include a metric. Try ending with something like:\n"
                f"   \"{ending}\"\n"
                f"   — adapt the phrasing so it flows naturally from your specific sentence."
            )

    if quant_guidance_parts:
        quant_section = (
            "📊 METRIC GUIDANCE (weave these in naturally, don't paste them verbatim):\n"
            + "\n".join(quant_guidance_parts)
            + "\n\n   Bullets WITHOUT a metric listed above: end with a qualitative result clause instead.\n"
            + "   Good qualitative endings: 'enabling the team to reproduce all experiments from a single command',\n"
            + "   'surfacing a mislabeling issue that affected 18% of the training split',\n"
            + "   'reducing annotator confusion by providing clearer labeling guidelines'.\n"
            + "   BAD endings: 'improving efficiency', 'enhancing performance', 'increasing accuracy' — too vague.\n"
        )
    else:
        quant_section = (
            "📊 NO METRICS for this block. End each bullet with a specific qualitative result:\n"
            "   — 'surfacing a data imbalance that had suppressed recall on the minority class'\n"
            "   — 'enabling two downstream engineers to consume the feature store without changes'\n"
            "   — 'cutting the manual label-review queue from 3 days to under 4 hours'\n"
            "   Avoid generic endings like 'improving performance' or 'enhancing the system'.\n"
        )

    # Result phrase examples for non-quantified bullets
    result_examples = "\n".join(f"   — \"{p}\"" for p in result_phrases[:3])

    complexity = "advanced" if block_index == 0 else ("intermediate" if block_index <= 1 else "foundational")

    prompt = f"""Write exactly {num_bullets} resume bullet points for an ML/software intern at "{experience_company}", targeting a {target_role} role at {target_company}.

WRITING STYLE:
Write each bullet as a single, flowing sentence — not an assembly of parts. The sentence should read like a
human engineer wrote it in past tense. Avoid: "Utilized X for Y, achieving Z" (robot template). 
Instead: "Built a BERT-based classifier that cut annotation review time from 3 days to under 4 hours by 
automatically filtering high-confidence examples." (one flowing thought)

REQUIRED FOR EACH BULLET:
1. Start with the exact verb given (do not change it, not even capitalization)
2. Name a specific technology WITHOUT version numbers
3. Describe a concrete technical choice or configuration (not just 'used X')
4. End with a specific result — quantitative OR qualitative, but specific

WORD COUNT: 24–34 words per bullet. Count before submitting.

ACTION VERBS (use exactly, in order):
- Bullet 1: {suggested_verbs[0]}
- Bullet 2: {suggested_verbs[1]}  
- Bullet 3: {suggested_verbs[2]}

TECHNOLOGIES TO DRAW FROM (pick what fits, don't force all):
Specific implementations: {specific_tech_str}
Company-realistic tools: {realistic_tech_str}
JD keywords to incorporate: {keywords_str}
Do NOT include version numbers anywhere.

{quant_section}

WHAT MAKES A GOOD RESULT CLAUSE (examples of the kind of specificity we want):
{result_examples}

INTERN SCOPE (Block {block_index} of {total_blocks} — {'most recent' if block_index == 0 else 'earlier'} role):
- Technology complexity: {complexity}
- Scope: {progression['scope'][0]} work, {progression['autonomy']}
- Scale: thousands not millions; 75-90\% not 99\% accuracy; research/academic datasets
{f'- Context: {believability}' if believability else ''}

DOMAIN: {exp_context.get('domain', 'ML/AI')} — vocabulary: {vocab_str}
JOB RESPONSIBILITIES: {resp_str}
{core_rule}
{ideal_bullet_instructions}
{dedup_instruction}

THINGS TO AVOID:
- Vague endings: "improving performance", "enhancing accuracy", "increasing efficiency"
- Robotic structure: "[Verb] [tech] for [task], achieving [metric]" repeated 3 times
- Identical sentence structure across bullets in this block
- Version numbers: no "PyTorch 2.1", "BERT-base-uncased", "K8s 1.28"
- Repeating the same technology twice in one block
- Dangling phrases: never end with "by", "of", "to", "from", "achieving", "using"
- Numbers in bullets not listed in the metric guidance above

GOOD EXAMPLE (notice: one flowing thought, specific result, natural language):
"Fine-tuned a BERT sequence classifier on 8K biomedical abstracts using domain-adaptive pretraining, pushing NER F1 from 71\% to 83\% on the held-out evaluation split."

BAD EXAMPLE (robot template, vague result):
"Utilized BERT for text classification implementing fine-tuning techniques, achieving improved accuracy by 17\%."

Return STRICT JSON:
{{"bullets": ["{suggested_verbs[0]}...", "{suggested_verbs[1]}...", "{suggested_verbs[2]}..."], "primary_keywords_used": ["kw1", "kw2", "kw3"], "specific_technologies_used": ["tech1", "tech2", "tech3"]}}"""

    return prompt


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

async def post_process_bullets(
    bullets: List[str],
    primary_kws: List[str],
    specific_techs: List[str],
    num_bullets: int,
    quantified_bullets_in_block: List[Tuple[int, str]],
    bullet_start_position: int,
    keywords_for_block: List[str],
    suggested_verbs: List[str],
    jd_text: str,
    _global_keyword_assignments: Dict[str, int],
    fix_capitalization_gpt,     # async callable
    latex_escape_text,          # sync callable
    log_event,                  # sync callable
) -> Tuple[List[str], Set[str]]:
    """
    Post-process GPT bullet output:
    1. Fix capitalization
    2. Strip numbers from non-quantified bullets (surgically)
    3. Validate no dangling endings
    4. Adjust length
    5. LaTeX escape
    6. Track keywords
    """
    cleaned: List[str] = []
    newly_used: Set[str] = set()

    # Which local indices should have numbers
    quantified_local_indices = {local_idx for local_idx, _ in quantified_bullets_in_block}

    for local_idx, b in enumerate(bullets[:num_bullets]):
        b = str(b).strip()

        # Fix capitalization
        b = await fix_capitalization_gpt(b)

        should_have_number = local_idx in quantified_local_indices

        if not should_have_number:
            b = strip_numbers_from_bullet(b)

        # Enforce correct starting verb (GPT sometimes changes it)
        if local_idx < len(suggested_verbs):
            expected_verb = suggested_verbs[local_idx]
            # Check if bullet starts with expected verb (case-insensitive)
            first_word = b.split()[0] if b.split() else ""
            if first_word.lower() != expected_verb.lower():
                # Try to find the verb elsewhere and move it, or just prepend
                if expected_verb.lower() in b.lower():
                    # Remove it from where it is and put it at start
                    b = re.sub(
                        rf'\b{re.escape(expected_verb)}\b',
                        '', b, count=1, flags=re.IGNORECASE
                    ).strip()
                b = expected_verb + " " + b[0].lower() + b[1:] if b else expected_verb

        # Catch and fix dangling endings
        dangling_pattern = re.compile(
            r',?\s+\b(by|of|to|from|through|via|using|across|with|achieving|improving|'
            r'enhancing|boosting|increasing|reducing|raising|lifting)\s*[.,]?\s*$',
            re.IGNORECASE
        )
        match = dangling_pattern.search(b)
        if match:
            log_event(f"⚠️ [DANGLING] Detected in bullet {bullet_start_position + local_idx}: '{b[-40:]}'")
            b = b[:match.start()].rstrip(".,;: ")

            if should_have_number:
                # We need a number — add a safe generic ending
                category = next(
                    (cat for idx, cat in quantified_bullets_in_block if idx == local_idx),
                    "percent_improvement"
                )
                ending = generate_quantification_ending(category, jd_text, is_fresher=True)
                b = b.rstrip(".,;: ") + ", " + ending

            # Either way, ensure proper sentence end
            b = b.rstrip(".,;: ") + "."

        # Adjust length
        b = adjust_bullet_length(b)

        # Final punctuation
        if not b.endswith("."):
            b = b.rstrip(".,;: ") + "."

        # LaTeX escape
        b = latex_escape_text(b)

        if b:
            cleaned.append(b)

            # Track keyword usage
            for kw in keywords_for_block:
                if kw.lower() in b.lower():
                    newly_used.add(kw.lower())

            # Track primary keyword globally
            if local_idx < len(primary_kws):
                pk = primary_kws[local_idx].lower().strip()
                if pk and pk not in _global_keyword_assignments:
                    _global_keyword_assignments[pk] = bullet_start_position + local_idx

            # Track specific technologies
            if local_idx < len(specific_techs):
                tech = specific_techs[local_idx].lower().strip()
                if tech:
                    newly_used.add(tech)
                    log_event(f"🔧 [TECH] Bullet {bullet_start_position + local_idx}: {specific_techs[local_idx]}")

    return cleaned, newly_used

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


def get_verb_categories_for_context(company_type: str, block_index: int = 0) -> List[str]:
    """Get verb categories appropriate for experience level."""
    if "research" in company_type.lower():
        base = ["research", "analysis", "development", "documentation"]
    elif "industry" in company_type.lower():
        base = ["development", "deployment", "optimization", "automation"]
    else:
        base = ["development", "analysis", "collaboration", "data_work"]
    
    # Earlier internships (higher block_index) get more foundational verbs
    if block_index >= 2:
        base = ["analysis", "data_work", "documentation", "collaboration"] + base
    
    return base


# ============================================================
# 🎯 RESULT PHRASES (Impact without numbers)
# ============================================================

RESULT_PHRASES = {
    "performance": [
        "cutting misclassification rate nearly in half on the held-out test set",
        "closing the gap between validation and test accuracy to under 3 percentage points",
        "outperforming the baseline ResNet implementation on every reported metric",
        "matching published benchmark results while using 40\% fewer training examples",
        "reducing false-positive rate substantially without sacrificing recall",
    ],
    "efficiency": [
        "dropping per-epoch training time from roughly 4 hours to under 90 minutes",
        "halving peak GPU memory usage through gradient checkpointing",
        "cutting preprocessing wall-time from 6 hours to under 40 minutes",
        "reducing the full experiment cycle from days to a few hours",
        "allowing the team to run 3x as many ablations within the same compute budget",
    ],
    "quality": [
        "producing fully reproducible runs across three independent seeds",
        "eliminating the label-leakage bug that had inflated prior accuracy estimates",
        "raising inter-annotator agreement from 0.61 to 0.79 on the validation subset",
        "making the codebase readable enough for two new team members to onboard in a day",
        "catching 94\% of edge-case failures through property-based testing",
    ],
    "scalability": [
        "allowing the pipeline to ingest 5x more records without additional infrastructure",
        "keeping inference latency flat as batch size scaled from 32 to 512",
        "enabling seamless addition of new label classes without retraining from scratch",
        "handling a 10x surge in nightly job volume without queue backlog",
        "reducing peak memory footprint enough to fit the model on a single V100",
    ],
    "insight": [
        "surfacing a previously unknown correlation between input length and prediction confidence",
        "identifying that 23\% of training labels were mislabeled, prompting a data-cleaning sprint",
        "confirming that the simpler logistic baseline was competitive, saving weeks of DL work",
        "revealing that data augmentation hurt more than helped on the imbalanced split",
        "flagging three feature groups with near-zero mutual information that were pruned",
    ],
    "collaboration": [
        "enabling another team member to reproduce the full pipeline from a single README command",
        "unblocking two downstream engineers who depended on the cleaned feature store",
        "reducing back-and-forth code-review cycles from an average of 4 rounds to 1",
        "allowing the research lead to demo live results to stakeholders two weeks ahead of schedule",
        "giving the annotation team a clear labeling guide that cut ambiguity questions by half",
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
# 📈 SKILL PROGRESSION FRAMEWORK
# ============================================================

INTERN_PROGRESSION = {
    "early": {
        "scope": ["assisted with", "supported", "contributed to", "participated in"],
        "tasks": ["data preprocessing", "baseline implementation", "literature review", "code documentation"],
        "autonomy": "under close guidance of senior researchers",
        "complexity": "foundational components and exploratory tasks",
        "technologies": "standard libraries and established frameworks",
    },
    "mid": {
        "scope": ["developed", "implemented", "designed", "built"],
        "tasks": ["model development", "pipeline creation", "experiment execution", "performance analysis"],
        "autonomy": "with regular mentorship from project leads",
        "complexity": "core system components and established methodologies",
        "technologies": "modern frameworks with some customization",
    },
    "late": {
        "scope": ["led", "architected", "spearheaded", "owned"],
        "tasks": ["end-to-end pipeline", "model optimization", "comprehensive evaluation", "technical documentation"],
        "autonomy": "independently with periodic reviews",
        "complexity": "production-ready solutions and novel approaches",
        "technologies": "cutting-edge tools with advanced configurations",
    },
}


def get_progression_context(block_index: int, total_blocks: int = 4) -> Dict[str, Any]:
    """Get appropriate progression context for experience block."""
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
        "in collaboration with senior researchers",
        "as part of a cross-functional research team",
        "working closely with Ph.D. students and postdocs",
        "under guidance of faculty advisors",
        "contributing to team-wide research initiatives",
    ],
    "scope_limiters": [
        "for internal research use",
        "as proof-of-concept implementation",
        "for academic research purposes",
        "within the lab environment",
        "for experimental validation",
    ],
}


def get_believability_phrase(scope: str = "medium", block_index: int = 0) -> str:
    """Get believability phrases appropriate for intern level."""
    # Earlier internships more likely to mention collaboration
    mention_collab = random.random() < (0.5 if block_index >= 2 else 0.3)
    
    if mention_collab:
        return random.choice(BELIEVABILITY_RULES["collaboration_phrases"])
    
    # Latest internship less likely to need scope limiting
    if block_index == 0 and random.random() < 0.7:
        return ""
    
    if random.random() < 0.3:
        return random.choice(BELIEVABILITY_RULES["scope_limiters"])
    
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
- Specific models: Llama, GPT-4, BERT-base, RoBERTa, T5, DistilBERT
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
- Specific models: Llama 3.1, GPT-4, BERT-base, RoBERTa-large
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

    # Fast-reject: definitively NOT skills under any circumstance
    hard_rejects = {
        # Degrees
        "phd", "ph.d", "ms", "m.s", "msc", "m.sc", "bs", "b.s", "bsc", "b.sc",
        "bachelor", "master", "masters", "degree", "university", "college",
        # Time/experience
        "experience", "years", "year", "month", "months", "week", "weeks",
        # Qualifiers
        "required", "preferred", "plus", "bonus", "nice to have",
        "strong", "excellent", "good", "proficient", "familiar", "advanced", "basic",
        "knowledge", "understanding", "ability", "skills", "skill",
        # Compliance/standards
        "iso", "nist", "gdpr", "hipaa", "sox", "pci", "cmmi", "itil",
        "compliance", "certified", "certification",
        "iso 42001", "nist ai rmf", "ai rmf", "rmf",
        # Pure filler
        "real-time applications", "computational efficiency",
        "clinical decision support", "end-to-end", "cross-functional",
        "data driven", "business intelligence",
    }
    if keyword_lower in hard_rejects:
        _validated_skills_cache[keyword_lower] = False
        log_event(f"❌ [SKILL FAST-REJECT] '{keyword}' → Hard reject")
        return False

    # Fast-reject: standards/compliance patterns
    if re.match(r"^(iso|nist|pci|gdpr|hipaa|sox)\s*[\d/]", keyword_lower):
        _validated_skills_cache[keyword_lower] = False
        log_event(f"❌ [SKILL PATTERN-REJECT] '{keyword}' → Standard pattern")
        return False

    # Fast-reject: anything 6+ words is definitely not a skill name
    if len(keyword.split()) >= 6:
        _validated_skills_cache[keyword_lower] = False
        log_event(f"❌ [SKILL LENGTH-REJECT] '{keyword}' → Too long")
        return False

    prompt = f"""Does "{keyword}" belong in the Skills section of a software/ML/data resume?

ACCEPT (return true) for:
- Programming languages: Python, Java, C++, SQL, R, Go, Rust, MATLAB, Scala, Julia
- ML/AI frameworks: PyTorch, TensorFlow, Keras, Scikit-learn, JAX, Hugging Face, OpenCV, NLTK, spaCy
- ML/AI concepts and areas: Machine Learning, Deep Learning, NLP, Computer Vision, Reinforcement Learning, LLMs, Generative AI, MLOps, Data Science, Statistics
- Specific model architectures: BERT, GPT, Transformer, ResNet, LSTM, CNN, GAN, ViT, XGBoost, LightGBM, YOLO, Diffusion Models
- Data/engineering tools: Pandas, NumPy, Spark, Kafka, Airflow, dbt, Hadoop, Flink
- MLOps/DevOps tools: Docker, Kubernetes, Git, GitHub, MLflow, DVC, Weights & Biases, Kubeflow, Seldon
- Cloud platforms/services: AWS, GCP, Azure, SageMaker, Lambda, BigQuery, Vertex AI, EC2, S3, Databricks, Snowflake
- Databases: PostgreSQL, MongoDB, Redis, MySQL, Pinecone, ChromaDB, Elasticsearch, DynamoDB, Cassandra, Weaviate
- Protocols/formats: REST, GraphQL, gRPC, JSON, YAML, Protobuf
- Methodologies: CI/CD, MLOps, DevOps, A/B Testing, Agile, Scrum, RLHF
- Data science methods: Feature Engineering, Hypothesis Testing, EDA, Statistical Modeling, Time Series, Bayesian Inference
- Soft skills that are standard on resumes: Leadership, Communication, Problem Solving, Teamwork, Collaboration

REJECT (return false) ONLY for:
- Complete sentences or full phrases describing job duties (7+ words)
- Pure outcome statements: "Improved Model Accuracy", "Reduced Latency", "High Performance"
- Pure adjectives standing alone: "Scalable", "Distributed", "Advanced", "Real-time"
- Company-specific internal tool names that mean nothing outside the company
- Compliance/regulatory standards: ISO 27001, NIST 800, GDPR Article X, HIPAA regulations
- Academic degrees and titles: PhD, MS, Bachelor of Science
- Time-based requirements: "3 years experience", "5+ years"
- Vague non-skills: "Fast Learner", "Detail Oriented", "Self Motivated"

IMPORTANT: When in DOUBT, ACCEPT. Skills sections in ML/data resumes commonly include both specific tools AND broad domain areas. "Machine Learning", "Deep Learning", "NLP", "Computer Vision" are ALL valid resume skills.

Return STRICT JSON only: {{"is_skill": true}} or {{"is_skill": false}}
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
        # On failure, DEFAULT TO ACCEPT — better to include than exclude
        _validated_skills_cache[keyword_lower] = True
        return True


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
    "realistic_technologies": ["6-10 specific technologies/tools likely used here"],
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
- realistic_technologies should be SPECIFIC (e.g., "PyTorch 2.0", not just "deep learning")
"""
    try:
        data = await gpt_json(prompt, temperature=0.2)
        result = {
            "type": data.get("type", "internship"),
            "domain": data.get("domain", "ML/AI"),
            "context": data.get("context", "Technical internship applying Machine Learning."),
            "technical_vocabulary": data.get("technical_vocabulary", ["model development", "data analysis"]),
            "realistic_technologies": data.get("realistic_technologies", ["Python", "PyTorch", "scikit-learn"]),
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
            "realistic_technologies": ["Python", "PyTorch", "scikit-learn", "Pandas", "NumPy"],
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
            "bullet_theme": "A concrete resume bullet theme showing this capability",
            "specific_technologies": ["2-3 specific tech implementations that demonstrate this"]
        }},
        ... (exactly 6 implicit requirements, ranked 1-6 by importance)
    ],
    "top_3_must_haves": [
        "Concrete thing #1 the job DEFINITELY wants candidates to have done",
        "Concrete thing #2 the job DEFINITELY wants candidates to have done",
        "Concrete thing #3 the job DEFINITELY wants candidates to have done"
    ],
    "ideal_candidate_bullet_themes": [
        {{
            "theme": "Specific resume bullet theme",
            "specific_tech_example": "Exact technology implementation (e.g., 'fine-tuned Llama 3.1 70B')",
            "implementation_detail": "How it was done (e.g., 'with LoRA adapters on 50K samples')"
        }},
        ... (4 themes)
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
- ideal_candidate_bullet_themes should be SPECIFIC TECHNOLOGY implementations
- specific_technologies should be ACTUAL tech names (Llama 3.1, BERT-base, K8s 1.28)
- Each bullet theme must be distinct and non-overlapping
- For fresher roles, focus on learning ability and foundational skills
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
                {
                    "theme": "End-to-end ML pipeline development",
                    "specific_tech_example": "PyTorch 2.1 with custom data loaders",
                    "implementation_detail": "processing 50K samples with DataLoader optimization"
                },
                {
                    "theme": "Large-scale data processing",
                    "specific_tech_example": "Pandas with Dask for parallelization",
                    "implementation_detail": "handling 100K+ records with efficient chunking"
                },
                {
                    "theme": "Model experimentation with systematic evaluation",
                    "specific_tech_example": "MLflow experiment tracking",
                    "implementation_detail": "logging 50+ experiments with hyperparameter sweeps"
                },
                {
                    "theme": "Cross-functional collaboration on ML projects",
                    "specific_tech_example": "Git with feature branching workflow",
                    "implementation_detail": "coordinating with 3-5 team members via PRs"
                },
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
    ✨ ENHANCED: Rank ALL bullet point themes by importance.
    Determines which 8 come from keywords and which 4 from ideal candidate.
    Ensures NO keyword is used twice.
    NOW includes specific technology implementations.
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
            "specific_technology": "Exact tech to mention (e.g., 'Llama 3.1 70B', 'BERT-base-uncased')",
            "implementation_context": "How the tech was used (e.g., 'with LoRA adapters', 'for sentiment analysis')",
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
            "specific_technology": "Exact tech implementation",
            "implementation_detail": "Concrete detail about usage",
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
9. specific_technology MUST be exact implementations (not generic "NLP" but "BERT-base")
10. implementation_context shows HOW the technology was used
11. For fresher resume: earlier blocks (2-3) use simpler tech, later (0-1) use advanced
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
                "specific_technology": kw,
                "implementation_context": "for model development",
                "source": "keyword",
            })
        ideal_bullets = []
        for i in range(4):
            theme_data = bullet_themes[i] if i < len(bullet_themes) else {}
            if isinstance(theme_data, dict):
                theme = theme_data.get("theme", f"Theme {i+1}")
                specific_tech = theme_data.get("specific_tech_example", "PyTorch")
                impl_detail = theme_data.get("implementation_detail", "for model training")
            else:
                theme = str(theme_data)
                specific_tech = "PyTorch"
                impl_detail = "for model training"
            
            ideal_bullets.append({
                "bullet_index": [2, 5, 8, 11][i] if i < 4 else 11,
                "block_index": i,
                "theme": theme,
                "implicit_requirement": top_3[i] if i < len(top_3) else "",
                "supporting_keywords": [],
                "specific_technology": specific_tech,
                "implementation_detail": impl_detail,
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

def strip_numbers_from_bullet(b: str) -> str:
    """
    Remove numbers from non-quantified bullets WITHOUT leaving dangling phrases.
    Strategy: remove the entire quantified phrase, not just the number.
    """

    # "from X% to Y%" — remove entire comparison phrase
    # Replace with a qualitative equivalent so the bullet still reads naturally
    b = re.sub(
        r',?\s*(?:improving|boosting|increasing|raising|lifting|enhancing)\s+'
        r'[\w\s-]+\s+from\s+\d+[\.,]?\d*\%?\s+to\s+\d+[\.,]?\d*\%?',
        lambda m: re.sub(r'from\s+\d[\d.,]*\%?\s+to\s+\d[\d.,]*\%?',
                         'through systematic optimization', m.group()),
        b, flags=re.IGNORECASE
    )

    # "by X%" — remove "by X%" but keep the surrounding verb
    b = re.sub(r'\bby\s+\d+[\.,]?\d*\s*\%', 'through systematic tuning', b, flags=re.IGNORECASE)

    # "achieving/attaining/reaching F1 score of X.XX"
    b = re.sub(
        r'\b(achieving|attaining|reaching|delivering|securing)\s+'
        r'(?:(?:an?|the)\s+)?(?:F1|f1|precision|recall|accuracy|BLEU|ROC.AUC)\s+'
        r'(?:score\s+)?of\s+\d+[\.,]\d+',
        r'\1 strong evaluation metrics',
        b, flags=re.IGNORECASE
    )

    # "F1 score of X.XX" standalone
    b = re.sub(
        r'\b(?:F1|f1|precision|recall|accuracy|BLEU|ROC.AUC)\s+(?:score\s+)?of\s+\d+[\.,]\d+',
        'strong evaluation scores',
        b, flags=re.IGNORECASE
    )

    # "processing/analyzing/handling X,XXX samples/records/images"
    b = re.sub(
        r'\b(processing|analyzing|handling|evaluating|training on)\s+'
        r'\d[\d,]*\s*[Kk]?\+?\s+'
        r'(?:training\s+)?(?:samples|records|images|examples|instances|documents|entries)',
        r'\1 a substantial corpus of domain-specific data',
        b, flags=re.IGNORECASE
    )

    # "X,XXX+" or "XK+" standalone numbers next to data words
    b = re.sub(r'\d[\d,]*\s*[Kk]\+?\s+(?=samples|records|images)', 'thousands of ', b)

    # Standalone bare percentage not attached to anything meaningful
    # Only strip if it's isolated (preceded by space and followed by space/end)
    b = re.sub(r'(?<=\s)\d+[\.,]?\d*\s*\%(?=[\s,.]|$)', 'a significant margin', b)

    # "Xx faster/improvement/speedup"
    b = re.sub(
        r'\d+\s*x\s+(?:faster|improvement|speedup|reduction|better)',
        'significantly faster',
        b, flags=re.IGNORECASE
    )

    # "improved performance by" (now dangles after above substitution if not caught)
    # catch trailing prepositions that now have nothing after them
    b = re.sub(r',?\s+\b(by|of|to|from|achieving|attaining|reaching)\s*[.,]?\s*$',
               '', b, flags=re.IGNORECASE)

    # Clean up double spaces and trailing punctuation artefacts
    b = re.sub(r'[ \t]{2,}', ' ', b)
    b = re.sub(r'\s+([,.])', r'\1', b)
    b = re.sub(r',\s*,', ',', b)
    b = b.strip()

    return b

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

MIN_BULLET_WORDS = 22
MAX_BULLET_WORDS = 34
IDEAL_BULLET_WORDS = 27


def get_word_count(text: str) -> int:
    return len((text or "").split())


def is_valid_bullet_length(text: str) -> bool:
    count = get_word_count(text)
    return MIN_BULLET_WORDS <= count <= MAX_BULLET_WORDS


def adjust_bullet_length(text: str) -> str:
    """
    Smart length adjustment:
    - Under minimum: return as-is (GPT should have written more)
    - In range: clean ending, done
    - Over max: find the best natural cut point
    """
    text = (text or "").strip()
    words = text.split()
    n = len(words)

    # Under minimum — don't truncate further
    if n <= MIN_BULLET_WORDS:
        return text.rstrip(".,;:") + "."

    # In range — just clean ending
    if n <= MAX_BULLET_WORDS:
        return text.rstrip(".,;:") + "."

    # Over MAX — find natural cut point working backwards from MAX
    # Priority: comma > semicolon > "and" > "with" > "using" > hard cut
    candidate = " ".join(words[:MAX_BULLET_WORDS])

    # Try to find the last comma in the final 10 words
    window_start = MAX_BULLET_WORDS - 10
    window_words = words[window_start:MAX_BULLET_WORDS]
    window_str = " ".join(window_words)

    last_comma = window_str.rfind(",")
    if last_comma > 0:
        # Reconstruct up to that comma
        prefix_words = words[:window_start]
        prefix_str = " ".join(prefix_words)
        suffix = window_str[:last_comma]
        combined = (prefix_str + " " + suffix).strip() if prefix_str else suffix
        # Only use if we don't go below minimum
        if get_word_count(combined) >= MIN_BULLET_WORDS:
            return combined.rstrip(".,;:") + "."

    # No good comma — check for connector words at boundary
    for connector in [" and ", " with ", " using ", " via ", " by ", " to ", " for "]:
        idx = candidate.rfind(connector)
        if idx > 0:
            trimmed = candidate[:idx]
            if get_word_count(trimmed) >= MIN_BULLET_WORDS:
                return trimmed.rstrip(".,;:") + "."

    # Absolute fallback — hard truncate
    return candidate.rstrip(".,;:") + "."


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
    """
    Match a LaTeX \\section{...} whose content CONTAINS all words from `name`
    (case-insensitive, order-independent). This handles:
      - \\section*{...}
      - Ampersands written as & or \\&
      - Extra words (e.g. "Awards & Achievements" matches "Achievements")
    """
    # Build a pattern that requires each significant word to appear inside the braces
    words = [w for w in re.split(r"\W+", name) if len(w) > 2]
    if not words:
        words = [name]

    # Each word must appear somewhere inside the { } after \section
    lookaheads = "".join(
        rf"(?=[^{{}}]*\b{re.escape(w)}\b)" for w in words
    )
    return re.compile(
        rf"(\\section\*?\{{{lookaheads}[^}}]*\}}[\s\S]*?)(?=\\section\*?\{{|\\end\{{document\}}|$)",
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
    # First pass: tiered keyword extraction
    prompt_tiered = f"""Analyze this job description and extract ALL technical keywords mentioned anywhere.

JOB DESCRIPTION:
{jd_text}

Return STRICT JSON:
{{
    "must_have": ["Python", "PyTorch", "Machine Learning", "SQL"],
    "should_have": ["Docker", "AWS", "MLOps", "Deep Learning"],
    "nice_to_have": ["Git", "Linux", "Agile", "NLP"],
    "key_responsibilities": ["5-7 main job duties as short phrases"],
    "domain_context": "brief domain description"
}}

EXTRACTION RULES:
- Extract EVERYTHING mentioned as a skill requirement — both specific tools AND broad areas
- must_have: explicitly required or "must have" in JD
- should_have: "preferred", "nice to have", or implied strongly
- nice_to_have: mentioned anywhere else or implied
- Include ALL of: programming languages, frameworks, libraries, tools, platforms, cloud services,
  ML concepts (Machine Learning, Deep Learning, NLP, Computer Vision, etc.),
  model architectures (BERT, GPT, Transformers, CNN, etc.),
  methodologies (MLOps, CI/CD, A/B Testing, etc.),
  data tools (Spark, Kafka, Airflow, etc.)
- DO NOT include: ISO/NIST/GDPR compliance standards, academic degrees, years of experience,
  personality traits, vague qualifiers (strong, excellent, good)
- BE EXHAUSTIVE — extract every single technical and domain term from the JD
"""

    # Second pass: catch anything the first pass missed
    prompt_deep = f"""You are an expert ATS keyword extractor for ML/AI/data roles.
Extract EVERY SINGLE technical skill, tool, framework, concept, and domain term from this JD.

JOB DESCRIPTION:
{jd_text}

Return STRICT JSON:
{{
    "all_extracted_keywords": [
        "every tool, framework, language, platform, concept, methodology, architecture mentioned"
    ],
    "implicit_skills": [
        "skills strongly implied but not explicitly stated",
        "e.g. if they mention 'train transformers' → PyTorch or TensorFlow implied",
        "e.g. if they mention 'NLP tasks' → BERT, spaCy implied"
    ],
    "domain_terms": [
        "domain-specific terms and concepts used in the JD",
        "e.g. 'recommendation systems', 'fraud detection', 'time series forecasting'"
    ]
}}

RULES:
- Cast a WIDE net — include everything that could appear in a resume skills section
- Include both SPECIFIC tools (PyTorch, scikit-learn) AND DOMAIN AREAS (Machine Learning, NLP)
- Include methodologies and practices (MLOps, Agile, CI/CD, A/B Testing)
- Include data science concepts (Feature Engineering, Statistical Modeling, EDA)
- DO NOT include: compliance standards (ISO, NIST, GDPR), degrees, years of experience,
  pure adjectives (scalable, real-time), personality traits, full sentences
- Aim for 30-60 items total — be exhaustive
"""

    try:
        # Run both passes concurrently
        tiered_task = gpt_json(prompt_tiered, temperature=0.0)
        deep_task = gpt_json(prompt_deep, temperature=0.0)
        tiered_data, deep_data = await asyncio.gather(tiered_task, deep_task)

        # --- Tiered keyword extraction ---
        must_raw    = [str(k).strip() for k in tiered_data.get("must_have",    []) if str(k).strip()]
        should_raw  = [str(k).strip() for k in tiered_data.get("should_have",  []) if str(k).strip()]
        nice_raw    = [str(k).strip() for k in tiered_data.get("nice_to_have", []) if str(k).strip()]
        responsibilities = list(tiered_data.get("key_responsibilities", []))
        domain = tiered_data.get("domain_context", "Technology")

        # Merge ALL deep extraction into tiered lists (implicit and domain included)
        tiered_seen: Set[str] = set()
        for k in must_raw + should_raw + nice_raw:
            tiered_seen.add(k.lower())


        # Batch fix capitalization for ALL keywords in one call
        all_raw = must_raw + should_raw + nice_raw
        if all_raw:
            all_fixed = await fix_capitalization_batch(all_raw)
            idx = 0
            must_have    = all_fixed[idx: idx + len(must_raw)];   idx += len(must_raw)
            should_have  = all_fixed[idx: idx + len(should_raw)]; idx += len(should_raw)
            nice_to_have = all_fixed[idx: idx + len(nice_raw)]
        else:
            must_have, should_have, nice_to_have = [], [], []

        # Global dedup preserving tier priority
        seen_final: Set[str] = set()

        def dedup(lst: List[str]) -> List[str]:
            out: List[str] = []
            for item in lst:
                item = str(item).strip()
                if item and item.lower() not in seen_final:
                    seen_final.add(item.lower())
                    out.append(item)
            return out

        must_have    = dedup(must_have)
        should_have  = dedup(should_have)
        nice_to_have = dedup(nice_to_have)
        all_keywords = must_have + should_have + nice_to_have

        log_event(
            f"💡 [JD KEYWORDS] must={len(must_have)}, should={len(should_have)}, "
            f"nice={len(nice_to_have)}, TOTAL={len(all_keywords)} "
        )

        return {
            "must_have":            must_have,
            "should_have":          should_have,
            "nice_to_have":         nice_to_have,
            "all_keywords":         all_keywords,
            "responsibilities":     responsibilities,
            "domain":               domain,
        }

    except Exception as e:
        log_event(f"⚠️ [JD KEYWORDS] Failed: {e}")
        return {
            "must_have":            [],
            "should_have":          [],
            "nice_to_have":         [],
            "all_keywords":         [],
            "responsibilities":     [],
            "domain":               "Technology",
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
# ✨ ENHANCED BULLET GENERATION v1.0.0 — TECHNOLOGY SPECIFICITY
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
    Complete replacement of generate_credible_bullets.
    
    What changed vs the old version:
    - Uses _build_bullet_prompt() for a prose-first, natural-flow prompt
    - Uses generate_quantification_ending() so numbers are woven in, not pasted
    - Uses post_process_bullets() for surgical number stripping
    - Fallbacks are full, natural sentences not assembled from parts
    """
    exp_context = await get_company_context_gpt(experience_company)
    progression = get_progression_context(block_index, total_blocks)

    # Which bullets in this block get quantification
    quantified_bullets_in_block: List[Tuple[int, str]] = []
    for i in range(num_bullets):
        bullet_pos = bullet_start_position + i
        if should_quantify_bullet(bullet_pos):
            category = get_quantification_category(bullet_pos, jd_text)
            if category:
                quantified_bullets_in_block.append((i, category))

    # Pre-select unique verbs
    verb_categories = get_verb_categories_for_context(exp_context.get("type", "internship"), block_index)
    suggested_verbs: List[str] = []
    for cat in (verb_categories * 3)[:num_bullets]:
        verb = get_diverse_verb(cat)
        suggested_verbs.append(verb)

    # Result phrases for the prompt examples
    result_phrase_examples = [get_result_phrase("performance"), get_result_phrase("efficiency")]
    believability = get_believability_phrase(block_index=block_index)

    # Build keyword pool
    available_must = [k for k in must_use_keywords
                      if k.lower() not in _global_keyword_assignments and k.lower() not in used_keywords]
    available_should = [k for k in should_use_keywords
                        if k.lower() not in _global_keyword_assignments and k.lower() not in used_keywords]
    core_pool = [k for k in (company_core_keywords or [])
                 if k.lower() not in _global_keyword_assignments and k.lower() not in used_keywords]

    keywords_for_block = core_pool[:3] + available_must[:6] + available_should[:4]
    keywords_for_block = [k for k in keywords_for_block if k]
    if keywords_for_block:
        keywords_for_block = await fix_capitalization_batch(keywords_for_block)

    # Convert to specific technologies
    specific_technologies: List[str] = []
    for kw in keywords_for_block[:10]:
        specific_tech = await get_specific_technology(
            kw,
            context="; ".join(responsibilities[:2]) if responsibilities else "",
            block_index=block_index,
        )
        specific_technologies.append(specific_tech)

    keywords_str = ", ".join(keywords_for_block[:10]) if keywords_for_block else "Python, Machine Learning"
    specific_tech_str = ", ".join(specific_technologies[:8]) if specific_technologies else keywords_str
    resp_str = "; ".join(responsibilities[:3]) if responsibilities else "Model Development; Evaluation; Deployment"

    core_focus_str = ", ".join(core_pool[:4]) if core_pool else ""
    core_rule = f"- Incorporate these target-company focus areas naturally: {core_focus_str}\n" if core_focus_str else ""

    tech_vocab = exp_context.get("technical_vocabulary", [])
    vocab_str = ", ".join(tech_vocab[:5]) if tech_vocab else ""

    realistic_tech = exp_context.get("realistic_technologies", [])
    realistic_tech_str = ", ".join(realistic_tech[:6]) if realistic_tech else ""

    # Build ideal candidate instructions
    ideal_bullet_instructions = ""
    if ideal_candidate and bullet_plan:
        ideal_themes_in_block = [
            ib for ib in bullet_plan.get("ideal_candidate_bullets", [])
            if ib.get("block_index") == block_index
        ]
        if ideal_themes_in_block:
            t = ideal_themes_in_block[0]  # one per block
            ideal_bullet_instructions = (
                f"\n🌟 ONE BULLET should demonstrate: \"{t.get('theme', '')}\"\n"
                f"   Technology example: {t.get('specific_technology', 'N/A')}\n"
                f"   Implementation context: {t.get('implementation_detail', 'N/A')}\n"
                f"   Weave in these keywords naturally: {', '.join(t.get('supporting_keywords', [])[:3])}\n"
                f"   This bullet should feel like it's describing real work, not citing a requirement.\n"
            )

    # Keyword dedup instruction
    already_used = list(_global_keyword_assignments.keys())
    dedup_instruction = ""
    if already_used:
        dedup_instruction = (
            f"\n🚫 DO NOT use these as primary keywords (already covered): "
            f"{', '.join(already_used[:20])}\n"
        )

    # Build the prompt
    prompt = _build_bullet_prompt(
        num_bullets=num_bullets,
        experience_company=experience_company,
        target_company=target_company,
        target_role=target_role,
        specific_tech_str=specific_tech_str,
        realistic_tech_str=realistic_tech_str,
        keywords_str=keywords_str,
        suggested_verbs=suggested_verbs,
        resp_str=resp_str,
        vocab_str=vocab_str,
        exp_context=exp_context,
        progression=progression,
        block_index=block_index,
        total_blocks=total_blocks,
        believability=believability,
        core_rule=core_rule,
        quantified_bullets_in_block=quantified_bullets_in_block,
        ideal_bullet_instructions=ideal_bullet_instructions,
        dedup_instruction=dedup_instruction,
        jd_text=jd_text,
        result_phrases=result_phrase_examples,
    )

    try:
        data = await gpt_json(prompt, temperature=0.35)
        bullets = data.get("bullets", []) or []
        primary_kws = data.get("primary_keywords_used", []) or []
        specific_techs = data.get("specific_technologies_used", []) or []

        cleaned, newly_used = await post_process_bullets(
            bullets=bullets,
            primary_kws=primary_kws,
            specific_techs=specific_techs,
            num_bullets=num_bullets,
            quantified_bullets_in_block=quantified_bullets_in_block,
            bullet_start_position=bullet_start_position,
            keywords_for_block=keywords_for_block,
            suggested_verbs=suggested_verbs,
            jd_text=jd_text,
            _global_keyword_assignments=_global_keyword_assignments,
            fix_capitalization_gpt=fix_capitalization_gpt,
            latex_escape_text=latex_escape_text,
            log_event=log_event,
        )

    except Exception as e:
        log_event(f"⚠️ [BULLETS] Generation failed for {experience_company}: {e}")
        cleaned = []
        newly_used = set()

    # Fallback: write complete natural sentences, not assembled templates
    fallback_sentences = [
        (
            "{verb} a {tech}-based preprocessing pipeline that normalized and deduplicated "
            "{count} domain-specific records, enabling downstream models to train on a "
            "clean, balanced dataset without manual intervention."
        ),
        (
            "{verb} experiment tracking for the team's {tech} runs using MLflow, making it "
            "straightforward to reproduce any of the 40+ logged configurations and compare "
            "hyperparameter sweeps side-by-side."
        ),
        (
            "{verb} a {tech} evaluation harness with stratified cross-validation and "
            "per-class reporting, surfacing a mislabeling pattern that affected the "
            "minority class and prompted a targeted annotation review."
        ),
    ]

    while len(cleaned) < num_bullets:
        idx = len(cleaned)
        verb = suggested_verbs[idx] if idx < len(suggested_verbs) else "Built"

        kw_list = keywords_for_block or ["Machine Learning"]
        generic_kw = kw_list[idx % len(kw_list)]
        tech1 = await get_specific_technology(generic_kw, context="model development", block_index=block_index)

        template = fallback_sentences[idx % len(fallback_sentences)]
        count_val = f"{random.randint(3, 15) * 1000 + random.randint(100, 999):,}"
        fallback = template.format(verb=verb, tech=tech1, count=count_val)

        # Apply quantification if needed
        should_have_number = any(local_idx == idx for local_idx, _ in quantified_bullets_in_block)
        if should_have_number:
            category = next(
                (cat for i, cat in quantified_bullets_in_block if i == idx),
                "percent_improvement"
            )
            ending = generate_quantification_ending(category, jd_text, is_fresher=True)
            fallback = fallback.rstrip(".,") + ", " + ending + "."

        fallback = adjust_bullet_length(fallback)
        cleaned.append(latex_escape_text(fallback))
        newly_used.add(tech1.lower())
        newly_used.add(generic_kw.lower())

    log_event(f"✅ [BULLETS BLOCK {block_index}] {len(cleaned)} bullets generated")
    for i, bullet in enumerate(cleaned):
        log_event(f"   • [{bullet_start_position + i}] {bullet[:90]}...")

    return cleaned[:num_bullets], newly_used


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
    """Rewrite all experience bullets using the enhanced v1.0.0 plan with technology specificity."""
    # Reset all tracking for new resume
    reset_verb_tracking()
    reset_result_phrase_tracking()
    reset_quantification_tracking()
    reset_keyword_assignment_tracking()
    reset_technology_tracking()

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

    # Extract experience company names from tex via GPT
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
    log_event(f"🔧 [TECH SPECIFIC] Unique technologies used: {len(_used_specific_technologies)}")

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

# REPLACE entire _pdf_page_count with this:

def _pdf_page_count(pdf_bytes: Optional[bytes]) -> int:
    """
    Count pages in a PDF byte string.
    Multiple strategies in order of reliability, with size-based fallback.
    """
    if not pdf_bytes or len(pdf_bytes) < 10:
        return 0

    # Strategy 1: /Pages dict with /Count — most authoritative
    # Walk ALL /Pages entries, take the max /Count found
    # (handles both regular and linearised PDFs)
    best_count = 0
    for m in re.finditer(rb"/Type\s*/Pages\b", pdf_bytes):
        snippet = pdf_bytes[m.start(): m.start() + 512]
        cm = re.search(rb"/Count\s+(\d+)", snippet)
        if cm:
            best_count = max(best_count, int(cm.group(1)))
    if best_count > 0:
        log_event(f"📄 [PAGE COUNT] /Pages /Count: {best_count}")
        return best_count

    # Strategy 2: global /Count scan — picks up linearised PDFs
    all_counts = [int(c) for c in re.findall(rb"/Count\s+(\d+)", pdf_bytes)]
    if all_counts:
        count = max(all_counts)
        if count > 0:
            log_event(f"📄 [PAGE COUNT] global /Count max: {count}")
            return count

    # Strategy 3: leaf /Page objects (have /MediaBox or /Contents; parent /Pages nodes don't)
    leaf_pages = re.findall(
        rb"/Type\s*/Page(?!\s*/Pages)\b(?=[\s/\]>])",
        pdf_bytes,
    )
    if leaf_pages:
        count = len(leaf_pages)
        log_event(f"📄 [PAGE COUNT] leaf /Page objects: {count}")
        return count

    # Strategy 4: /MediaBox — appears once per page, never on /Pages parent nodes
    count = len(re.findall(rb"/MediaBox\s*\[", pdf_bytes))
    if count > 0:
        log_event(f"📄 [PAGE COUNT] /MediaBox: {count}")
        return count

    # Strategy 5 (SIZE-BASED FALLBACK):
    # pdflatex resume output is highly predictable:
    #   1-page resume with fonts embedded  ≈  60 KB – 95 KB
    #   2-page resume                      ≈ 110 KB – 180 KB
    # Use 100 KB as the threshold.
    size_kb = len(pdf_bytes) / 1024
    if size_kb > 134:
        log_event(
            f"📄 [PAGE COUNT] size-based fallback: {size_kb:.1f} KB → 2 pages"
        )
        return 2
    log_event(
        f"📄 [PAGE COUNT] size-based fallback: {size_kb:.1f} KB → 1 page"
    )
    return 1

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

# REPLACE the ACHIEVEMENT_SECTION_NAMES list with this:

ACHIEVEMENT_SECTION_NAMES = [
    "Achievements",
    "Achievements & Leadership",
    "Awards",
    "Honors",
    "Certifications",
    "Awards & Achievements",
    "Achievements & Awards",
    "Honors & Awards",
    "Awards & Honors",
    "Extracurricular",
    "Extracurricular Activities",
    "Activities",
    "Leadership",
    "Volunteer",
    "Publications",
    "Projects",          # try projects before experience
]


def remove_one_achievement_bullet(tex_content: str) -> Tuple[str, bool]:
    """
    Remove the LAST bullet from ANY achievement-type section.
    If the section becomes empty after removal, remove the entire section header too.
    Returns (modified_tex, True) if something was removed, else (original_tex, False).
    """
    for sec in ACHIEVEMENT_SECTION_NAMES:
        pat = section_rx(sec)
        m = pat.search(tex_content)
        if not m:
            continue
        full = m.group(1)
        items = find_resume_items(full)
        if not items:
            continue

        # Remove the last item
        s, _, _, e = items[-1]
        new_sec = full[:s] + full[e:]

        # If section now has NO items at all, strip the entire section block
        remaining_items = find_resume_items(new_sec)
        if not remaining_items:
            # Remove the whole section from tex_content
            log_event(f"✂️ [TRIM] Section '{sec}' now empty — removing entire section")
            result = tex_content[:m.start()] + tex_content[m.end():]
        else:
            log_event(f"✂️ [TRIM] Removed last bullet from '{sec}' ({len(remaining_items)} remain)")
            result = tex_content[:m.start()] + new_sec + tex_content[m.end():]

        return result, True

    return tex_content, False
async def filter_valid_skills(keywords: List[str]) -> List[str]:
    """
    Validate each keyword in parallel. Preserves order and duplicates.
    Uses per-keyword caching — cache is populated by is_valid_skill().
    """
    if not keywords:
        return []

    tasks = [is_valid_skill(kw) for kw in keywords]
    results = await asyncio.gather(*tasks)

    valid_skills: List[str] = []
    removed_log: List[str] = []

    for kw, ok in zip(keywords, results):
        if ok:
            valid_skills.append(kw)
        else:
            removed_log.append(kw)

    if removed_log:
        log_event(
            f"🧹 [SKILL FILTER] Removed {len(removed_log)}: "
            f"{', '.join(removed_log[:8])}"
            + (f" ... +{len(removed_log)-8} more" if len(removed_log) > 8 else "")
        )

    log_event(f"✅ [SKILL FILTER] Kept {len(valid_skills)}/{len(keywords)}")
    return valid_skills


def clear_skill_validation_cache() -> None:
    """
    Clear the skill validation cache. Call this at the start of each request
    so stale False-cached entries from old strict filter don't persist.
    """
    global _validated_skills_cache
    cleared = len(_validated_skills_cache)
    _validated_skills_cache = {}
    log_event(f"🗑️ [SKILL CACHE] Cleared {cleared} cached entries")

def remove_last_bullet_from_sections(
    tex_content: str,
    sections: Tuple[str, ...] = ("Projects", "Experience"),
) -> Tuple[str, bool]:
    """
    Remove the very last bullet across ALL listed sections (finds last match globally).
    Skips sections with only 1 bullet left to avoid gutting the resume.
    Returns (modified_tex, True) if something was removed, else (original_tex, False).
    """
    # Collect ALL section matches across ALL listed sections, sorted by position
    all_matches: List[Tuple[int, re.Match, str]] = []  # (last_item_start, match, section_name)

    for sec in sections:
        pat = section_rx(sec)
        for match in pat.finditer(tex_content):
            full = match.group(1)
            items = find_resume_items(full)
            # Keep at least 1 bullet per section — don't remove the only bullet
            if len(items) < 2:
                continue
            # Track position of the last bullet in this match
            last_item_start = items[-1][0]
            all_matches.append((last_item_start, match, sec))

    if not all_matches:
        log_event("✂️ [TRIM] No removable bullets found in sections (all have ≤1 bullet)")
        return tex_content, False

    # Pick the match whose last bullet appears LATEST in the document
    # (removes from the bottom of the page first)
    all_matches.sort(key=lambda x: x[0], reverse=True)
    _, target_match, target_sec = all_matches[0]

    full = target_match.group(1)
    items = find_resume_items(full)
    s, _, _, e = items[-1]
    new_sec = full[:s] + full[e:]

    log_event(f"✂️ [TRIM] Removed last bullet from '{target_sec}' ({len(items)-1} remain)")
    result = tex_content[:target_match.start()] + new_sec + tex_content[target_match.end():]
    return result, True


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

async def optimize_resume(
    base_tex: str,
    jd_text: str,
    target_company: str,
    target_role: str,
    extra_keywords: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    log_event("🟦 [OPTIMIZE] Starting v1.0.0 — inclusive skill extraction + strict junk filter")

    # ← CRITICAL: clear stale cache from previous requests / old strict filter
    # Without this, keywords rejected by the old strict GPT prompt stay False
    # forever for the lifetime of the FastAPI process, silently dropping valid skills.
    clear_skill_validation_cache()

    # 1) JD keywords — exhaustive dual-pass extraction
    jd_info = await extract_keywords_with_priority(jd_text)

    # 2) Company-core expectations
    company_core = await extract_company_core_requirements(target_company, target_role, jd_text)
    core_keywords_raw = company_core.get("core_keywords", []) or []
    core_keywords = await fix_capitalization_batch(
        [str(k).strip() for k in core_keywords_raw if str(k).strip()]
    )

    # 3) Ideal candidate profiling
    log_event("🌟 [IDEAL CANDIDATE] Profiling ideal candidate...")
    ideal_candidate = await profile_ideal_candidate(jd_text, target_company, target_role)
    log_event(f"🌟 [IDEAL CANDIDATE] Top 3 must-haves: {ideal_candidate.get('top_3_must_haves', [])}")

    # 4) Validate extracted keywords — permissive filter (removes only true junk)
    log_event("🔍 [SKILL VALIDATION] Filtering junk while preserving all legitimate JD skills...")
    all_keywords_raw = list(jd_info.get("all_keywords", []) or [])

    # Merge core keywords into raw pool before validation
    for k in core_keywords:
        if k and k.lower() not in [x.lower() for x in all_keywords_raw]:
            all_keywords_raw.append(k)

    # Validate everything in parallel — all 6 lists run concurrently
    (
        validated_keywords,
        must_have_validated,
        should_have_validated,
        nice_to_have_validated,
        core_validated,
    ) = await asyncio.gather(
        filter_valid_skills(all_keywords_raw),
        filter_valid_skills(jd_info.get("must_have",          [])),
        filter_valid_skills(jd_info.get("should_have",        [])),
        filter_valid_skills(jd_info.get("nice_to_have",       [])),
        filter_valid_skills(core_keywords),
    )

    jd_info["must_have"]             = must_have_validated
    jd_info["should_have"]           = should_have_validated
    jd_info["nice_to_have"]          = nice_to_have_validated
    jd_info["all_keywords"]          = validated_keywords
    jd_info["company_core_keywords"] = core_validated
    core_keywords                    = core_validated
    all_keywords                     = validated_keywords

    log_event(
        f"🔍 [VALIDATION COMPLETE] "
        f"must={len(must_have_validated)}, should={len(should_have_validated)}, "
        f"nice={len(nice_to_have_validated)}, core={len(core_validated)}, "
        f"total_validated={len(validated_keywords)}"
    )

    # 5) Extra keywords (user-supplied)
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

    log_event(
        f"📊 [KEYWORDS] JD={len(jd_info.get('all_keywords', []))} "
        f"+ CORE={len(core_keywords)} + EXTRA={len(extra_list)} → TOTAL={len(all_keywords)}"
    )

    # 6) Rank and plan all 12 bullets
    log_event("📋 [BULLET RANKING] Planning 12 bullets (8 keyword + 4 ideal) with SPECIFIC TECH...")
    bullet_plan = await rank_all_bullet_points(
        jd_text=jd_text,
        target_company=target_company,
        target_role=target_role,
        jd_keywords=all_keywords,
        ideal_candidate=ideal_candidate,
    )
    log_event(
        f"📋 [BULLET PLAN] keyword_bullets={len(bullet_plan.get('keyword_bullets', []))}, "
        f"ideal_bullets={len(bullet_plan.get('ideal_candidate_bullets', []))}"
    )

    # 7) Coursework
    courses = await extract_coursework_gpt(jd_text, max_courses=24)

    # 8) Split preamble / body
    preamble, body = _split_preamble_body(base_tex)

    # 9) Update coursework
    body = replace_relevant_coursework_distinct(body, courses, max_per_line=8)
    log_event("✅ [COURSEWORK] Updated")

    # 10) Rewrite experience bullets
    body, exp_used_keywords = await rewrite_experience_with_skill_alignment(
        body, jd_text, jd_info,
        target_company=target_company,
        target_role=target_role,
        company_core_keywords=core_keywords,
        bullet_plan=bullet_plan,
        ideal_candidate=ideal_candidate,
    )
    log_event(
        f"✅ [EXPERIENCE] {len(exp_used_keywords)} keywords used, "
        f"{len(_global_keyword_assignments)} unique primary keywords, "
        f"{len(_used_specific_technologies)} specific technologies"
    )

    # ── 11) BUILD SKILLS SECTION ────────────────────────────────────────────
    # Priority order (highest → lowest):
    #   A) must_have from JD             — always present, highest priority
    #   B) company core keywords         — implicit tool expectations
    #   C) should_have from JD           — present
    #   D) nice_to_have from JD          — present
    #   F) keywords used in bullets      — ensures consistency (filtered for skill names only)
    #   G) extra user-supplied keywords  — always present
    #
    # Only hard junk excluded:
    #   ✗ Sentences / full phrases describing job duties (caught by is_valid_skill)
    #   ✗ Pure outcome phrases: "Improved Accuracy", "Reduced Latency"
    #   ✗ Compliance standards: ISO, NIST, GDPR
    #   ✗ Degrees, time periods, personality traits

    skills_raw: List[str] = []
    skills_seen: Set[str] = set()

    def _add_to_skills(kw_list: List[str]) -> None:
        for kw in kw_list:
            kw = (kw or "").strip()
            if kw and kw.lower() not in skills_seen:
                skills_seen.add(kw.lower())
                skills_raw.append(kw)

    # A) Must-have — highest priority
    _add_to_skills(jd_info.get("must_have", []))

    # B) Company core keywords
    _add_to_skills(core_keywords)

    # C) Should-have from JD
    _add_to_skills(jd_info.get("should_have", []))

    # D) Nice-to-have from JD
    _add_to_skills(jd_info.get("nice_to_have", []))

    # F) Keywords actually used in bullets — filter to short skill-name-length only
    bullet_kws_fixed = [
        fix_skill_capitalization_sync(k) for k in exp_used_keywords
        if k and len(k.split()) <= 4  # only short terms, not phrases
    ]
    _add_to_skills(bullet_kws_fixed)

    # G) Extra user-supplied keywords
    _add_to_skills(extra_list)

    # Final validation pass — removes any remaining junk that slipped through
    # NOTE: cache is warm from step 4, so this is fast (no new GPT calls)
    skills_list = await filter_valid_skills(skills_raw)
    if skills_list:
        skills_list = await fix_capitalization_batch(skills_list)

    body = await replace_skills_section(body, skills_list)
    log_event(
        f"✅ [SKILLS] {len(skills_list)} validated skills "
        f"(must={len(jd_info.get('must_have', []))}, "
        f"core={len(core_keywords)}, "
        f"should={len(jd_info.get('should_have', []))}, "
        f"nice={len(jd_info.get('nice_to_have', []))}, "
        f"bullet_kws={len(bullet_kws_fixed)}, "
        f"extra={len(extra_list)})"
    )

    # 12) Merge back
    final_tex = _merge_tex(preamble, body)

    # 13) Coverage — measured against the full exhaustive keyword set
    coverage = compute_coverage(final_tex, all_keywords)
    log_event(f"📊 [COVERAGE] {coverage['ratio']:.1%}")

    all_numbers_used = (
        list(_used_numbers_by_category["percent"])  +
        list(_used_numbers_by_category["count"])    +
        list(_used_numbers_by_category["metric"])   +
        list(_used_numbers_by_category["comparison"])
    )
    log_event(f"🎲 [UNIQUE NUMBERS] Used: {all_numbers_used}")

    return final_tex, {
        "jd_info":           jd_info,
        "company_core":      company_core,
        "ideal_candidate":   ideal_candidate,
        "bullet_plan":       bullet_plan,
        "all_keywords":      all_keywords,
        "coverage":          coverage,
        "exp_used_keywords": list(exp_used_keywords),
        "skills_list":       skills_list,
        "unique_numbers_used":        all_numbers_used,
        "global_keyword_assignments": dict(_global_keyword_assignments),
        "specific_technologies_used": list(_used_specific_technologies),
        "skills_breakdown": {
            "must_have_count":    len(jd_info.get("must_have",    [])),
            "core_count":         len(core_keywords),
            "should_have_count":  len(jd_info.get("should_have",  [])),
            "nice_to_have_count": len(jd_info.get("nice_to_have", [])),
            "bullet_kws_count":   len(bullet_kws_fixed),
            "extra_count":        len(extra_list),
            "total":              len(skills_list),
        },
    }

# ============================================================
# 🚀 API Endpoint v1.0.0 — Alignment + Reliable Trim Loop
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
        _ = use_humanize  # ignored

        jd_text = (jd_text or "").strip()
        if not jd_text:
            raise HTTPException(status_code=400, detail="jd_text is required.")

        # ─────────────────────────────────────────────
        # Load base resume
        # ─────────────────────────────────────────────
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
            if not default_path or not default_path.exists():
                raise HTTPException(status_code=500, detail="Default base resume not found")

            raw_tex = default_path.read_text(encoding="utf-8")
            log_event(f"📄 Using default base: {default_path}")

        target_company, target_role = await extract_company_role(jd_text)
        safe_company = safe_filename(target_company)
        safe_role    = safe_filename(target_role)

        # ─────────────────────────────────────────────
        # Run optimizer
        # ─────────────────────────────────────────────
        optimized_tex, info = await optimize_resume(
            raw_tex,
            jd_text,
            target_company=target_company,
            target_role=target_role,
            extra_keywords=extra_keywords,
        )

        # ─────────────────────────────────────────────
        # Compile helper
        # ─────────────────────────────────────────────
        cur_tex = optimized_tex

        def _compile(tex_str: str) -> bytes:
            rendered = render_final_tex(tex_str)
            try:
                result = compile_latex_safely(rendered)
            except Exception as exc:
                debug_path = Path(f"/tmp/debug_failed_{safe_company}_{safe_role}.tex")
                debug_path.write_text(rendered, encoding="utf-8")
                raise HTTPException(
                    status_code=500,
                    detail=f"LaTeX compilation failed: {exc}",
                )
            if not result:
                raise HTTPException(
                    status_code=500,
                    detail="LaTeX compilation produced empty output.",
                )
            return result

        # ─────────────────────────────────────────────
        # Initial compile
        # ─────────────────────────────────────────────
        cur_pdf       = _compile(cur_tex)
        initial_pages = _pdf_page_count(cur_pdf)

        # ─────────────────────────────────────────────
        # Trim loop
        # ─────────────────────────────────────────────
        MAX_TRIMS        = 60
        trim_count       = 0
        no_shrink_streak = 0
        prev_pdf_size    = len(cur_pdf)

        while trim_count < MAX_TRIMS:
            pages = _pdf_page_count(cur_pdf)
            if pages <= 1:
                break

            new_tex, removed = remove_one_achievement_bullet(cur_tex)

            if not removed:
                new_tex, removed = remove_last_bullet_from_sections(
                    cur_tex, ("Experience",)
                )

            if not removed:
                break

            try:
                new_pdf = _compile(new_tex)
            except HTTPException:
                break

            trim_count += 1
            new_size = len(new_pdf)

            if new_size >= prev_pdf_size:
                no_shrink_streak += 1
                if no_shrink_streak >= 4:
                    cur_tex = new_tex
                    cur_pdf = new_pdf
                    break
            else:
                no_shrink_streak = 0

            cur_tex       = new_tex
            cur_pdf       = new_pdf
            prev_pdf_size = new_size

        final_pages = _pdf_page_count(cur_pdf)

        optimized_tex_final = cur_tex
        pdf_bytes_optimized = cur_pdf
        coverage            = info["coverage"]

        # ─────────────────────────────────────────────
        # 🎯 ALIGNMENT CALCULATION (NEW)
        # ─────────────────────────────────────────────
        ratio = float(coverage.get("ratio", 0.0))
        alignment_score = int(round(ratio * 100))

        matched_keywords = coverage.get("present", [])
        missing_keywords = coverage.get("missing", [])

        confidence_score = round(
            min(0.99, 0.5 + (ratio * 0.5)),
            2
        )

        verdict = (
            "Excellent Match" if alignment_score >= 80 else
            "Strong Match" if alignment_score >= 65 else
            "Good Match" if alignment_score >= 50 else
            "Needs Improvement"
        )

        # ─────────────────────────────────────────────
        # Save PDF
        # ─────────────────────────────────────────────
        paths    = build_output_paths(target_company, target_role)
        opt_path = paths["optimized"]
        saved_paths: List[str] = []

        if pdf_bytes_optimized:
            opt_path.parent.mkdir(parents=True, exist_ok=True)
            opt_path.write_bytes(pdf_bytes_optimized)
            saved_paths.append(str(opt_path))

        # ─────────────────────────────────────────────
        # Tech mapping
        # ─────────────────────────────────────────────
        generic_to_specific_mappings: Dict[str, str] = {}
        for keyword in info.get("all_keywords", [])[:10]:
            try:
                specific_tech = await get_specific_technology(
                    keyword, context="", block_index=0
                )
                generic_to_specific_mappings[keyword] = specific_tech
            except Exception:
                generic_to_specific_mappings[keyword] = keyword

        # ─────────────────────────────────────────────
        # FINAL RESPONSE
        # ─────────────────────────────────────────────
        return JSONResponse({

            # 🔥 Alignment block for Preview
            "alignment_score": alignment_score,
            "alignment_percent": f"{alignment_score}%",
            "matched_keywords_count": len(matched_keywords),
            "missing_keywords_count": len(missing_keywords),
            "confidence_score": confidence_score,
            "verdict": verdict,

            # Legacy compatibility
            "eligibility": {
                "score": ratio,
                "present": matched_keywords,
                "missing": missing_keywords,
                "total": coverage["total"],
                "verdict": verdict,
            },

            "company_name": target_company,
            "role": target_role,

            "optimized": {
                "tex": render_final_tex(optimized_tex_final),
                "pdf_b64": base64.b64encode(pdf_bytes_optimized).decode("ascii"),
                "filename": str(opt_path) if pdf_bytes_optimized else "",
            },

            "tex_string": render_final_tex(optimized_tex_final),
            "pdf_base64": base64.b64encode(pdf_bytes_optimized).decode("ascii"),

            "coverage_ratio": ratio,
            "coverage_present": matched_keywords,
            "coverage_missing": missing_keywords,

            "trim_summary": {
                "items_removed": trim_count,
                "final_pages": final_pages,
            },

            "technology_specificity": {
                "generic_to_specific_mappings": generic_to_specific_mappings,
                "specific_technologies_used": info.get("specific_technologies_used", []),
            },

            "skills_list": info.get("skills_list", []),
            "unique_numbers_used": info.get("unique_numbers_used", []),
        })

    except HTTPException:
        raise
    except Exception as e:
        log_event(f"💥 [PIPELINE] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))