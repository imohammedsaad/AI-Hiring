"""
Resume-Job Matching API
=======================
FastAPI backend for intelligent resume-to-job-description matching.
Uses a hybrid approach: semantic embeddings + skill matching + experience
+ TF-IDF with proper NLP preprocessing.

Designed to be consumed by Salesforce APEX via HTTP callouts.

Optimized for Render.com free tier — zero heavy ML dependencies.
No PyTorch, no sentence-transformers, no scikit-learn.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi import UploadFile, File, Form
from fastapi import Body
import base64
import pdfplumber
from io import BytesIO
import numpy as np
import httpx
import os
import re
import math
import logging
import time
from collections import Counter
import docx
from typing import Union, List

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Resume-Job Matcher API",
    description="Hybrid NLP engine for resume screening — keyword, TF-IDF, "
                "semantic embeddings, and skill-gap analysis.",
    version="3.0.0",
)

# CORS — wide-open for dev; lock down origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# HuggingFace Inference API config (free, no model download needed)
# ---------------------------------------------------------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

# Upgraded: use a much more capable model for explanations
HF_EXPLAIN_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
HF_EXPLAIN_URL = f"https://router.huggingface.co/hf-inference/models/{HF_EXPLAIN_MODEL}"

# Fallback explanation model (lighter, more reliable)
HF_EXPLAIN_FALLBACK_MODEL = "google/flan-t5-base"
HF_EXPLAIN_FALLBACK_URL = f"https://router.huggingface.co/hf-inference/models/{HF_EXPLAIN_FALLBACK_MODEL}"

# Reusable async HTTP client — created once, reused across requests
http_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup_event():
    """Create the shared HTTP client on startup (lightweight, instant)."""
    global http_client
    http_client = httpx.AsyncClient(timeout=30.0)
    logger.info("App started ✓ — no heavy model to load!")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up the HTTP client."""
    global http_client
    if http_client:
        await http_client.aclose()


# ---------------------------------------------------------------------------
# Stopwords — English stopwords for proper TF-IDF (no NLTK dependency)
# ---------------------------------------------------------------------------
STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "get", "got", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't",
    "it", "its", "itself", "just", "let", "let's", "may", "me", "might",
    "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of",
    "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "per", "same", "shall", "shan't",
    "she", "should", "shouldn't", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "upon", "us", "very", "was", "wasn't", "we", "were",
    "weren't", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "won't", "would", "wouldn't", "you", "your",
    "yours", "yourself", "yourselves", "also", "etc", "e.g", "i.e",
    "well", "using", "used", "use", "including", "include", "includes",
    "able", "will", "within", "without",
})


# ---------------------------------------------------------------------------
# Pure-Python / NumPy math utilities (replaces scikit-learn)
# ---------------------------------------------------------------------------

def cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two vectors using NumPy."""
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def tokenize(text: str) -> list[str]:
    """
    Tokenizer that preserves multi-word technical terms.
    Lowercase, alphanumeric tokens, strips stopwords.
    """
    tokens = re.findall(r"[a-z0-9]+(?:[.+#/_-][a-z0-9]+)*", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def manual_tfidf_vectors(doc1: str, doc2: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Improved TF-IDF with BM25-inspired term frequency saturation.

    TF  = (count * (k1 + 1)) / (count + k1 * (1 - b + b * dl / avg_dl))
    IDF = log((N - df + 0.5) / (df + 0.5) + 1)

    This produces far more meaningful similarity scores than raw TF-IDF
    on short document pairs.
    """
    tokens1 = tokenize(doc1)
    tokens2 = tokenize(doc2)

    if not tokens1 or not tokens2:
        return np.array([0.0]), np.array([0.0])

    # Build vocabulary from both documents
    vocab = sorted(set(tokens1) | set(tokens2))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    # Term frequencies
    tf1 = Counter(tokens1)
    tf2 = Counter(tokens2)
    len1 = len(tokens1)
    len2 = len(tokens2)
    avg_dl = (len1 + len2) / 2.0

    # BM25 parameters
    k1 = 1.5
    b = 0.75
    num_docs = 2

    # Document frequency
    df = {}
    for w in vocab:
        df[w] = (1 if w in tf1 else 0) + (1 if w in tf2 else 0)

    # Build BM25-weighted vectors
    vec1 = np.zeros(vocab_size)
    vec2 = np.zeros(vocab_size)

    for w, idx in word_to_idx.items():
        idf = math.log((num_docs - df[w] + 0.5) / (df[w] + 0.5) + 1.0)

        # BM25-style term frequency saturation for doc1
        raw_tf1 = tf1.get(w, 0)
        vec1[idx] = (raw_tf1 * (k1 + 1)) / (raw_tf1 + k1 * (1 - b + b * len1 / avg_dl)) * idf

        # BM25-style term frequency saturation for doc2
        raw_tf2 = tf2.get(w, 0)
        vec2[idx] = (raw_tf2 * (k1 + 1)) / (raw_tf2 + k1 * (1 - b + b * len2 / avg_dl)) * idf

    return vec1, vec2


# ---------------------------------------------------------------------------
# Embedding helper — calls HuggingFace free Inference API
# ---------------------------------------------------------------------------
async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Get sentence embeddings from HuggingFace Inference API (free tier).
    Falls back to TF-IDF if the API is unavailable.

    Preprocessing: truncate to ~256 words to stay within model limits
    and reduce noise from overly long documents.
    """
    # Truncate to meaningful content (first ~256 words)
    truncated = []
    for t in texts:
        words = t.split()
        truncated.append(" ".join(words[:256]))

    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
            headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    payload = {
                "inputs": truncated,
                "options": {"wait_for_model": True},
            }

    try:
        resp = await http_client.post(HF_API_URL, json=payload, headers=headers)
        resp.raise_for_status()
        embeddings = resp.json()

        # The API returns a list of embeddings (list of lists of floats)
        if isinstance(embeddings, list) and len(embeddings) == len(texts):
            return embeddings

        logger.warning(f"Unexpected HF API response shape, falling back to TF-IDF")
        return []

    except Exception as e:
        logger.warning(f"HF Inference API call failed ({e}), falling back to TF-IDF")
        return []


# ---------------------------------------------------------------------------
# Comprehensive skill taxonomy with categories
# ---------------------------------------------------------------------------
SKILL_CATEGORIES = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go",
        "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "perl",
        "matlab", "dart", "lua", "haskell", "elixir", "clojure",
        "objective-c", "visual basic", "assembly", "fortran", "cobol",
        "groovy", "shell scripting", "bash", "powershell",
    ],
    "web_frontend": [
        "react", "angular", "vue", "svelte", "next.js", "nuxt.js",
        "gatsby", "html", "css", "sass", "less", "tailwind css",
        "bootstrap", "material ui", "jquery", "webpack", "vite",
        "redux", "mobx", "zustand", "react native", "flutter",
        "ionic", "electron", "pwa",
    ],
    "web_backend": [
        "node.js", "django", "flask", "fastapi", "spring boot",
        "express", "asp.net", "ruby on rails", "laravel", "nest.js",
        "gin", "fiber", "actix", "rocket", "phoenix",
        "graphql", "rest api", "grpc", "websocket", "oauth",
        "jwt", "microservices", "serverless",
    ],
    "databases": [
        "sql", "nosql", "mongodb", "postgresql", "mysql", "oracle",
        "sqlite", "redis", "cassandra", "dynamodb", "couchdb",
        "neo4j", "elasticsearch", "influxdb", "firebase",
        "supabase", "cockroachdb", "mariadb",
    ],
    "data_and_ml": [
        "machine learning", "deep learning", "nlp",
        "natural language processing", "computer vision",
        "tensorflow", "pytorch", "keras", "pandas", "numpy",
        "scikit-learn", "xgboost", "lightgbm", "catboost",
        "hugging face", "transformers", "llm", "large language models",
        "generative ai", "reinforcement learning",
        "data analysis", "data engineering", "data science",
        "data visualization", "tableau", "power bi", "looker",
        "apache spark", "hadoop", "airflow", "kafka",
        "etl", "data pipeline", "feature engineering",
        "model deployment", "mlops", "experiment tracking",
        "a/b testing", "statistical analysis", "regression",
        "classification", "clustering", "time series",
        "recommendation systems", "neural networks", "cnn", "rnn",
        "lstm", "gpt", "bert", "attention mechanism",
        "opencv", "spacy", "nltk",
    ],
    "cloud_and_devops": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
        "terraform", "ansible", "puppet", "chef",
        "ci/cd", "jenkins", "github actions", "gitlab ci",
        "circleci", "argocd", "helm", "istio",
        "cloudformation", "aws lambda", "ec2", "s3", "rds",
        "azure devops", "azure functions",
        "cloud run", "app engine", "bigquery",
        "linux", "nginx", "apache", "load balancing",
        "monitoring", "prometheus", "grafana", "datadog",
        "elk stack", "splunk", "new relic",
        "infrastructure as code", "site reliability engineering",
    ],
    "salesforce": [
        "salesforce", "apex", "soql", "sosl", "lightning",
        "lwc", "lightning web components", "visualforce",
        "salesforce admin", "salesforce developer",
        "salesforce cpq", "salesforce service cloud",
        "salesforce marketing cloud", "salesforce commerce cloud",
        "salesforce integration", "salesforce flow",
        "salesforce platform", "force.com",
        "mulesoft", "heroku", "salesforce dx",
    ],
    "security": [
        "cybersecurity", "penetration testing", "vulnerability assessment",
        "siem", "soc", "encryption", "ssl/tls", "firewall",
        "network security", "application security", "owasp",
        "identity management", "access control", "compliance",
        "gdpr", "hipaa", "iso 27001", "soc 2",
    ],
    "mobile": [
        "android", "ios", "react native", "flutter", "swift",
        "kotlin", "xamarin", "cordova", "mobile development",
    ],
    "tools_and_practices": [
        "git", "github", "gitlab", "bitbucket", "svn",
        "jira", "confluence", "trello", "asana",
        "agile", "scrum", "kanban", "waterfall",
        "tdd", "bdd", "unit testing", "integration testing",
        "selenium", "cypress", "jest", "pytest", "junit",
        "code review", "pair programming",
        "design patterns", "solid principles",
        "system design", "architecture",
    ],
    "soft_skills": [
        "communication", "leadership", "problem solving",
        "teamwork", "collaboration", "critical thinking",
        "project management", "time management",
        "presentation", "mentoring", "stakeholder management",
        "analytical thinking", "decision making",
        "adaptability", "creativity", "negotiation",
    ],
}

# Flatten into a single list for backward compatibility
SKILLS = []
for category_skills in SKILL_CATEGORIES.values():
    SKILLS.extend(category_skills)
SKILLS = sorted(set(SKILLS))

# Comprehensive synonym mapping
SKILL_SYNONYMS = {
    # AI/ML
    "ml": "machine learning",
    "dl": "deep learning",
    "artificial intelligence": "machine learning",
    "ai": "machine learning",
    "ai/ml": "machine learning",
    "gen ai": "generative ai",
    "genai": "generative ai",
    "large language model": "large language models",
    "llms": "large language models",
    "natural language understanding": "nlp",
    "nlu": "nlp",
    "convolutional neural network": "cnn",
    "recurrent neural network": "rnn",
    "random forest": "machine learning",
    "support vector machine": "machine learning",
    "svm": "machine learning",
    "gradient boosting": "xgboost",

    # Cloud
    "aws services": "aws",
    "amazon web services": "aws",
    "amazon s3": "s3",
    "amazon ec2": "ec2",
    "amazon rds": "rds",
    "microsoft azure": "azure",
    "google cloud platform": "gcp",
    "cloud architecture": "gcp",
    "cloud computing": "aws",
    "emr": "aws",

    # DevOps
    "devops": "ci/cd",
    "continuous integration": "ci/cd",
    "continuous deployment": "ci/cd",
    "container orchestration": "kubernetes",
    "containerization": "docker",
    "k8s": "kubernetes",
    "infrastructure": "infrastructure as code",
    "iac": "infrastructure as code",
    "sre": "site reliability engineering",

    # Web frameworks / variants
    "fast api": "fastapi",
    "nodejs": "node.js",
    "node": "node.js",
    "reactjs": "react",
    "react.js": "react",
    "angularjs": "angular",
    "angular.js": "angular",
    "vuejs": "vue",
    "vue.js": "vue",
    "nextjs": "next.js",
    "nuxtjs": "nuxt.js",
    "nestjs": "nest.js",
    "expressjs": "express",
    "ruby on rails": "ruby on rails",
    "ror": "ruby on rails",
    "springboot": "spring boot",
    "asp.net core": "asp.net",
    "dotnet": "asp.net",
    ".net": "asp.net",
    ".net core": "asp.net",

    # Databases
    "postgres": "postgresql",
    "mongo": "mongodb",
    "mssql": "sql",
    "sql server": "sql",
    "t-sql": "sql",
    "pl/sql": "sql",
    "amazon dynamodb": "dynamodb",

    # Salesforce
    "sfdc": "salesforce",
    "sf": "salesforce",
    "lightning components": "lwc",
    "lightning web component": "lwc",
    "aura components": "lightning",

    # Testing
    "test driven development": "tdd",
    "behavior driven development": "bdd",
    "automated testing": "unit testing",
    "qa automation": "selenium",

    # Data
    "big data": "apache spark",
    "data warehousing": "etl",
    "data warehouse": "etl",
    "business intelligence": "data visualization",
    "bi": "data visualization",
    "data analytics": "data analysis",

    # Mobile
    "android development": "android",
    "ios development": "ios",
    "mobile app development": "mobile development",
    "cross platform": "react native",

    # General
    "version control": "git",
    "source control": "git",
    "project management": "project management",
    "pmp": "project management",
    "product management": "project management",
    "scrum master": "scrum",
    "agile methodology": "agile",
}

# Multi-word skills sorted by length (longest first) for greedy matching
_MULTI_WORD_SKILLS = sorted(
    [s for s in SKILLS if " " in s or "." in s or "/" in s],
    key=len,
    reverse=True,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class InputData(BaseModel):
    resume: str
    candidate_skills: Union[str, List[str]]
    candidate_experience: float
    job: str
    required_skills: Union[str, List[str]]
    preferred_skills: Union[str, List[str]]
    required_experience: float

class MatchResponse(BaseModel):
    final_score: float
    semantic_score: float
    tfidf_score: float
    keyword_score: float
    skill_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    recommendation: str
    processing_time_ms: float

# ---------------------------------------------------------------------------
# Scoring engines
# ---------------------------------------------------------------------------
def parse_skills(skill_text):
    if not skill_text:
        return []
    return [s.strip().lower() for s in skill_text.split(',') if s.strip()]


def smart_match(candidate_skills: list[str], required_skills: list[str]) -> list[str]:
    """
    Improved skill matching with fuzzy tolerance.
    Avoids false positives like "r" matching "react".
    """
    matched = []
    for r in required_skills:
        r_clean = r.strip().lower()
        for c in candidate_skills:
            c_clean = c.strip().lower()
            # Exact match
            if r_clean == c_clean:
                matched.append(r)
                break
            # Synonym-resolved match
            r_resolved = SKILL_SYNONYMS.get(r_clean, r_clean)
            c_resolved = SKILL_SYNONYMS.get(c_clean, c_clean)
            if r_resolved == c_resolved:
                matched.append(r)
                break
            # Substring match — only for multi-word skills (min 3 chars)
            # to avoid false positives like "r" in "react"
            if len(r_clean) >= 3 and len(c_clean) >= 3:
                if r_clean in c_clean or c_clean in r_clean:
                    matched.append(r)
                    break

    return list(set(matched))


def required_skill_score(candidate_skills, required_skills):
    if not required_skills:
        return 0.0, []

    matched = smart_match(candidate_skills, required_skills)

    return len(matched) / len(required_skills), matched

def preferred_skill_score(candidate_skills, preferred_skills):
    if not preferred_skills:
        return 0.0

    matched = smart_match(candidate_skills, preferred_skills)
    return len(matched) / len(preferred_skills)

def experience_score(candidate_exp, required_exp):
    if not required_exp:
        return 1.0

    if candidate_exp >= required_exp:
        return 1.0
    
    # Partial credit with diminishing penalty
    ratio = candidate_exp / required_exp
    # Candidates close to required experience get more credit
    return min(ratio * 1.1, 1.0)  # slight boost, capped at 1.0


def keyword_score(resume: str, job: str) -> float:
    """Improved word-overlap with stopword filtering."""
    resume_tokens = set(tokenize(resume))
    job_tokens = set(tokenize(job))

    if not job_tokens:
        return 0.0

    matched = resume_tokens & job_tokens
    return len(matched) / len(job_tokens)


def tfidf_score(resume: str, job: str) -> float:
    """TF-IDF cosine similarity with BM25-inspired weighting."""
    try:
        vec1, vec2 = manual_tfidf_vectors(resume, job)
        return cosine_sim(vec1, vec2)
    except Exception:
        logger.warning("TF-IDF computation failed, returning 0")
        return 0.0


async def embedding_score(resume: str, job: str) -> float:
    """Semantic similarity via HuggingFace API embeddings."""
    embeddings = await get_embeddings([resume, job])

    if embeddings and len(embeddings) == 2:
        emb1 = np.array(embeddings[0])
        emb2 = np.array(embeddings[1])
        return cosine_sim(emb1, emb2)

    # Fallback: use TF-IDF score as approximation for semantic score
    logger.info("Using TF-IDF as fallback for semantic score")
    return tfidf_score(resume, job)

# ---------------------------------------------------------------------------
# Skill extraction & analysis — improved with multi-word awareness
# ---------------------------------------------------------------------------

def normalize_to_list(skills):
    if not skills:
        return []

    if isinstance(skills, list):
        raw = skills
    else:
        raw = skills.split(',')

    normalized = []

    for s in raw:
        clean = s.strip().lower()
        if not clean:
            continue

        # synonym mapping
        if clean in SKILL_SYNONYMS:
            clean = SKILL_SYNONYMS[clean]

        normalized.append(clean)

    return list(set(normalized))


def extract_skills(text: str) -> list[str]:
    """
    Improved skill extraction: match multi-word phrases first (greedy),
    then single-word skills, with word-boundary awareness.
    """
    text_lower = text.lower()
    found = set()

    # Phase 1: Multi-word skills (greedy, longest first)
    for skill in _MULTI_WORD_SKILLS:
        if skill in text_lower:
            found.add(skill)

    # Phase 2: Single-word skills with word-boundary check
    for skill in SKILLS:
        if " " in skill or "." in skill or "/" in skill:
            continue  # already handled above
        # Word boundary: avoid "r" matching inside "react"
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.add(skill)

    return sorted(found)


def get_skill_category(skill: str) -> str:
    """Return the category a skill belongs to."""
    for category, skills_list in SKILL_CATEGORIES.items():
        if skill in skills_list:
            return category.replace("_", " ").title()
    return "Other"


def skill_analysis(resume: str, job: str) -> tuple[float, list[str], list[str]]:
    """Skill gap report with improved matching."""
    resume_skills = set(extract_skills(resume))
    job_skills = set(extract_skills(job))

    # Also check synonym-resolved versions
    resume_resolved = set()
    for s in resume_skills:
        resume_resolved.add(SKILL_SYNONYMS.get(s, s))
    job_resolved = set()
    for s in job_skills:
        job_resolved.add(SKILL_SYNONYMS.get(s, s))

    matched_resolved = resume_resolved & job_resolved
    missing_resolved = job_resolved - resume_resolved

    # Map back to original skill names for readability
    matched = sorted([s for s in job_skills if SKILL_SYNONYMS.get(s, s) in matched_resolved])
    missing = sorted([s for s in job_skills if SKILL_SYNONYMS.get(s, s) in missing_resolved])

    score = len(matched) / len(job_skills) if job_skills else 0.0
    return score, matched, missing


class ResumeRequest(BaseModel):
    file: str


# ---------------------------
# FILE PARSER
# ---------------------------
def extract_text(file_bytes):
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            return "\n".join([p.extract_text() or "" for p in pdf.pages])
    except:
        try:
            doc = docx.Document(BytesIO(file_bytes))
            return "\n".join([p.text for p in doc.paragraphs])
        except:
            return ""


# ---------------------------
# EXPERIENCE ENGINE (IMPROVED)
# ---------------------------
def extract_explicit_experience(text: str):
    """Extract explicitly stated years of experience with multiple patterns."""
    patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)',
        r'(?:experience|exp)\s*(?:of)?\s*(\d+)\+?\s*(?:years?|yrs?)',
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:in|of|working)',
        r'over\s+(\d+)\s+(?:years?|yrs?)',
    ]
    all_matches = []
    text_lower = text.lower()
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        all_matches.extend([int(m) for m in matches])

    if all_matches:
        return max(all_matches)
    return None


def extract_date_ranges(text: str) -> list[tuple[int, int]]:
    """Extract year ranges like '2019 - 2022' or '2019 – Present'."""
    current_year = 2026
    ranges = []

    # Pattern: YYYY - YYYY or YYYY – YYYY
    pattern1 = re.findall(r'(20\d{2})\s*[-–—]\s*(20\d{2})', text)
    for start, end in pattern1:
        ranges.append((int(start), int(end)))

    # Pattern: YYYY - Present / Current
    pattern2 = re.findall(r'(20\d{2})\s*[-–—]\s*(?:present|current|now|ongoing)', text.lower())
    for start in pattern2:
        ranges.append((int(start), current_year))

    return ranges


def extract_project_count(text):
    """Count projects more carefully — look for project headers, not just the word."""
    # Look for patterns like "Project:", "Project 1", numbered projects
    project_patterns = [
        r'(?:^|\n)\s*(?:project\s*[:#\d])',
        r'(?:^|\n)\s*(?:\d+\.\s+\w+.*?project)',
        r'(?:^|\n)\s*[•\-]\s*(?:\w+.*?project)',
    ]
    count = 0
    for pattern in project_patterns:
        count += len(re.findall(pattern, text.lower()))

    # Fallback: count explicit "project" mentions but cap at reasonable number
    if count == 0:
        raw_count = len(re.findall(r'\bproject\b', text.lower()))
        count = min(raw_count, 8)  # cap to avoid inflated counts

    return count


def extract_education_year(text):
    matches = re.findall(r'(20\d{2})\s*[-–]\s*(20\d{2})', text)
    if matches:
        return int(matches[0][1])
    return None


def estimate_experience(text):
    explicit = extract_explicit_experience(text)

    if explicit:
        return {
            "years": explicit,
            "label": f"{explicit} years",
            "confidence": 0.95
        }

    # Attempt to infer from date ranges in work experience sections
    date_ranges = extract_date_ranges(text)
    if date_ranges:
        # Calculate total non-overlapping years
        total_years = sum(end - start for start, end in date_ranges)
        # Cap at reasonable maximum
        total_years = min(total_years, 30)
        if total_years > 0:
            return {
                "years": total_years,
                "label": f"{total_years} years (from date ranges)",
                "confidence": 0.85
            }

    # Infer from projects and education
    project_count = extract_project_count(text)
    grad_year = extract_education_year(text)

    inferred_years = min(project_count * 0.4, 3)

    if grad_year:
        years_since_grad = max(0, 2026 - grad_year)
        inferred_years = max(inferred_years, min(years_since_grad, 5))

    if inferred_years < 1:
        return {
            "years": round(inferred_years, 2),
            "label": "Fresher",
            "confidence": 0.7
        }
    else:
        return {
            "years": round(inferred_years, 2),
            "label": f"{round(inferred_years, 1)} years (estimated)",
            "confidence": 0.6
        }


# ---------------------------------------------------------------------------
# Explanation generation — upgraded for professional quality
# ---------------------------------------------------------------------------

async def generate_explanation(resume, job, matched, missing, score):
    """
    Generate a professional, context-aware explanation using Mistral-7B.
    Falls back to flan-t5-base, then to a structured template.
    """
    matched_str = ", ".join(matched[:10]) if matched else "none identified"
    missing_str = ", ".join(missing[:8]) if missing else "none"

    if score > 0.75:
        fit_level = "strong"
    elif score > 0.50:
        fit_level = "moderate"
    else:
        fit_level = "weak"

    prompt = (
        f"<s>[INST] You are a professional HR analyst writing a brief candidate evaluation. "
        f"Write exactly 2-3 concise sentences evaluating this candidate's fit.\n\n"
        f"Overall fit level: {fit_level} (score: {round(score * 100)}%)\n"
        f"Key matching skills: {matched_str}\n"
        f"Missing skills: {missing_str}\n"
        f"Resume summary: {resume[:300]}\n"
        f"Job requirements: {job[:300]}\n\n"
        f"Write a professional evaluation in 2-3 sentences. "
        f"Be specific about strengths and gaps. Do not use bullet points. [/INST]"
    )

    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    # Attempt 1: Mistral-7B-Instruct
    try:
        resp = await http_client.post(
            HF_EXPLAIN_URL,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 150,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "return_full_text": False,
                    "do_sample": True,
                },
                "options": {"wait_for_model": True},
            },
            headers=headers,
            timeout=25.0,
        )
        if resp.status_code == 200:
            result = resp.json()
            text = _extract_generated_text(result)
            if text and len(text) > 30:
                return _clean_explanation(text)
    except Exception as e:
        logger.warning(f"Mistral explanation failed: {e}")

    # Attempt 2: flan-t5-base (lighter, more reliable)
    fallback_prompt = (
        f"Write a professional 2-sentence evaluation of a job candidate. "
        f"Fit: {fit_level}. Matching skills: {matched_str}. "
        f"Missing skills: {missing_str}. Score: {round(score * 100)}%."
    )
    try:
        resp = await http_client.post(
            HF_EXPLAIN_FALLBACK_URL,
            json={
                "inputs": fallback_prompt,
                "parameters": {"max_new_tokens": 120},
                "options": {"wait_for_model": True},
            },
            headers=headers,
            timeout=20.0,
        )
        if resp.status_code == 200:
            result = resp.json()
            text = _extract_generated_text(result)
            if text and len(text) > 20:
                return _clean_explanation(text)
    except Exception as e:
        logger.warning(f"Fallback explanation also failed: {e}")

    # Attempt 3: High-quality structured template
    return fallback_explanation(matched, missing, score)


def _extract_generated_text(result) -> str:
    """Safely extract generated text from various HF API response formats."""
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], dict):
            return result[0].get("generated_text", "").strip()
        if isinstance(result[0], str):
            return result[0].strip()
    if isinstance(result, dict) and "generated_text" in result:
        return result["generated_text"].strip()
    return ""


def _clean_explanation(text: str) -> str:
    """Clean and validate generated explanation text."""
    # Remove common artifacts
    text = re.sub(r'\[/?INST\]', '', text)
    text = re.sub(r'</?s>', '', text)
    text = text.strip()

    # Truncate to ~3 sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 3:
        text = " ".join(sentences[:3])

    # Ensure it ends with punctuation
    if text and text[-1] not in '.!?':
        text += '.'

    return text


def fallback_explanation(matched, missing, score):
    """
    Professional template-based explanation as final fallback.
    Context-aware with varied language.
    """
    matched_str = ", ".join(matched[:6]) if matched else None
    missing_str = ", ".join(missing[:5]) if missing else None
    pct = round(score * 100)

    if score > 0.80:
        opening = f"This candidate demonstrates excellent alignment ({pct}% match) with the role requirements."
        if matched_str:
            detail = f" Core competencies including {matched_str} directly address the position's key demands."
        else:
            detail = " The candidate's overall profile strongly aligns with the job description."
        if missing_str:
            closing = f" Minor gaps in {missing_str} could be addressed through onboarding or targeted development."
        else:
            closing = " No significant skill gaps were identified."
    elif score > 0.60:
        opening = f"The candidate shows solid potential ({pct}% match) with notable strengths in relevant areas."
        if matched_str:
            detail = f" Demonstrated proficiency in {matched_str} provides a solid foundation for this role."
        else:
            detail = ""
        if missing_str:
            closing = f" However, gaps in {missing_str} represent areas requiring further evaluation or upskilling."
        else:
            closing = " Overall skill coverage is adequate for consideration."
    elif score > 0.40:
        opening = f"This candidate presents a partial match ({pct}%) with the stated requirements."
        if matched_str:
            detail = f" While {matched_str} are relevant, they only partially cover the role's technical needs."
        else:
            detail = " Limited overlap was found between the candidate's profile and role requirements."
        if missing_str:
            closing = f" Key gaps in {missing_str} would need to be addressed through training or additional hiring criteria."
        else:
            closing = ""
    else:
        opening = f"The candidate's profile shows limited alignment ({pct}%) with this role's requirements."
        if missing_str:
            detail = f" Critical gaps were identified in {missing_str}, which are fundamental to the position."
        else:
            detail = " The candidate's skill set does not substantially overlap with the stated job requirements."
        closing = " Recommend considering for alternative roles or revisiting if requirements are flexible."

    return (opening + detail + closing).strip()


# ---------------------------
# SECTION EXTRACTION (Improved)
# ---------------------------
_SECTION_HEADERS = {
    "education": [
        r"education", r"academic", r"qualifications", r"degrees?",
    ],
    "projects": [
        r"projects?", r"portfolio", r"work samples?",
    ],
    "certifications": [
        r"certifications?", r"certificates?", r"accreditations?",
        r"licenses?", r"professional development",
    ],
    "experience": [
        r"(?:work\s+)?experience", r"employment\s+history",
        r"professional\s+experience", r"career\s+history",
    ],
    "skills": [
        r"(?:technical\s+)?skills?", r"competenc(?:ies|e)",
        r"technologies", r"tools?\s*(?:&|and)\s*technologies",
    ],
    "summary": [
        r"summary", r"objective", r"profile", r"about\s+me",
        r"professional\s+summary",
    ],
}


def extract_section(text: str, section_name: str) -> str:
    """
    Improved section extraction using multiple header patterns
    and smart boundary detection.
    """
    if not text:
        return ""

    section_name_lower = section_name.lower()
    patterns = _SECTION_HEADERS.get(section_name_lower, [section_name_lower])

    # Build all-sections pattern for boundary detection
    all_headers = []
    for headers in _SECTION_HEADERS.values():
        all_headers.extend(headers)
    boundary_pattern = "|".join(all_headers)

    for pattern in patterns:
        # Look for section header (usually on its own line, possibly with decorators)
        header_regex = (
            r'(?:^|\n)\s*'
            r'(?:[#*\-=_\s]*)'     # optional decorators
            rf'({pattern})'
            r'(?:\s*[:\-–—])?'     # optional colon/dash
            r'\s*\n'
        )

        match = re.search(header_regex, text, re.IGNORECASE)
        if match:
            start = match.end()
            # Find the next section header
            rest = text[start:]
            next_section = re.search(
                rf'(?:^|\n)\s*(?:[#*\-=_\s]*)(?:{boundary_pattern})(?:\s*[:\-–—])?\s*\n',
                rest,
                re.IGNORECASE,
            )
            if next_section:
                content = rest[:next_section.start()]
            else:
                content = rest

            content = content.strip()
            if content:
                return content[:2000]  # cap at 2000 chars

    return ""


# ---------------------------
# MAIN ENDPOINTS
# ---------------------------
@app.post("/parse_resume")
def parse_resume(req: ResumeRequest):
    file_bytes = base64.b64decode(req.file)

    text = extract_text(file_bytes)

    # --- Existing ---
    skills = extract_skills(text)
    experience = estimate_experience(text)

    # --- Section extraction ---
    education = extract_section(text, "Education")
    projects = extract_section(text, "Projects")
    certifications = extract_section(text, "Certifications")

    return {
        "text": text[:5000],
        "skills": skills,
        "experience": experience,
        "sections": {
            "education": education,
            "projects": projects,
            "certifications": certifications
        }
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def home():
    """Health-check endpoint."""
    return {"status": "ok", "message": "Resume-Job Matcher API is running 🚀"}


@app.post("/predict")
async def predict(data: InputData):

    start_time = time.time()

    candidate_skills = list(set(
    normalize_to_list(data.candidate_skills) +
    extract_skills(data.resume)
    ))
    # Semantic
    semantic = await embedding_score(data.resume, data.job)

    # TF-IDF (supplementary signal)
    tfidf = tfidf_score(data.resume, data.job)

    # Skills
    req_skills = normalize_to_list(data.required_skills)
    pref_skills = normalize_to_list(data.preferred_skills)
   
    req_score, matched = required_skill_score(
        candidate_skills, req_skills
    )

    pref_score = preferred_skill_score(
        candidate_skills, pref_skills
    )

    # Experience
    exp_score = experience_score(
        data.candidate_experience,
        data.required_experience
    )

    # Blended semantic score (embedding + tfidf for robustness)
    blended_semantic = 0.75 * semantic + 0.25 * tfidf

    # Final score — balanced multi-signal formula
    final = (
        0.30 * blended_semantic +
        0.35 * req_score +
        0.20 * exp_score +
        0.10 * pref_score +
        0.05 * keyword_score(data.resume, data.job)
    )

    # Generate missing skills
    missing_skills = list(set(req_skills) - set(matched))

    # Generate explanation
    explanation = await generate_explanation(
        data.resume,
        data.job,
        matched,
        missing_skills,
        final
    )

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    # Granular recommendation
    recommendation = _get_recommendation(final)

    # Final response
    return {
        "final_score": round(final, 4),
        "semantic_score": round(semantic, 4),
        "required_skill_score": round(req_score, 4),
        "preferred_skill_score": round(pref_score, 4),
        "experience_score": round(exp_score, 4),
        "matched_skills": matched,
        "recommendation": recommendation,
        "explanation": explanation,
    }


@app.post("/predict_file")
async def predict_file(data: dict = Body(...)):

    try:
        start_time = time.time()
        file_base64 = data["file"]
        job = data["job"]

        file_bytes = base64.b64decode(file_base64)
        file_stream = BytesIO(file_bytes)

        text = ""
        with pdfplumber.open(file_stream) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

        # Semantic embedding score
        semantic = await embedding_score(text, job)

        # Skill analysis
        s_score, matched, missing = skill_analysis(text, job)

        # TF-IDF and keyword (supplementary)
        tfidf = tfidf_score(text, job)
        kw = keyword_score(text, job)

        # Blended semantic
        blended_semantic = 0.75 * semantic + 0.25 * tfidf

        # Final — consistent formula with /predict
        final = (
            0.30 * blended_semantic +
            0.35 * s_score +
            0.20 * 1.0 +          # experience not available in this endpoint
            0.10 * 0.0 +          # preferred skills not available
            0.05 * kw
        )

        recommendation = _get_recommendation(final)

        # Generate explanation
        explanation = await generate_explanation(text, job, matched, missing, final)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "final_score": round(final, 4),
            "semantic_score": round(semantic, 4),
            "skill_score": round(s_score, 4),
            "matched_skills": matched,
            "recommendation": recommendation,
            "explanation": explanation,
        }

    except Exception as e:
        logger.error(f"predict_file error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _get_recommendation(score: float) -> str:
    """Granular 5-level recommendation with emoji indicators."""
    if score > 0.85:
        return "Excellent Match 🌟"
    elif score > 0.70:
        return "Strong Match ✅"
    elif score > 0.55:
        return "Moderate Match ⚠️"
    elif score > 0.35:
        return "Below Average Match 📉"
    else:
        return "Low Match ❌"


# ---------------------------------------------------------------------------
# Standalone run (python main.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
