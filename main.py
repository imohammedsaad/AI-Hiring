"""
Resume-Job Matching API
=======================
FastAPI backend for intelligent resume-to-job-description matching.
Uses a hybrid approach: semantic embeddings (60%) + skill matching (40%).

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
import pdfplumber
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
    version="2.1.0",
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

HF_EXPLAIN_MODEL = "google/flan-t5-small"
HF_EXPLAIN_URL = f"https://router.huggingface.co/hf-inference/models/{HF_EXPLAIN_MODEL}"

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
    """Simple word tokenizer — lowercase, alphanumeric tokens only."""
    return re.findall(r"[a-z0-9]+(?:[.+#][a-z0-9]+)*", text.lower())


def manual_tfidf_vectors(doc1: str, doc2: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Hand-rolled TF-IDF vectorizer (no scikit-learn).

    TF  = term count in doc / total terms in doc
    IDF = log(N / df)  where N = 2 (two documents), df = docs containing term
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

    # Document frequency (how many of the 2 docs contain the term)
    num_docs = 2
    df = {}
    for w in vocab:
        df[w] = (1 if w in tf1 else 0) + (1 if w in tf2 else 0)

    # Build TF-IDF vectors
    vec1 = np.zeros(vocab_size)
    vec2 = np.zeros(vocab_size)

    for w, idx in word_to_idx.items():
        idf = math.log(num_docs / df[w]) + 1  # smoothed IDF (add 1 like sklearn)
        vec1[idx] = (tf1.get(w, 0) / len1) * idf
        vec2[idx] = (tf2.get(w, 0) / len2) * idf

    return vec1, vec2


# ---------------------------------------------------------------------------
# Embedding helper — calls HuggingFace free Inference API
# ---------------------------------------------------------------------------
async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Get sentence embeddings from HuggingFace Inference API (free tier).
    Falls back to TF-IDF if the API is unavailable.
    """
    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
            headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    payload = {
                "inputs": texts,
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
# Skill bank — extend this as needed
# ---------------------------------------------------------------------------
SKILLS = [
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r",
    # Web & frameworks
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "spring boot", "express",
    # Data & ML
    "sql", "nosql", "mongodb", "postgresql", "mysql",
    "machine learning", "deep learning", "nlp",
    "tensorflow", "pytorch", "keras", "pandas", "numpy", "scikit-learn",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd",
    "jenkins", "github actions",
    # Salesforce-specific (relevant for your APEX integration)
    "salesforce", "apex", "soql", "lightning", "lwc", "visualforce",
    "salesforce admin", "salesforce developer",
    # General
    "git", "agile", "scrum", "rest api", "graphql", "microservices",
    "data analysis", "data engineering", "data science",
    "communication", "leadership", "problem solving",
]

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

def required_skill_score(candidate_skills, required_skills):
    if not required_skills:
        return 0.0

    matched = set(candidate_skills) & set(required_skills)
    return len(matched) / len(required_skills), list(matched)

def preferred_skill_score(candidate_skills, preferred_skills):
    if not preferred_skills:
        return 0.0

    matched = set(candidate_skills) & set(preferred_skills)
    return len(matched) / len(preferred_skills)

def experience_score(candidate_exp, required_exp):
    if not required_exp:
        return 1.0

    if candidate_exp >= required_exp:
        return 1.0

    return candidate_exp / required_exp

def keyword_score(resume: str, job: str) -> float:
    """Phase 2 — simple word-overlap ratio."""
    resume_words = set(resume.lower().split())
    job_words = set(job.lower().split())

    if not job_words:
        return 0.0

    matched = resume_words & job_words
    return len(matched) / len(job_words)


def tfidf_score(resume: str, job: str) -> float:
    """Phase 3 — TF-IDF cosine similarity (manual, no sklearn)."""
    try:
        vec1, vec2 = manual_tfidf_vectors(resume, job)
        return cosine_sim(vec1, vec2)
    except Exception:
        logger.warning("TF-IDF computation failed, returning 0")
        return 0.0


async def embedding_score(resume: str, job: str) -> float:
    """Phase 4 — semantic similarity via HuggingFace API embeddings."""
    embeddings = await get_embeddings([resume, job])

    if embeddings and len(embeddings) == 2:
        emb1 = np.array(embeddings[0])
        emb2 = np.array(embeddings[1])
        return cosine_sim(emb1, emb2)

    # Fallback: use TF-IDF score as approximation for semantic score
    logger.info("Using TF-IDF as fallback for semantic score")
    return tfidf_score(resume, job)

# ---------------------------------------------------------------------------
# Skill extraction & analysis
# ---------------------------------------------------------------------------

def normalize_to_list(skills):
    if not skills:
        return []
    
    if isinstance(skills, list):
        return [s.strip().lower() for s in skills if s]
    
    return [s.strip().lower() for s in skills.split(',') if s.strip()]

def extract_skills(text: str) -> list[str]:
    """Phase 5 — find known skills in text."""
    text_lower = text.lower()
    return [skill for skill in SKILLS if skill in text_lower]


def skill_analysis(resume: str, job: str) -> tuple[float, list[str], list[str]]:
    """Phase 7 — skill gap report."""
    resume_skills = set(extract_skills(resume))
    job_skills = set(extract_skills(job))

    matched = sorted(resume_skills & job_skills)
    missing = sorted(job_skills - resume_skills)

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
# EXPERIENCE ENGINE (REAL CORE)
# ---------------------------
def extract_explicit_experience(text):
    matches = re.findall(r'(\d+)\s*(year|yr)', text.lower())
    if matches:
        return max([int(m[0]) for m in matches])
    return None

def extract_project_count(text):
    return len(re.findall(r'project', text.lower()))

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

    # Infer from projects
    project_count = extract_project_count(text)

    # Infer from education
    grad_year = extract_education_year(text)

    # crude inference
    inferred_years = min(project_count * 0.3, 2)

    if grad_year:
        inferred_years += 0.5

    if inferred_years < 1:
        return {
            "years": round(inferred_years, 2),
            "label": "Fresher",
            "confidence": 0.7
        }
    else:
        return {
            "years": round(inferred_years, 2),
            "label": f"{round(inferred_years,1)} years (estimated)",
            "confidence": 0.6
        }

async def generate_explanation(resume, job, matched, missing, score):
    prompt = f"""
    Candidate resume: {resume[:200]}
    Job: {job[:200]}

    Matched skills: {matched}
    Missing skills: {missing}
    Score: {score}

    Give a short explanation (2 lines).
    """

    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    try:
        resp = await http_client.post(
            HF_EXPLAIN_URL,
            json={
                "inputs": prompt,
                "options": {"wait_for_model": True}
            },
            headers=headers,
            timeout=20.0
        )

        result = resp.json()

        # 🔥 Handle multiple formats safely
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "").strip()

        if isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"].strip()

        if isinstance(result, dict) and "error" in result:
            logger.warning(f"HF Error: {result['error']}")
            return fallback_explanation(matched, missing, score)

        return fallback_explanation(matched, missing, score)

    except Exception as e:
        logger.warning(f"Explanation failed: {e}")
        return fallback_explanation(matched, missing, score)

def fallback_explanation(matched, missing, score):
    if score > 0.75:
        return f"Strong fit. Matches key skills: {', '.join(matched)}."

    if score > 0.5:
        return f"Moderate fit. Has {', '.join(matched)} but lacks {', '.join(missing[:3])}."

    return f"Weak fit. Missing important skills like {', '.join(missing[:3])}."

# ---------------------------
# MAIN ENDPOINT
# ---------------------------
def extract_section(text, section_name):
    pattern = rf"{section_name}(.+?)(\n[A-Z][a-z]+|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""

@app.post("/parse_resume")
def parse_resume(req: ResumeRequest):
    file_bytes = base64.b64decode(req.file)

    text = extract_text(file_bytes)

    # --- Existing ---
    skills = extract_skills(text)
    experience = estimate_experience(text)

    # --- NEW: Section extraction ---
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

    candidate_skills = normalize_to_list(data.candidate_skills)
    # Semantic
    semantic = await embedding_score(data.resume, data.job)

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

    # Final score
    final = (
        0.4 * semantic +
        0.3 * req_score +
        0.2 * exp_score +
        0.1 * pref_score
    )
    # 🔥 Generate missing skills (better explanation)
    missing_skills = list(set(req_skills) - set(matched))

    # 🔥 Generate explanation
    explanation = await generate_explanation(
        data.resume,
        data.job,
        matched,
        missing_skills,
        final
    )

    # 🔥 Final response
    return {
        "final_score": round(final, 4),
        "semantic_score": round(semantic, 4),
        "required_skill_score": round(req_score, 4),
        "preferred_skill_score": round(pref_score, 4),
        "experience_score": round(exp_score, 4),
        "matched_skills": matched,
        "recommendation": (
            "Strong Match" if final > 0.75 else
            "Moderate Match" if final > 0.5 else
            "Low Match"
        ),
        "explanation": explanation
    }

@app.post("/predict_file")
async def predict_file(data: dict = Body(...)):

    try:
        file_base64 = data["file"]
        job = data["job"]

        file_bytes = base64.b64decode(file_base64)
        file_stream = BytesIO(file_bytes)

        text = ""
        with pdfplumber.open(file_stream) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

        # ---- YOUR EXISTING LOGIC ----
        semantic = await embedding_score(text, job)
        s_score, matched, missing = skill_analysis(text, job)
        tfidf = tfidf_score(text, job)
        kw = keyword_score(text, job)

        final = 0.6 * semantic + 0.4 * s_score

        if final > 0.75:
            rec = "Strong Match ✅"
        elif final > 0.50:
            rec = "Moderate Match ⚠️"
        else:
            rec = "Low Match ❌"

        return {
            "final_score": round(final, 4),
            "semantic_score": round(semantic, 4),
            "skill_score": round(s_score, 4),
            "matched_skills": matched,
            "missing_skills": missing,
            "recommendation": rec
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ---------------------------------------------------------------------------
# Standalone run (python main.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
