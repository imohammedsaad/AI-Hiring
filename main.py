"""
Resume-Job Matching API
=======================
FastAPI backend for intelligent resume-to-job-description matching.
Uses a hybrid approach: semantic embeddings (60%) + skill matching (40%).

Designed to be consumed by Salesforce APEX via HTTP callouts.

Optimized for Render.com free tier — uses lightweight dependencies only.
No PyTorch / sentence-transformers required.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import httpx
import os
import logging
import time

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
    version="2.0.0",
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
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

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
# Embedding helper — calls HuggingFace free Inference API
# ---------------------------------------------------------------------------
async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Get sentence embeddings from HuggingFace Inference API (free tier).
    Falls back to TF-IDF if the API is unavailable.
    """
    headers = {}
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
    resume: str = Field(..., min_length=1, description="Full resume text")
    job: str = Field(..., min_length=1, description="Job description text")


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

def keyword_score(resume: str, job: str) -> float:
    """Phase 2 — simple word-overlap ratio."""
    resume_words = set(resume.lower().split())
    job_words = set(job.lower().split())

    if not job_words:
        return 0.0

    matched = resume_words & job_words
    return len(matched) / len(job_words)


def tfidf_score(resume: str, job: str) -> float:
    """Phase 3 — TF-IDF cosine similarity."""
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([resume, job])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(similarity[0][0])
    except Exception:
        logger.warning("TF-IDF computation failed, returning 0")
        return 0.0


async def embedding_score(resume: str, job: str) -> float:
    """Phase 4 — semantic similarity via HuggingFace API embeddings."""
    embeddings = await get_embeddings([resume, job])

    if embeddings and len(embeddings) == 2:
        emb1 = np.array(embeddings[0])
        emb2 = np.array(embeddings[1])
        sim = cosine_similarity([emb1], [emb2])
        return float(sim[0][0])

    # Fallback: use TF-IDF score as approximation for semantic score
    logger.info("Using TF-IDF as fallback for semantic score")
    return tfidf_score(resume, job)

# ---------------------------------------------------------------------------
# Skill extraction & analysis
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def home():
    """Health-check endpoint."""
    return {"status": "ok", "message": "Resume-Job Matcher API is running 🚀"}


@app.post("/predict", response_model=MatchResponse, tags=["Matching"])
async def predict(data: InputData):
    """
    Main matching endpoint.

    Combines:
    - Semantic similarity (60 % weight)
    - Skill-gap matching (40 % weight)

    Also returns TF-IDF and keyword scores for reference.
    """
    start = time.perf_counter()

    try:
        # --- Core scores ---
        semantic = await embedding_score(data.resume, data.job)
        s_score, matched, missing = skill_analysis(data.resume, data.job)

        # --- Supplementary scores ---
        tfidf = tfidf_score(data.resume, data.job)
        kw = keyword_score(data.resume, data.job)

        # --- Weighted final score ---
        final = 0.6 * semantic + 0.4 * s_score

        # --- Recommendation ---
        if final > 0.75:
            rec = "Strong Match ✅"
        elif final > 0.50:
            rec = "Moderate Match ⚠️"
        else:
            rec = "Low Match ❌"

        elapsed_ms = (time.perf_counter() - start) * 1000

        return MatchResponse(
            final_score=round(final, 4),
            semantic_score=round(semantic, 4),
            tfidf_score=round(tfidf, 4),
            keyword_score=round(kw, 4),
            skill_score=round(s_score, 4),
            matched_skills=matched,
            missing_skills=missing,
            recommendation=rec,
            processing_time_ms=round(elapsed_ms, 2),
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ---------------------------------------------------------------------------
# Standalone run (python main.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
