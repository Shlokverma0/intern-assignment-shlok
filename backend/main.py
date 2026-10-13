"""
FastAPI backend for the question generation + duplicate detection system.

Endpoints:
    GET  /api/corpus        - return the Q1-Q10 test corpus
    POST /api/generate      - generate a new question and verify it
    POST /api/detect        - compare two questions
    POST /api/evaluate      - run the detector over Q1 vs all

The API is intentionally stateless: every request loads what it needs.
Corpus and embeddings are cached at module level so repeated requests
don't pay the load cost twice.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.detector import classify
from src.embeddings import embed_batch, embed_text
from src.generator import generate_question, select_diverse_context

from backend.schemas import (
    ComparisonRow,
    CorpusItem,
    DetectRequest,
    DetectResponse,
    EvaluationResponse,
    EvaluationRow,
    GenerateRequest,
    GenerateResponse,
    Metrics,
)


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "questions.csv"
FRONTEND_DIR = ROOT / "frontend"

DUPLICATE_IDS = {"Q2", "Q3", "Q4", "Q10"}
SKIP_IDS = {"Q8"}  # Q8 is marked ambiguous in the assignment


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Question Generator API",
    description="Generate new questions with generation-time avoidance and "
                "independent duplicate detection.",
    version="1.0.0",
)

# Allow the static frontend (served from the same app) and any local
# development origin to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Cached corpus + embeddings
# ---------------------------------------------------------------------------

_corpus: pd.DataFrame | None = None
_embeddings = None


def _get_corpus() -> pd.DataFrame:
    """Load the corpus once and reuse it across requests."""
    global _corpus
    if _corpus is None:
        if not CORPUS_PATH.exists():
            raise HTTPException(status_code=500, detail=f"Corpus not found at {CORPUS_PATH}")
        _corpus = pd.read_csv(CORPUS_PATH)
    return _corpus


def _get_embeddings():
    """Compute embeddings for the corpus once and cache them."""
    global _embeddings
    if _embeddings is None:
        df = _get_corpus()
        _embeddings = embed_batch(df["question"].tolist())
    return _embeddings


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    """Simple health check."""
    return {"status": "ok"}


@app.get("/api/corpus", response_model=list[CorpusItem])
def get_corpus() -> list[CorpusItem]:
    """Return the Q1-Q10 test corpus."""
    df = _get_corpus()
    return [
        CorpusItem(id=row["id"], question=row["question"], expected=row.get("expected"))
        for _, row in df.iterrows()
    ]


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """
    Generate a new question on the given topic using the LLM, then verify
    the result against the existing corpus.
    """
    df = _get_corpus()
    embs = _get_embeddings()

    # 1. Select a diverse context to send to the LLM.
    context = select_diverse_context(df, embs, k=req.context_size)

    # 2. Generate a new question.
    try:
        new_question = generate_question(req.topic, context)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    # 3. Verify against every existing question.
    new_emb = embed_text(new_question)
    comparisons: list[ComparisonRow] = []
    is_dup = False

    for i in range(len(df)):
        result = classify(new_question, df.iloc[i]["question"], new_emb, embs[i])
        comparisons.append(
            ComparisonRow(
                existing_id=df.iloc[i]["id"],
                existing_question=df.iloc[i]["question"],
                label=result["label"],
                exact_score=result["exact_score"],
                semantic_score=result["semantic_score"],
            )
        )
        if result["label"] != "DIFFERENT":
            is_dup = True

    return GenerateResponse(
        generated_question=new_question,
        context_used=context,
        comparisons=comparisons,
        is_duplicate=is_dup,
    )


@app.post("/api/detect", response_model=DetectResponse)
def detect(req: DetectRequest) -> DetectResponse:
    """Compare two questions and classify the pair."""
    emb_a = embed_text(req.question_a)
    emb_b = embed_text(req.question_b)
    result = classify(req.question_a, req.question_b, emb_a, emb_b)
    return DetectResponse(**result)


@app.post("/api/evaluate", response_model=EvaluationResponse)
def evaluate() -> EvaluationResponse:
    """
    Run the detector over Q1 vs every other question and return both the
    per-pair rows and the aggregate metrics.
    """
    df = _get_corpus()
    embs = _get_embeddings()

    q1_text = df.iloc[0]["question"]
    q1_emb = embs[0]

    rows: list[EvaluationRow] = []
    tp = fp = fn = tn = 0

    for i in range(1, len(df)):
        result = classify(q1_text, df.iloc[i]["question"], q1_emb, embs[i])
        rows.append(
            EvaluationRow(
                comparison=f"Q1 vs {df.iloc[i]['id']}",
                question=df.iloc[i]["question"],
                expected=df.iloc[i].get("expected", ""),
                detector_says=result["label"],
                semantic_score=result["semantic_score"],
            )
        )

        qid = df.iloc[i]["id"]
        if qid in SKIP_IDS:
            continue

        predicted_dup = result["label"] in ("EXACT_DUPLICATE", "SEMANTIC_DUPLICATE")
        actual_dup = qid in DUPLICATE_IDS

        if predicted_dup and actual_dup:
            tp += 1
        elif predicted_dup and not actual_dup:
            fp += 1
        elif not predicted_dup and actual_dup:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return EvaluationResponse(
        rows=rows,
        metrics=Metrics(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            precision=round(precision, 3),
            recall=round(recall, 3),
            f1=round(f1, 3),
        ),
    )


# ---------------------------------------------------------------------------
# Serve the frontend
# ---------------------------------------------------------------------------

# Mount the frontend directory at /app so index.html is served there.
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")