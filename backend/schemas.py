"""
Pydantic schemas for the FastAPI endpoints.

Defines request and response models for the API. Keeping these separate
from the endpoint logic makes the API self-documenting and gives us
automatic validation for free.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request body for /api/generate."""

    topic: str = Field(..., min_length=2, max_length=200)
    context_size: int = Field(5, ge=1, le=20)


class ComparisonRow(BaseModel):
    """One row in the detector comparison table."""

    existing_id: str
    existing_question: str
    tags: Optional[str] = None
    label: str
    exact_score: float
    semantic_score: float


class GenerateResponse(BaseModel):
    """Response body for /api/generate."""

    generated_question: str
    context_used: List[str]
    comparisons: List[ComparisonRow]
    is_duplicate: bool


class DetectRequest(BaseModel):
    """Request body for /api/detect."""

    question_a: str = Field(..., min_length=2)
    question_b: str = Field(..., min_length=2)


class DetectResponse(BaseModel):
    """Response body for /api/detect."""

    label: str
    exact_score: float
    semantic_score: float


class CorpusItem(BaseModel):
    """A single question in the corpus."""

    id: str
    question: str
    tags: Optional[str] = None
    expected: Optional[str] = None


class EvaluationRow(BaseModel):
    """One row in the evaluation table."""

    comparison: str
    question: str
    tags: Optional[str] = None
    expected: str
    detector_says: str
    semantic_score: float


class Metrics(BaseModel):
    """Aggregate metrics."""

    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float


class EvaluationResponse(BaseModel):
    """Response body for /api/evaluate."""

    rows: List[EvaluationRow]
    metrics: Metrics