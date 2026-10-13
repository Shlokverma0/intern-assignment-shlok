"""
Duplicate detection module.

Two independent layers are used to classify a pair of questions:

    Layer 1 - Exact match (fuzzy string similarity via rapidfuzz)
        Catches questions that are identical or near-identical in wording
        after normalizing punctuation and casing.

    Layer 2 - Semantic match (cosine similarity of embeddings)
        Catches paraphrases and questions with equivalent meaning but
        different wording.

Both layers run independently. If either flags a duplicate, the pair is
considered a duplicate. This layered approach is more reliable than using
either layer alone:

    - Exact-only misses paraphrases
    - Semantic-only does not give a strong signal on truly identical strings
"""

from __future__ import annotations

import re
from typing import Tuple

import numpy as np
from rapidfuzz import fuzz

# Thresholds were tuned empirically on the Q1-Q10 test corpus.
# See README for the threshold comparison table.
EXACT_THRESHOLD = 90.0
SEMANTIC_THRESHOLD = 0.80


def _normalize(text: str) -> str:
    """
    Normalize a question for exact matching.

    Steps:
        - Lowercase
        - Strip punctuation
        - Collapse multiple whitespace
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# Layer 1: Exact / fuzzy match
# ---------------------------------------------------------------------------

def exact_similarity(q1: str, q2: str) -> float:
    """Return a 0-100 fuzzy similarity score between two questions."""
    return float(fuzz.ratio(_normalize(q1), _normalize(q2)))


def is_exact_duplicate(q1: str, q2: str, threshold: float = EXACT_THRESHOLD) -> Tuple[bool, float]:
    """
    Check whether two questions are exact or near-exact duplicates.

    Returns:
        (is_duplicate, score)
    """
    score = exact_similarity(q1, q2)
    return score >= threshold, score


# ---------------------------------------------------------------------------
# Layer 2: Semantic similarity
# ---------------------------------------------------------------------------

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.

    Embeddings produced by `embed_text`/`embed_batch` are already L2-normalized,
    so cosine similarity reduces to a simple dot product.
    """
    return float(np.dot(emb1, emb2))


def is_semantic_duplicate(
    emb1: np.ndarray,
    emb2: np.ndarray,
    threshold: float = SEMANTIC_THRESHOLD,
) -> Tuple[bool, float]:
    """
    Check whether two embeddings are semantically similar enough to be
    considered the same question.

    Returns:
        (is_duplicate, similarity_score)
    """
    score = cosine_similarity(emb1, emb2)
    return score >= threshold, score


# ---------------------------------------------------------------------------
# Combined classifier
# ---------------------------------------------------------------------------

def classify(
    q1: str,
    q2: str,
    emb1: np.ndarray,
    emb2: np.ndarray,
) -> dict:
    """
    Classify a pair of questions.

    Returns a dict with keys:
        - label: "EXACT_DUPLICATE" | "SEMANTIC_DUPLICATE" | "DIFFERENT"
        - exact_score: float
        - semantic_score: float
    """
    exact_hit, exact_score = is_exact_duplicate(q1, q2)
    semantic_hit, semantic_score = is_semantic_duplicate(emb1, emb2)

    if exact_hit:
        label = "EXACT_DUPLICATE"
    elif semantic_hit:
        label = "SEMANTIC_DUPLICATE"
    else:
        label = "DIFFERENT"

    return {
        "label": label,
        "exact_score": round(exact_score, 2),
        "semantic_score": round(semantic_score, 4),
    }