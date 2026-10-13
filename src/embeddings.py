"""
Text embedding module.

Uses sentence-transformers to convert text into dense vector representations.
The model is loaded lazily and cached in memory for repeated use within the
same process.

Model choice: all-MiniLM-L6-v2
    - Lightweight (~80MB), runs on CPU without GPU
    - Fast inference (sub-second for small batches)
    - Good semantic similarity quality for the task
    - Free and fully local (no API calls)
"""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

# Model is loaded once and reused across calls to avoid repeated disk I/O.
_model: SentenceTransformer | None = None


def _load_model() -> SentenceTransformer:
    """Load the sentence-transformer model lazily."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text: str) -> np.ndarray:
    """
    Embed a single text string.

    Args:
        text: Input text.

    Returns:
        Normalized embedding vector of shape (384,).
    """
    model = _load_model()
    return model.encode(text, normalize_embeddings=True)


def embed_batch(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts in a single batched call.

    Args:
        texts: List of input texts.

    Returns:
        Array of shape (n_texts, 384), each row L2-normalized.
    """
    model = _load_model()
    return model.encode(texts, normalize_embeddings=True, batch_size=32)