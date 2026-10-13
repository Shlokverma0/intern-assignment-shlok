"""
Text embedding module.

Uses FastEmbed (ONNX runtime, no torch) to convert text into dense vector
representations. The model is loaded lazily and cached in memory.

Why FastEmbed instead of sentence-transformers:
    sentence-transformers requires torch, which pulls in ~2GB of CUDA
    libraries and uses ~500MB RAM at runtime. FastEmbed uses ONNX runtime
    instead — same underlying model, but ~200MB RAM. This fits inside
    Render's 512MB free tier.

Model: BAAI/bge-small-en-v1.5 (similar quality to all-MiniLM-L6-v2)
"""

from __future__ import annotations

from typing import List

import numpy as np
from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model: TextEmbedding | None = None


def _load_model() -> TextEmbedding:
    """Load the FastEmbed model lazily."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string. Returns a normalized 1D numpy array."""
    model = _load_model()
    vec = list(model.embed([text]))[0]
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def embed_batch(texts: List[str]) -> np.ndarray:
    """Embed a list of texts. Returns a (n_texts, dim) normalized array."""
    model = _load_model()
    vectors = list(model.embed(texts))
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms