"""
LLM-based question generation with duplicate avoidance.

Two responsibilities:

    1. Context selection - pick a small, diverse subset of existing
       questions to send to the LLM without exceeding its context window.
       KMeans clustering is used so the selected questions cover different
       concepts rather than being near-duplicates of each other.

    2. Generation with avoidance - prompt the LLM to produce a new question
       that is conceptually different from the ones provided.

Note: This handles generation-time avoidance only. A separate detector
(see detector.py) verifies the result independently.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sklearn.cluster import KMeans

load_dotenv()

# Model choice notes:
#   Groq decommissioned llama-3.1-70b-versatile and llama-3.3-70b-versatile.
#   Current recommendation from Groq's deprecation page is openai/gpt-oss-120b.
LLM_MODEL = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0.9

_client: Groq | None = None


def _get_client() -> Groq:
    """Return a cached Groq client, initializing it on first use."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Ensure a .env file exists with "
                "GROQ_API_KEY=your_key_here"
            )
        _client = Groq(api_key=api_key)
    return _client


def select_diverse_context(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    k: int = 5,
) -> List[str]:
    """
    Select k diverse questions from the corpus to use as LLM context.

    Strategy: cluster embeddings with KMeans, then pick one representative
    from each cluster. This avoids sending near-duplicate questions to the
    LLM (which would waste context and reduce the diversity of guidance).

    Args:
        df: DataFrame with a 'question' column.
        embeddings: Array of shape (n_questions, embedding_dim).
        k: Number of questions to select.

    Returns:
        List of selected question strings.
    """
    n = min(k, len(df))
    if n == 0:
        return []
    if len(df) <= k:
        return df["question"].tolist()

    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    picked: List[str] = []
    for cluster_id in range(n):
        idx = int(np.where(labels == cluster_id)[0][0])
        picked.append(df.iloc[idx]["question"])
    return picked


def generate_question(topic: str, existing_questions: List[str]) -> str:
    """
    Generate a single new question on `topic` while steering away from
    the questions in `existing_questions`.

    Args:
        topic: Topic string (e.g. "reflection of light").
        existing_questions: List of questions to avoid duplicating.

    Returns:
        The generated question as a plain string.
    """
    existing_block = "\n".join(f"- {q}" for q in existing_questions)

    prompt = f"""You are building a question bank on the topic: "{topic}".

The following questions already exist. Do NOT produce anything that is
a duplicate of these — neither identical wording nor the same meaning
expressed differently:

{existing_block}

Generate ONE new question on the same topic that covers a DIFFERENT
concept or angle than every question listed above.

Rules:
- Standalone and clearly worded
- Not a paraphrase of any existing question
- Return ONLY the question text. No numbering, no quotes, no explanation."""

    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
    )
    return response.choices[0].message.content.strip()