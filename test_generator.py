"""
End-to-end test for the generation pipeline.

Flow:
    1. Load the existing question corpus and compute embeddings.
    2. Select a diverse context (top-k questions) for the LLM.
    3. Generate a new question with generation-time avoidance.
    4. Verify the generated question against the corpus using the detector.

The final line reports whether the avoidance mechanism worked for this run.
Note that generation is non-deterministic (temperature=0.9), so repeated runs
may produce different questions.
"""

from __future__ import annotations

import pandas as pd

from src.detector import classify
from src.embeddings import embed_batch, embed_text
from src.generator import generate_question, select_diverse_context

TOPIC = "reflection of light"
CONTEXT_SIZE = 5


def _hr(width: int = 75) -> None:
    print("=" * width)


def main() -> None:
    df = pd.read_csv("data/questions.csv")
    print(f"\nLoaded {len(df)} existing questions\n")

    print("Generating embeddings...")
    embeddings = embed_batch(df["question"].tolist())
    print("Done.\n")

    # --- Step 1: pick diverse context for the LLM ---
    context = select_diverse_context(df, embeddings, k=CONTEXT_SIZE)
    print(f"Sending {len(context)} context questions to the LLM:")
    for q in context:
        print(f"  - {q}")
    print()

    # --- Step 2: generate a new question ---
    print("Generating new question via Groq...")
    new_question = generate_question(TOPIC, context)
    print(f'\nGenerated question: "{new_question}"\n')

    # --- Step 3: verify against corpus ---
    print("Verifying against existing corpus...")
    new_emb = embed_text(new_question)

    _hr()
    print("New question vs existing corpus")
    _hr()

    is_duplicate = False
    for i in range(len(df)):
        result = classify(new_question, df.iloc[i]["question"], new_emb, embeddings[i])
        marker = "  <-- DUPLICATE" if result["label"] != "DIFFERENT" else ""
        print(
            f"vs {df.iloc[i]['id']:4s} | {result['label']:20s} | "
            f"sem={result['semantic_score']:.4f} | {df.iloc[i]['question']}{marker}"
        )
        if result["label"] != "DIFFERENT":
            is_duplicate = True

    _hr()
    if is_duplicate:
        print("RESULT: Generated question is a DUPLICATE — avoidance mechanism failed")
    else:
        print("RESULT: Generated question is NEW — avoidance mechanism worked")


if __name__ == "__main__":
    main()