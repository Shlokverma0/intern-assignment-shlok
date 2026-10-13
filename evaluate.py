"""
Evaluation script.

Runs the duplicate detector against the Q1-Q10 test corpus and reports:
    - Pairwise classification for Q1 vs all others
    - Explicit checks for an exact-duplicate pair and a different pair
    - Aggregate metrics (precision, recall, F1)

The Q8 case is marked as ambiguous in the assignment, so it is excluded
from the metrics (see README for reasoning).
"""

from __future__ import annotations

import pandas as pd

from src.detector import classify
from src.embeddings import embed_batch

# Ground-truth labels for Q1 vs each other question.
# Q8 is intentionally omitted from the metrics because the assignment
# itself marks it as "potentially ambiguous".
DUPLICATE_IDS = {"Q2", "Q3", "Q4", "Q10"}
SKIP_IDS = {"Q8"}


def _hr(width: int = 75) -> None:
    print("=" * width)


def run_q1_vs_all(df: pd.DataFrame, embeddings) -> None:
    """Print Q1 vs every other question with the detector's verdict."""
    _hr()
    print("Q1 vs all other questions")
    _hr()

    q1_text = df.iloc[0]["question"]
    q1_emb = embeddings[0]

    for i in range(1, len(df)):
        result = classify(q1_text, df.iloc[i]["question"], q1_emb, embeddings[i])
        print(
            f"Q1 vs {df.iloc[i]['id']:4s} | {result['label']:20s} | "
            f"exact={result['exact_score']:6.2f} | sem={result['semantic_score']:.4f} | "
            f"{df.iloc[i]['question']}"
        )


def run_explicit_checks(df: pd.DataFrame, embeddings) -> None:
    """Check two known pairs to demonstrate exact and different behaviour."""
    idx = {row["id"]: i for i, row in df.iterrows()}

    _hr()
    print("Explicit pair checks")
    _hr()

    # Q2 vs Q10 - should be flagged as an exact duplicate.
    r = classify(
        df.iloc[idx["Q2"]]["question"],
        df.iloc[idx["Q10"]]["question"],
        embeddings[idx["Q2"]],
        embeddings[idx["Q10"]],
    )
    print(
        f"Q2 vs Q10 | {r['label']:20s} | exact={r['exact_score']:.2f} | "
        f"sem={r['semantic_score']:.4f}  (expected: EXACT_DUPLICATE)"
    )

    # Q5 vs Q9 - both different concepts, should be DIFFERENT.
    r = classify(
        df.iloc[idx["Q5"]]["question"],
        df.iloc[idx["Q9"]]["question"],
        embeddings[idx["Q5"]],
        embeddings[idx["Q9"]],
    )
    print(
        f"Q5 vs Q9  | {r['label']:20s} | exact={r['exact_score']:.2f} | "
        f"sem={r['semantic_score']:.4f}  (expected: DIFFERENT)"
    )


def compute_metrics(df: pd.DataFrame, embeddings) -> None:
    """Compute precision, recall, and F1 for Q1 vs the rest of the corpus."""
    _hr()
    print("Metrics (Q1 vs all, Q8 excluded as ambiguous)")
    _hr()

    q1_text = df.iloc[0]["question"]
    q1_emb = embeddings[0]

    tp = fp = fn = tn = 0
    for i in range(1, len(df)):
        qid = df.iloc[i]["id"]
        if qid in SKIP_IDS:
            continue

        result = classify(q1_text, df.iloc[i]["question"], q1_emb, embeddings[i])
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

    print(f"True Positives  : {tp}")
    print(f"False Positives : {fp}")
    print(f"False Negatives : {fn}")
    print(f"True Negatives  : {tn}")
    print()
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1 Score  : {f1:.3f}")


def main() -> None:
    df = pd.read_csv("data/questions.csv")
    print(f"\nLoaded {len(df)} questions from data/questions.csv\n")

    print("Generating embeddings...")
    embeddings = embed_batch(df["question"].tolist())
    print("Done.\n")

    run_q1_vs_all(df, embeddings)
    print()
    run_explicit_checks(df, embeddings)
    print()
    compute_metrics(df, embeddings)


if __name__ == "__main__":
    main()