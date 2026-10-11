# Ye script Q1-Q10 test data pe detector ko chala ke result dikhata hai.
# Saath mein precision/recall/F1 bhi nikalta hai taaki pata chale
# ki detector kitna accurate hai.

import pandas as pd
from src.embeddings import embed_many
from src.detector import classify


def run_tests():
    df = pd.read_csv("data/questions.csv")
    print(f"\n{len(df)} questions load kiye\n")

    print("Embeddings bana rahe hain...")
    embs = embed_many(df["question"].tolist())
    print("Ho gaya.\n")

    # ------- Test 1: Q1 ko baaki sab se compare karo -------
    print("=" * 75)
    print("Q1 vs baaki sab:")
    print("=" * 75)

    q1_text = df.iloc[0]["question"]
    q1_emb = embs[0]

    for i in range(1, len(df)):
        res = classify(q1_text, df.iloc[i]["question"], q1_emb, embs[i])
        print(
            f"Q1 vs {df.iloc[i]['id']:4s} | {res['label']:20s} | "
            f"exact={res['exact_score']:6.2f} | sem={res['semantic_score']:.4f} | "
            f"{df.iloc[i]['question']}"
        )

    # ------- Test 2: Q2 vs Q10 (exact duplicate hona chahiye) -------
    # Ye specifically check karne ke liye hai ki exact match kaam kar raha hai ya nahi.
    print("\n" + "=" * 75)
    print("Q2 vs Q10 (expected: EXACT_DUPLICATE):")
    print("=" * 75)

    i2 = df.index[df["id"] == "Q2"][0]
    i10 = df.index[df["id"] == "Q10"][0]
    res = classify(
        df.iloc[i2]["question"], df.iloc[i10]["question"], embs[i2], embs[i10]
    )
    print(
        f"Q2 vs Q10 | {res['label']:20s} | exact={res['exact_score']:.2f} | "
        f"sem={res['semantic_score']:.4f}"
    )

    # ------- Test 3: Q5 vs Q9 (dono different hone chahiye) -------
    # Ye false positive check hai - dekho ki alag questions ko duplicate na bole.
    print("\n" + "=" * 75)
    print("Q5 vs Q9 (expected: DIFFERENT):")
    print("=" * 75)

    i5 = df.index[df["id"] == "Q5"][0]
    i9 = df.index[df["id"] == "Q9"][0]
    res = classify(
        df.iloc[i5]["question"], df.iloc[i9]["question"], embs[i5], embs[i9]
    )
    print(
        f"Q5 vs Q9  | {res['label']:20s} | exact={res['exact_score']:.2f} | "
        f"sem={res['semantic_score']:.4f}"
    )

    # ------- Metrics nikal rahe hain -------
    # Ground truth Q1 ke perspective se:
    #   Q2, Q3, Q4, Q10 -> semantically same concept (duplicate hone chahiye)
    #   Q5, Q6, Q7, Q9  -> alag concepts (different hone chahiye)
    #   Q8              -> ambiguous hai (assignment mein bhi likha hai), skip kar rahe hain
    print("\n" + "=" * 75)
    print("METRICS (Q1 vs All)")
    print("=" * 75)

    duplicate_ids = {"Q2", "Q3", "Q4", "Q10"}
    skip_ids = {"Q8"}

    tp = fp = fn = tn = 0
    for i in range(1, len(df)):
        qid = df.iloc[i]["id"]
        if qid in skip_ids:
            continue
        res = classify(q1_text, df.iloc[i]["question"], q1_emb, embs[i])
        predicted_dup = res["label"] in ("EXACT_DUPLICATE", "SEMANTIC_DUPLICATE")
        actual_dup = qid in duplicate_ids

        if predicted_dup and actual_dup:
            tp += 1
        elif predicted_dup and not actual_dup:
            fp += 1
        elif not predicted_dup and actual_dup:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"True Positives  : {tp}")
    print(f"False Positives : {fp}")
    print(f"False Negatives : {fn}")
    print(f"True Negatives  : {tn}")
    print(f"\nPrecision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1 Score  : {f1:.3f}")


if __name__ == "__main__":
    run_tests()