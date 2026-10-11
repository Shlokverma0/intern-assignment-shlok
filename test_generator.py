# Ye script generator ko test karti hai.
# Q1-Q10 ko existing maan kar ek naya question generate karta hai,
# phir detector se check karta hai ki wo duplicate hai ya nahi.

import pandas as pd
from src.embeddings import embed_many, embed_one
from src.detector import classify
from src.generator import generate_question, pick_diverse_questions


def main():
    df = pd.read_csv("data/questions.csv")
    print(f"{len(df)} existing questions mile\n")

    print("Embeddings bana rahe hain...")
    embs = embed_many(df["question"].tolist())
    print("Done.\n")

    # Context ke liye diverse questions pick karo
    context = pick_diverse_questions(df, embs, k=5)
    print("LLM ko ye 5 questions context mein bhej rahe hain:")
    for q in context:
        print(f"  - {q}")
    print()

    # Naya question generate karo
    print("Groq se naya question generate kar rahe hain...")
    new_q = generate_question("reflection of light", context)
    print(f"\nNaya question: \"{new_q}\"\n")

    # Detector se check karo ki ye duplicate hai ya nahi
    print("Ab detector se check kar rahe hain ki ye duplicate hai ya nahi...")
    new_emb = embed_one(new_q)

    print("\n" + "=" * 75)
    print("New question vs existing corpus:")
    print("=" * 75)

    is_dup = False
    for i in range(len(df)):
        res = classify(new_q, df.iloc[i]["question"], new_emb, embs[i])
        marker = " <-- DUP!" if res["label"] != "DIFFERENT" else ""
        print(
            f"vs {df.iloc[i]['id']:4s} | {res['label']:20s} | "
            f"sem={res['semantic_score']:.4f} | {df.iloc[i]['question']}{marker}"
        )
        if res["label"] != "DIFFERENT":
            is_dup = True

    print("\n" + "=" * 75)
    if is_dup:
        print("RESULT: Generated question DUPLICATE hai - avoid mechanism fail hui")
    else:
        print("RESULT: Generated question NAYA hai - avoid mechanism kaam kiya")


if __name__ == "__main__":
    main()