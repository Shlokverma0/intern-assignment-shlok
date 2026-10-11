# LLM se naya question generate karne ke liye Groq use kar rahe hain.
# Generation ke waqt hum existing questions LLM ko dikhate hain taaki wo
# wahi questions dobara na banaye. Ye "generation-time avoidance" hai.

import os
from dotenv import load_dotenv
from groq import Groq
from sklearn.cluster import KMeans
import numpy as np

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY nahi mili. .env file check kar - GROQ_API_KEY=gsk_... hona chahiye"
            )
        _client = Groq(api_key=api_key)
    return _client


def pick_diverse_questions(df, embeddings, k=5):
    # Saare questions LLM ko nahi bhej sakte - context limit hai.
    # Random 5 pick karein to ho sakta hai 5 same concept ke ho.
    # Isliye KMeans clustering karke har cluster se ek pick kar rahe hain -
    # isse diverse set milta hai (different concepts cover ho jate hain).
    n = min(k, len(df))
    if n == 0:
        return []

    # Agar sirf 1-2 questions hain to clustering ka matlab nahi
    if len(df) <= k:
        return df["question"].tolist()

    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    picked = []
    for cluster_id in range(n):
        idx = np.where(labels == cluster_id)[0][0]
        picked.append(df.iloc[idx]["question"])
    return picked


def generate_question(topic, existing_questions):
    # Prompt mein existing questions daal rahe hain, aur explicitly bol rahe hain
    # inko avoid karo. Ye simplest form of generation-time avoidance hai.

    existing_block = "\n".join(f"- {q}" for q in existing_questions)

    prompt = f"""You are helping build a question bank on the topic: "{topic}".

The following questions already exist. Do NOT create anything that is a duplicate
of these - neither exact wording nor same meaning in different words:

{existing_block}

Now generate ONE new question on the same topic that covers a DIFFERENT concept
or angle than all of the above.

Rules:
- Standalone and clear
- Not a paraphrase of any existing question
- Return ONLY the question text, nothing else - no numbering, no quotes, no explanation"""

    client = _get_client()
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,  # thoda creative rakha taaki variety aaye
    )
    return response.choices[0].message.content.strip()