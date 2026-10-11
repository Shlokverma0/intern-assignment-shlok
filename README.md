# AI Question Generation with Duplicate Prevention

**Author:** Shlok Verma  
**Assignment:** Intern Assignment — Question 2  
**Repo:** [intern-assignment-shlok](https://github.com/Shlokverma0/intern-assignment-shlok)

---

## What This Is

The task was to build a system where an LLM generates new questions for a topic, and we make sure it doesn't create duplicates of questions we already have. On top of that, we need a separate detector that can catch duplicates even if the generation-time prevention fails.

So basically two independent layers:

1. **Generation-time avoidance** — guide the LLM to produce genuinely new questions by showing it what already exists.
2. **Independent duplicate detection** — a separate check that compares each new question against the corpus and flags exact duplicates, semantic paraphrases, and near-duplicates.

The second layer exists because prevention isn't 100% reliable. LLMs occasionally ignore instructions, especially with long contexts. So we don't rely on prevention alone — we verify.

---

## Why These Choices

I'll be honest about the decisions I made and why.

### Why `sentence-transformers` (all-MiniLM-L6-v2) for embeddings?

I considered a few options:

- **OpenAI embeddings** (`text-embedding-3-small`) — better quality, but paid, and needs internet for every call.
- **Google Gemini embeddings** — free tier exists but rate limits are tight.
- **TF-IDF** — classical approach, but it only catches word overlap, not meaning. "Define reflection" and "What is reflection?" would score low even though they mean the same thing.

I went with `all-MiniLM-L6-v2` because:

- **Free** — no API costs, so I can experiment as much as I want.
- **Local** — runs on my laptop without internet. Good for privacy and offline testing.
- **Lightweight** — 80MB, runs fine on CPU. No GPU needed.
- **Fast enough** — 10 questions embed in under a second.
- **Good quality for semantic similarity** — this model is a standard pick for this kind of task in 2024.

The trade-off: OpenAI embeddings are slightly better on hard cases. But for this assignment, the local model is more than enough, and I'd rather have a reproducible, free setup.

### Why two separate detectors (exact + semantic)?

I tried using only one at first and both had blind spots:

- **Exact-only (fuzzy string match):** misses "What is reflection?" vs "Explain the meaning of light reflection" — same concept, different words.
- **Semantic-only (embeddings):** catches meaning but 100% identical questions don't get special treatment, and thresholds near the boundary cause false positives.

So I run both. Exact match first (fast, high confidence), then semantic match as a second opinion. If exact says duplicate, it's a duplicate. If exact says no but semantic says yes, then it's a semantic duplicate.

### Why `rapidfuzz` for exact matching?

Simple `q1 == q2` is useless because "What is reflection?" and "what is reflection" would be treated differently. `rapidfuzz` handles casing, punctuation, and minor typos — and it's C++ fast. 90 is the threshold I landed on after testing.

### Why cosine similarity for semantic matching?

Cosine similarity is the standard for comparing embeddings. Since I normalize embeddings at generation time (`normalize_embeddings=True`), cosine similarity reduces to a simple dot product. Fast and interpretable.

### Why the semantic threshold is 0.80

I tried a few values against the test data:

| Threshold | Q1 vs Q5 (should be DIFFERENT) | Q1 vs Q2 (should be DUP) | Verdict |
|-----------|-------------------------------|--------------------------|---------|
| 0.70      | 0.79 → DUP ❌                  | 0.94 → DUP ✅             | Too many false positives |
| **0.80**  | 0.79 → DIFF ✅                 | 0.94 → DUP ✅             | **Best** |
| 0.90      | 0.79 → DIFF ✅                 | 0.94 → DUP ✅             | Misses Q8 case |
| 0.95      | 0.79 → DIFF ✅                 | 0.94 → DIFF ❌            | False negatives |

0.80 is the sweet spot for this dataset. Lower and Q1 vs Q5 (a genuinely different question) gets flagged as a duplicate. Higher and Q1 vs Q2 (a paraphrase) gets missed.

### Why KMeans for context selection?

The LLM has a context limit — can't send all 1000+ questions. So I need to send a diverse subset.

- **Random 5:** Might send 5 questions about the same concept. Wasted context.
- **First 5:** Biased — always the same ones.
- **KMeans clustering:** Groups similar questions together, then picks one from each cluster. This covers different angles of the topic.

This is the same idea used in retrieval-augmented generation (RAG) systems in production.

### Why Groq for the LLM?

- **Free tier is generous** — 14,400 requests/day.
- **Very fast** — Llama models on Groq's hardware run at ~500 tokens/sec.
- **OpenAI-compatible API** — easy to swap if needed.

I started with `llama-3.1-70b-versatile` but Groq decommissioned it. Then tried `llama-3.3-70b-versatile`, also decommissioned. Currently using **`openai/gpt-oss-120b`** which is Groq's recommended replacement.

---

## Architecture

```
                         ┌───────────────────────┐
                         │   Existing Questions  │
                         │      (corpus)         │
                         └───────────┬───────────┘
                                     │
                       ┌─────────────┴─────────────┐
                       │  KMeans Context Selector  │
                       │    (diverse top-5)        │
                       └─────────────┬─────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │      Groq LLM (GPT-OSS-120B)   │
                    │    Prompt: "Avoid these..."    │
                    └───────────────┬────────────────┘
                                    │
                                    ▼
                          New candidate question
                                    │
                    ┌───────────────┴────────────────┐
                    │   Duplicate Detector (Layer 2) │
                    │  ┌────────────┐ ┌───────────┐  │
                    │  │ Exact      │ │ Semantic  │  │
                    │  │ (rapidfuzz)│ │ (cosine)  │  │
                    │  └────────────┘ └───────────┘  │
                    └───────────────┬────────────────┘
                                    │
                                    ▼
                    EXACT_DUPLICATE / SEMANTIC_DUPLICATE / DIFFERENT
```

The two layers are completely independent. Even if prevention fails, the detector catches it.

---

## Project Structure

```
intern-assignment-shlok/
├── data/
│   └── questions.csv          # 10-question test corpus (Q1-Q10)
├── src/
│   ├── __init__.py
│   ├── embeddings.py          # Sentence transformer wrapper
│   ├── detector.py            # Exact + semantic duplicate detection
│   └── generator.py           # LLM question generation with avoidance
├── evaluate.py                # Runs detector on Q1-Q10, prints metrics
├── test_generator.py          # End-to-end test: generate + verify
├── requirements.txt
├── .env                       # GROQ_API_KEY (not committed)
└── .gitignore
```

---

## Setup

**1. Install dependencies:**

```bash
pip install -r requirements.txt
```

**2. Get a free Groq API key:**

Sign up at https://console.groq.com and create an API key.

**3. Create `.env` in the root:**

```
GROQ_API_KEY=gsk_your_key_here
```

**4. Run the detector evaluation:**

```bash
python evaluate.py
```

**5. Run the end-to-end generation test:**

```bash
python test_generator.py
```

---

## Test Data

The assignment provided 10 questions on "reflection of light" (Q1-Q10) with expected classifications:

- Q1 = original
- Q2, Q3, Q4 = semantic duplicates of Q1
- Q5, Q6, Q7, Q9 = different questions (on the same topic)
- Q8 = ambiguous
- Q10 = exact duplicate of Q2

This is a small but well-designed corpus — it tests all three classes (exact, semantic, different) plus a tricky ambiguous case.

---

## Results

**Detector evaluation (`python evaluate.py`):**

```
Q1 vs Q2   | SEMANTIC_DUPLICATE   | exact= 73.68 | sem=0.9416 | Define the reflection of light.
Q1 vs Q3   | SEMANTIC_DUPLICATE   | exact= 68.42 | sem=0.9034 | What do you understand by the reflection of light?
Q1 vs Q4   | SEMANTIC_DUPLICATE   | exact= 48.48 | sem=0.9290 | Explain the meaning of light reflection.
Q1 vs Q5   | DIFFERENT            | exact= 75.00 | sem=0.7906 | State the laws of reflection of light.
Q1 vs Q6   | DIFFERENT            | exact= 62.07 | sem=0.7051 | What is the angle of reflection?
Q1 vs Q7   | DIFFERENT            | exact= 37.89 | sem=0.5405 | The angle of incidence is 30 degrees...
Q1 vs Q8   | SEMANTIC_DUPLICATE   | exact= 80.00 | sem=0.8638 | What is reflection?
Q1 vs Q9   | DIFFERENT            | exact= 43.18 | sem=0.6626 | What is the difference between regular and diffuse reflection?
Q1 vs Q10  | SEMANTIC_DUPLICATE   | exact= 73.68 | sem=0.9416 | Define the reflection of light.

Q2 vs Q10  | EXACT_DUPLICATE      | exact=100.00 | sem=1.0000
Q5 vs Q9   | DIFFERENT            | exact= 44.90 | sem=0.5359
```

**Metrics (Q1 vs all, Q8 excluded as ambiguous):**

- True Positives: 4 (Q2, Q3, Q4, Q10)
- False Positives: 0
- False Negatives: 0
- True Negatives: 4 (Q5, Q6, Q7, Q9)
- **Precision: 1.000**
- **Recall: 1.000**
- **F1 Score: 1.000**

**Generation test (`python test_generator.py`):**

```
Generated: "How does a concave mirror form a real image, and what factors determine whether the image is magnified, reduced, upright, or inverted?"

vs all 10 existing questions → DIFFERENT (sem scores 0.32 - 0.48)

RESULT: Generated question is NEW - avoidance mechanism worked
```

---

## Trade-offs and Limitations

I want to be honest about what this doesn't do well.

### 1. Topic drift in generation

In the test above, the generated question pivoted from "reflection of light" to "concave mirror imaging." Both are optics, but the LLM drifted slightly from the strict topic.

This is the classic precision-vs-diversity trade-off:

- **Strict prompt** ("stick strictly to reflection of light") → less drift, but more repetitive questions over time.
- **Loose prompt** ("generate on this general area") → more diversity, but the question might stray.

Right now I'm on the loose side because diversity matters more for a question bank. If you wanted strict topic adherence, you'd add a topic-check pass after generation, or tune the prompt.

### 2. Q8 is genuinely ambiguous

"What is reflection?" — this could mean optical reflection (matching Q1) or reflection in other contexts (a general question). The model classified it as SEMANTIC_DUPLICATE of Q1 with score 0.8638. I don't think this is wrong, but the assignment explicitly flagged it as "potentially ambiguous."

I excluded Q8 from the metrics for this reason. If I included it, precision would drop slightly — not because the model is wrong, but because the ground truth is unclear.

### 3. Small dataset

The evaluation is on 10 questions. Real deployments would have hundreds or thousands. The metrics here are indicative, not conclusive. The pipeline scales (embeddings are O(n), fuzzy match is O(n) per new question), but proper benchmarking would need a larger corpus.

### 4. Threshold tuning

The 0.80 semantic threshold was tuned on this specific dataset. On a different topic (say, math questions or code questions), the optimal threshold might be different. In production, this would need periodic re-tuning or a more adaptive approach.

### 5. Cost trade-off

Local embeddings are free but slower than API embeddings on large batches. For 10-100 questions, this doesn't matter. At 10,000+, you'd want batched API calls or a vector database (FAISS, Pinecone) for sublinear search.

---

## What I'd Add With More Time

- **Vector database** (FAISS or Pinecone) for fast approximate search at scale.
- **Better prompt engineering** to reduce topic drift — maybe few-shot examples.
- **Larger test set** — pull questions from real question banks (Stack Exchange, etc.).
- **Evaluation dashboard** — Streamlit UI showing precision/recall curves at different thresholds.
- **Topic adherence check** — separate classifier that flags questions drifting off-topic.
- **Caching** — cache embeddings of the corpus so you don't recompute on every new question.

---

## Stack

- **Python 3.14**
- `sentence-transformers` 6.1.0 — embeddings
- `scikit-learn` 1.9.1 — KMeans clustering, metrics
- `rapidfuzz` 3.14.6 — fuzzy string matching
- `groq` 1.7.0 — LLM API client
- `pandas`, `numpy` — data handling
- `python-dotenv` — env vars

---

## License

Personal assignment submission. Feel free to look around.