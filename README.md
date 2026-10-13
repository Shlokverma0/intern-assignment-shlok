# AI Question Generation with Duplicate Prevention

**Author:** Shlok Verma  
**Assignment:** Intern Assignment — Question 2  
**Repo:** [intern-assignment-shlok](https://github.com/Shlokverma0/intern-assignment-shlok)

---

## Overview

A full-stack system where an LLM generates new questions for a given topic, and a separate detector verifies that the generated questions don't duplicate anything already in the corpus.

The design uses **two independent layers**:

1. **Generation-time avoidance** — the LLM is shown a diverse subset of existing questions and instructed to steer away from them.
2. **Independent duplicate detection** — a separate, deterministic check classifies each generated question as an exact duplicate, a semantic duplicate, or genuinely new.

Layer 2 exists because prevention is never 100% reliable. LLMs occasionally ignore instructions, especially under long contexts. So rather than trusting prevention alone, we verify.

---

## Architecture

```
┌──────────────────────────────┐
│      Frontend (HTML/JS)      │
│   Dark-themed single-page UI │
│   Served at /app/            │
└──────────────┬───────────────┘
               │  HTTP/JSON
               ▼
┌──────────────────────────────┐
│      FastAPI Backend         │
│  /api/generate               │
│  /api/detect                 │
│  /api/evaluate               │
│  /api/corpus                 │
│  /api/health                 │
└──────────────┬───────────────┘
               │  Python calls
               ▼
┌──────────────────────────────┐
│        src/ modules          │
│  embeddings · detector ·     │
│  generator                   │
└──────────────────────────────┘
```

The frontend and backend are cleanly separated. The frontend only talks to the backend over HTTP; the backend owns all domain logic. This mirrors a production-style split, where the UI can be replaced (mobile app, CLI, another web client) without touching the backend.

---

## Project Structure

```
intern-assignment-shlok/
├── backend/
│   ├── __init__.py
│   ├── main.py                # FastAPI app + endpoints
│   └── schemas.py             # Pydantic request/response models
├── data/
│   └── questions.csv          # Q1-Q10 test corpus
├── frontend/
│   ├── index.html             # UI markup
│   ├── script.js              # UI logic + API calls
│   └── style.css              # Dark theme styling
├── src/
│   ├── __init__.py
│   ├── embeddings.py          # Sentence-transformer wrapper
│   ├── detector.py            # Exact + semantic duplicate detection
│   └── generator.py           # LLM question generation with avoidance
├── evaluate.py                # CLI: run detector over Q1-Q10, print metrics
├── test_generator.py          # CLI: end-to-end generate + verify
├── requirements.txt
├── .env                       # GROQ_API_KEY (not committed)
└── .gitignore
```

---

## Design Choices

### Embeddings: `sentence-transformers` with `all-MiniLM-L6-v2`

Alternatives considered:

| Option | Why not chosen |
|---|---|
| OpenAI `text-embedding-3-small` | Better quality, but paid and requires internet per call |
| Google Gemini embeddings | Free tier exists but rate limits are tight |
| TF-IDF | Classical, but only catches word overlap, not meaning |

Chosen because it's **free, local, lightweight (~80MB), CPU-friendly, and fast**. For a corpus of 10-1000 questions, the quality is more than sufficient.

### Two-layer duplicate detection

Neither layer alone is reliable:

- **Exact match alone** misses paraphrases ("What is reflection?" vs "Define reflection").
- **Semantic match alone** gives no special treatment to literally identical questions and is threshold-sensitive.

Running both gives a strong signal: exact match first (high-confidence), then semantic match as a fallback.

### Exact match: `rapidfuzz`

A simple `q1 == q2` fails on casing and punctuation. `rapidfuzz` handles both, is C++-fast, and fuzzy-matches typos. Threshold of **90** was tuned against the corpus.

### Semantic match: cosine similarity

Cosine similarity is the standard metric for comparing embeddings. Since embeddings are L2-normalized, it reduces to a dot product.

**Threshold: 0.80**

Tuned against the Q1-Q10 corpus:

| Threshold | Q1 vs Q5 (should be DIFFERENT) | Q1 vs Q2 (should be DUP) | Verdict |
|---|---|---|---|
| 0.70 | 0.79 → DUP ❌ | 0.94 → DUP ✅ | Too many false positives |
| **0.80** | 0.79 → DIFF ✅ | 0.94 → DUP ✅ | **Best** |
| 0.90 | 0.79 → DIFF ✅ | 0.94 → DUP ✅ | Misses the Q8 case |
| 0.95 | 0.79 → DIFF ✅ | 0.94 → DIFF ❌ | False negatives |

### Context selection: KMeans

The LLM has a limited context window, so we send only a small, diverse set of existing questions. Random selection can pick near-duplicates. KMeans clusters embeddings and picks one representative per cluster, maximizing diversity.

### LLM: Groq with `openai/gpt-oss-120b`

Groq's free tier is generous, its inference is very fast, and its API is OpenAI-compatible. The `llama-3.x` models were decommissioned during this project; `openai/gpt-oss-120b` is the current Groq-recommended replacement.

---

## Running Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a Groq API key

Sign up at https://console.groq.com and create a free API key.

### 3. Create `.env` in the project root

```
GROQ_API_KEY=your_key_here
```

### 4. Start the FastAPI server

```bash
uvicorn backend.main:app --reload --port 8000
```

### 5. Open the UI

- **Web UI:** http://localhost:8000/app/
- **Interactive API docs (Swagger):** http://localhost:8000/docs
- **Health check:** http://localhost:8000/api/health

### Optional CLI scripts

```bash
python evaluate.py          # Detector metrics on Q1-Q10
python test_generator.py    # End-to-end generate + verify
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/corpus` | Return the Q1-Q10 corpus |
| POST | `/api/generate` | Generate a new question and verify it against the corpus |
| POST | `/api/detect` | Classify a pair of questions |
| POST | `/api/evaluate` | Run detector on Q1 vs all and return metrics |

Example request:

```bash
curl -X POST http://localhost:8000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"question_a": "What is reflection of light?", "question_b": "Define reflection."}'
```

Response:

```json
{
  "label": "SEMANTIC_DUPLICATE",
  "exact_score": 73.68,
  "semantic_score": 0.9416
}
```

---

## Test Data

The assignment provided 10 questions on "reflection of light":

| ID | Question | Expected |
|---|---|---|
| Q1 | What is reflection of light? | Original |
| Q2 | Define the reflection of light. | Semantic duplicate of Q1 |
| Q3 | What do you understand by the reflection of light? | Semantic duplicate of Q1 |
| Q4 | Explain the meaning of light reflection. | Semantic duplicate of Q1 |
| Q5 | State the laws of reflection of light. | Different |
| Q6 | What is the angle of reflection? | Different |
| Q7 | The angle of incidence is 30°. What is the angle of reflection? | Different |
| Q8 | What is reflection? | Ambiguous |
| Q9 | What is the difference between regular and diffuse reflection? | Different |
| Q10 | Define the reflection of light. | Exact duplicate of Q2 |

---

## Results

### Detector evaluation (Q1 vs all)

```
Q1 vs Q2   | SEMANTIC_DUPLICATE   | exact= 73.68 | sem=0.9416
Q1 vs Q3   | SEMANTIC_DUPLICATE   | exact= 68.42 | sem=0.9034
Q1 vs Q4   | SEMANTIC_DUPLICATE   | exact= 48.48 | sem=0.9290
Q1 vs Q5   | DIFFERENT            | exact= 75.00 | sem=0.7906
Q1 vs Q6   | DIFFERENT            | exact= 62.07 | sem=0.7051
Q1 vs Q7   | DIFFERENT            | exact= 37.89 | sem=0.5405
Q1 vs Q8   | SEMANTIC_DUPLICATE   | exact= 80.00 | sem=0.8638  (ambiguous)
Q1 vs Q9   | DIFFERENT            | exact= 43.18 | sem=0.6626
Q1 vs Q10  | SEMANTIC_DUPLICATE   | exact= 73.68 | sem=0.9416

Q2 vs Q10  | EXACT_DUPLICATE      | exact=100.00 | sem=1.0000
Q5 vs Q9   | DIFFERENT            | exact= 44.90 | sem=0.5359
```

### Aggregate metrics (Q1 vs all, Q8 excluded as ambiguous)

| Metric | Value |
|---|---|
| True Positives | 4 |
| False Positives | 0 |
| False Negatives | 0 |
| True Negatives | 4 |
| **Precision** | **1.000** |
| **Recall** | **1.000** |
| **F1 Score** | **1.000** |

### Generation test

A typical run produces a question like:

> *"How does a plane mirror produce a virtual image of an object, and what are the key characteristics of that image?"*

Verified against all 10 existing questions → all `DIFFERENT`, semantic scores in the 0.33-0.55 range. The avoidance mechanism worked.

---

## UI

The frontend is a single-page application (no build step, no framework) with three tabs:

- **Generate** — enter a topic, generate a question, see it verified against the corpus
- **Evaluate** — run the detector on Q1-Q10 and view metrics
- **Compare** — sanity check two custom questions side-by-side

It talks to the FastAPI backend exclusively over HTTP and updates the UI without a page reload.

---

## Trade-offs and Limitations

### 1. Topic drift in generation

Under a loose prompt, the LLM occasionally drifts from the strict topic. In one run, "reflection of light" produced a question about plane mirrors. Both are optics, but the drift is real.

This is the classic **precision-vs-diversity trade-off**:

- A stricter prompt → less drift, but repetitive questions over time.
- A looser prompt → more diversity, but the question may stray.

A post-generation topic classifier or a stricter few-shot prompt would tighten this. Not implemented here.

### 2. Q8 is genuinely ambiguous

"What is reflection?" could refer to optical reflection or reflection in general. The model classified it as a semantic duplicate of Q1 with score 0.8638. This is defensible, but the assignment explicitly flagged it as ambiguous, so Q8 is excluded from metrics.

### 3. Small evaluation set

The metrics are computed on 10 questions. Real deployments have hundreds or thousands. The pipeline scales (embeddings are O(n); each new detection is O(n) per question), but proper benchmarking would require a larger corpus.

### 4. Threshold tuning

The 0.80 semantic threshold was tuned on this specific corpus. A different domain (code questions, math problems) might need a different threshold. A production system would re-tune periodically or use an adaptive approach.

### 5. Cost trade-off

Local embeddings are free but slower than API embeddings on large batches. For 10-100 questions this doesn't matter. At 10,000+, batched API calls or a vector database (FAISS, Pinecone) would be needed for sublinear search.

---

## What I'd Add With More Time

- Vector database (FAISS) for fast approximate search at scale
- Few-shot prompt engineering to reduce topic drift
- A larger evaluation set pulled from real question banks
- A topic-adherence classifier to flag drift
- Caching corpus embeddings to disk so restarts are instant
- Dockerfile for one-command deployment
- CI to run the evaluation on every commit

---

## Stack

- **Python 3.14**
- `fastapi`, `uvicorn`, `pydantic` — backend API
- `sentence-transformers` — embeddings
- `scikit-learn` — KMeans clustering, metrics
- `rapidfuzz` — fuzzy string matching
- `groq` — LLM API client
- `pandas`, `numpy` — data handling
- Vanilla **HTML / CSS / JavaScript** — frontend (no framework, no build step)

---

## License

Personal assignment submission.