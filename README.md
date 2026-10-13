# AI Question Generation with Duplicate Prevention

**Author:** Shlok Verma  
**Assignment:** Intern Assignment — Question 2  
**Repo:** [intern-assignment-shlok](https://github.com/Shlokverma0/intern-assignment-shlok)  
**Live Demo:** [intern-assignment-shlok.onrender.com/app](https://intern-assignment-shlok.onrender.com/app/)

---

## Overview

A full-stack system where an LLM generates new questions for a given topic, and a separate detector verifies that the generated questions don't duplicate anything already in the corpus.

The design uses **two independent layers**:

1. **Generation-time avoidance** — the LLM is shown a diverse subset of existing questions and instructed to steer away from them.
2. **Independent duplicate detection** — a separate, deterministic check classifies each generated question as an exact duplicate, a semantic duplicate, or genuinely new.

Layer 2 exists because prevention is never 100% reliable. LLMs occasionally ignore instructions, especially under long contexts. So rather than trusting prevention alone, we verify.

---

## Live Deployment

The app is deployed on Render's free tier:

- **Web UI:** https://intern-assignment-shlok.onrender.com/app/
- **API docs (Swagger):** https://intern-assignment-shlok.onrender.com/docs
- **Health check:** https://intern-assignment-shlok.onrender.com/api/health

**Note:** the free tier spins down after 15 minutes of inactivity, so the first request may take 30-60 seconds while the container wakes up.

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
│   └── questions.csv          # Q1-Q10 test corpus with tags
├── frontend/
│   ├── index.html             # UI markup
│   ├── script.js              # UI logic + API calls
│   └── style.css              # Dark theme styling
├── src/
│   ├── __init__.py
│   ├── embeddings.py          # FastEmbed wrapper
│   ├── detector.py            # Exact + semantic duplicate detection
│   └── generator.py           # LLM question generation with avoidance
├── evaluate.py                # CLI: run detector over Q1-Q10, print metrics
├── test_generator.py          # CLI: end-to-end generate + verify
├── render.yaml                # Render deployment config
├── requirements.txt
├── .env                       # GROQ_API_KEY (not committed)
└── .gitignore
```

---

## Design Choices

### Embeddings: FastEmbed with `BAAI/bge-small-en-v1.5`

Alternatives considered:

| Option | Why not chosen |
|---|---|
| OpenAI `text-embedding-3-small` | Better quality, but paid and requires internet per call |
| `sentence-transformers` (`all-MiniLM-L6-v2`) | Good quality, but requires `torch` (~500MB RAM) |
| Google Gemini embeddings | Free tier exists but rate limits are tight |
| TF-IDF | Classical, but only catches word overlap, not meaning |

**The reason for choosing FastEmbed** is a real production constraint. The first deploy attempt on Render's free tier (512MB RAM) ran out of memory — `sentence-transformers` pulls in `torch` and its CUDA dependencies, which use roughly 500MB just to load the model.

FastEmbed uses **ONNX runtime instead of torch**, so the memory footprint drops to around 200MB — comfortably within the free tier. Same model quality, different backend.

### Two-layer duplicate detection

Neither layer alone is reliable:

- **Exact match alone** misses paraphrases ("What is reflection?" vs "Define reflection").
- **Semantic match alone** gives no special treatment to literally identical questions and is threshold-sensitive.

Running both gives a strong signal: exact match first (high-confidence), then semantic match as a fallback.

### Exact match: `rapidfuzz`

A simple `q1 == q2` fails on casing and punctuation. `rapidfuzz` handles both, is C++-fast, and fuzzy-matches typos. Threshold of **90** was tuned against the corpus.

### Semantic match: cosine similarity

Cosine similarity is the standard metric for comparing embeddings. Since embeddings are L2-normalized, it reduces to a dot product.

**Threshold: 0.87**

FastEmbed's similarity scores run higher than sentence-transformers, so the threshold was re-tuned after the migration. Empirically:

| Threshold | Q1 vs Q5 (should be DIFFERENT) | Q1 vs Q2 (should be DUP) | Verdict |
|---|---|---|---|
| 0.80 | 0.84 → DUP ❌ | 0.95 → DUP ✅ | Too many false positives |
| **0.87** | 0.84 → DIFF ✅ | 0.95 → DUP ✅ | **Best** |
| 0.92 | 0.84 → DIFF ✅ | 0.95 → DUP ✅ | Misses the Q8 case |

All similarity scores are returned as **percentages (0-100)** for consistency with the exact match score.

### Tags and metadata

Each question in the corpus carries a `tags` field with key-value pairs (e.g. `topic:optics`, `difficulty:easy`, `type:definition`).

**Key design decision:** tags are deliberately excluded from duplicate detection. The detector compares questions purely on content — exact wording and semantic meaning — regardless of tags.

This is intentional. Two questions with different tags can still be duplicates ("What is reflection?" vs "What is reflection of light?"), and two questions with identical tags can be completely different ("State the laws of reflection" vs "What is the angle of reflection?"). Tags are for organizational purposes only.

New tag keys and values can be introduced at any time — no schema change is required, because tags are stored as a raw string field.

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
| GET | `/api/corpus` | Return the Q1-Q10 corpus with tags |
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
  "semantic_score": 94.66
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
Q1 vs Q2   | SEMANTIC_DUPLICATE   | exact= 73.68 | sem= 94.66%
Q1 vs Q3   | SEMANTIC_DUPLICATE   | exact= 68.42 | sem= 90.80%
Q1 vs Q4   | SEMANTIC_DUPLICATE   | exact= 48.48 | sem= 94.26%
Q1 vs Q5   | DIFFERENT            | exact= 75.00 | sem= 83.89%
Q1 vs Q6   | DIFFERENT            | exact= 62.07 | sem= 83.00%
Q1 vs Q7   | DIFFERENT            | exact= 37.89 | sem= 73.19%
Q1 vs Q8   | SEMANTIC_DUPLICATE   | exact= 80.00 | sem= 87.02%  (ambiguous)
Q1 vs Q9   | DIFFERENT            | exact= 43.18 | sem= 75.78%
Q1 vs Q10  | SEMANTIC_DUPLICATE   | exact= 73.68 | sem= 94.66%

Q2 vs Q10  | EXACT_DUPLICATE      | exact=100.00 | sem=100.00%
Q5 vs Q9   | DIFFERENT            | exact= 44.90 | sem= 69.09%
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

"What is reflection?" could refer to optical reflection or reflection in general. The model classified it as a semantic duplicate of Q1 with score 87.02%. This is defensible, but the assignment explicitly flagged it as ambiguous, so Q8 is excluded from metrics.

### 3. Small evaluation set

The metrics are computed on 10 questions. Real deployments have hundreds or thousands. The pipeline scales (embeddings are O(n); each new detection is O(n) per question), but proper benchmarking would require a larger corpus.

### 4. Threshold tuning

The 0.87 semantic threshold was tuned on this specific corpus with FastEmbed. A different embedding model or domain would need re-tuning. A production system would re-tune periodically or use an adaptive approach.

### 5. Deployment memory constraint

The switch to FastEmbed was forced by a real constraint — Render's 512MB free tier. In a paid environment with more memory, `sentence-transformers` would remain a valid choice and would only require re-tuning the semantic threshold back to ~0.80.

---

## What I'd Add With More Time

- Vector database (FAISS) for fast approximate search at scale
- Few-shot prompt engineering to reduce topic drift
- A larger evaluation set pulled from real question banks
- A topic-adherence classifier to flag drift
- Caching corpus embeddings to disk so cold starts are instant
- Dockerfile for one-command deployment
- CI to run the evaluation on every commit

---

## Stack

- **Python 3.11** (deployed on Render) / **3.14** (local dev)
- `fastapi`, `uvicorn`, `pydantic` — backend API
- `fastembed` — embeddings (ONNX runtime, no torch)
- `scikit-learn` — KMeans clustering, metrics
- `rapidfuzz` — fuzzy string matching
- `groq` — LLM API client
- `pandas`, `numpy` — data handling
- Vanilla **HTML / CSS / JavaScript** — frontend (no framework, no build step)
- **Render** — deployment platform

---

## License

Personal assignment submission.