# AI Question Generation with Duplicate Prevention

**Author:** Shlok Verma
**Assignment:** Intern Assignment, Question 2
**GitHub:** https://github.com/Shlokverma0/intern-assignment-shlok

---

## Live Links

| What | URL |
|---|---|
| **Frontend (Vercel)** | https://intern-assignment-shlok.vercel.app |
| **Backend (Render)** | https://intern-assignment-shlok.onrender.com |
| **API Docs (Swagger)** | https://intern-assignment-shlok.onrender.com/docs |
| **Health Check** | https://intern-assignment-shlok.onrender.com/api/health |
| **GitHub Repo** | https://github.com/Shlokverma0/intern-assignment-shlok |
| **Demo Video** | https://drive.google.com/file/d/1tu0SSqPcg2vSuu0VtrASY_9QLO5oEFY2/view?usp=sharing |

> **Note for reviewers:** The backend runs on Render's free tier. After 15 minutes of inactivity, the first request may take 30-60 seconds (cold start). Please open the [Health Check](https://intern-assignment-shlok.onrender.com/api/health) link first to wake the server, then use the frontend. An external uptime monitor pings the health endpoint every 10 minutes to keep the backend warm.

### Try It in 30 Seconds

1. Open the [Health Check](https://intern-assignment-shlok.onrender.com/api/health) link and wait until it responds OK.
2. Open the [Frontend](https://intern-assignment-shlok.vercel.app).
3. **Generate** tab: enter a topic (for example "reflection of light") and generate a question. It is verified against the corpus automatically.
4. **Evaluate** tab: run the detector on Q1-Q10 and see precision, recall and F1.
5. **Compare** tab: paste any two questions and see whether they are duplicates.

---

## Overview

A full-stack system where an LLM generates new questions for a given topic, and a separate detector verifies that the generated question does not duplicate anything already in the corpus.

There are two independent layers:

1. **Generation-time avoidance:** the LLM is shown a diverse subset of existing questions and told to steer away from them.
2. **Independent duplicate detection:** a separate, deterministic check classifies each question as an exact duplicate, a semantic duplicate, or genuinely new.

Layer 2 exists because prevention is never 100% reliable. LLMs sometimes ignore instructions, so the output is verified instead of trusted.

---

## Architecture

```text
Frontend (Vercel)
  Dark-themed single-page UI
  HTML / CSS / JavaScript
        |
        |  HTTP / JSON
        v
FastAPI Backend (Render)
  /api/generate
  /api/detect
  /api/evaluate
  /api/corpus
  /api/health
        |
        |  Python calls
        v
src/ modules
  embeddings . detector . generator
```

Frontend and backend are cleanly separated. The frontend only talks to the backend over HTTP, so the UI can be replaced (mobile app, CLI, another web client) without touching the backend.

---

## Project Structure

```text
intern-assignment-shlok/
├── backend/
│   ├── __init__.py
│   ├── main.py            # FastAPI app + endpoints
│   └── schemas.py         # Pydantic models
├── data/
│   └── questions.csv      # Q1-Q10 test corpus with tags
├── frontend/
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   └── vercel.json
├── src/
│   ├── __init__.py
│   ├── embeddings.py      # FastEmbed wrapper
│   ├── detector.py        # Exact + semantic detection
│   └── generator.py       # LLM generation with avoidance
├── evaluate.py            # CLI evaluation
├── test_generator.py      # CLI end-to-end test
├── render.yaml
├── requirements.txt
└── .gitignore
```

The `.env` file holds the API key locally and is excluded from the repo through `.gitignore`.

---

## Design Choices

### Embeddings: FastEmbed with `BAAI/bge-small-en-v1.5`

| Option | Why not chosen |
|---|---|
| OpenAI embeddings | Paid, needs internet on every call |
| sentence-transformers | Requires torch (~500MB RAM) |
| Google Gemini | Tight rate limits |
| TF-IDF | Only word overlap, not meaning |

**Why FastEmbed:** The first deploy on Render's free tier (512MB RAM) ran out of memory. `sentence-transformers` pulls in torch and needs ~500MB just to load the model. FastEmbed uses the ONNX runtime instead and drops memory to roughly 200MB with comparable quality, so it fits the free tier.

### Two-Layer Detection

- **Exact match alone** misses paraphrases.
- **Semantic match alone** is threshold-sensitive and does not distinguish identical strings.

Together they give a strong signal.

### Exact Match: RapidFuzz

Threshold: **90**. Handles casing, punctuation and small typos.

### Semantic Match: Cosine Similarity

Threshold: **0.87**. FastEmbed scores run higher than sentence-transformers, so the threshold was re-tuned on this corpus.

| Threshold | Q1 vs Q5 (different) | Q1 vs Q2 (duplicate) | Verdict |
|---|---|---|---|
| 0.80 | 0.84, wrongly flagged as duplicate | 0.95, duplicate | Too many false positives (Q5, Q6) |
| **0.87** | 0.84, different | 0.95, duplicate | **Best balance** |
| 0.92 | 0.84, different | 0.95, duplicate | Misses true duplicate Q3 (0.908) |

All semantic scores are returned as **percentages (0-100)**.

### Tags

Each question has a `tags` field with key-value pairs (for example `topic:optics`, `difficulty:easy`, `type:definition`).

**Tags are deliberately excluded from duplicate detection.** Detection is purely content-based: two questions with different tags can still be duplicates, and two with the same tags can be completely different. Tags are for organization only.

New tag keys and values can be introduced at any time. No schema change is required, because tags are stored as a raw string field.

### Context Selection: KMeans

The LLM has a limited context window, and random selection can pick near-duplicates. KMeans clusters the embeddings and picks one question per cluster, which maximizes diversity in the prompt.

### LLM: Groq `openai/gpt-oss-120b`

Free tier, fast inference, OpenAI-compatible API. The `llama-3.x` models were decommissioned during the project.

---

## Scalability

How the design behaves as the question collection grows. The figures below are estimates from the algorithmic cost, not benchmarked at that scale.

### Current approach (up to roughly 10,000 questions)

| Step | Cost |
|---|---|
| Corpus embeddings | Computed once at startup, cached in memory |
| New question embedding | O(1), a single FastEmbed call |
| Exact match (RapidFuzz) | O(n) per new question |
| Semantic match (cosine) | O(n) per new question, one dot product per pair |

At around 1,000 questions, a single detection is expected to stay well under 100ms on CPU. At around 10,000 questions it grows to roughly a second per new question, which is acceptable for this use case.

### Scaling beyond 10,000 questions

1. **Linear search in the semantic detector.** Replace the brute-force dot product with an approximate nearest-neighbor index (FAISS IVF, HNSW or a hosted vector database). This reduces lookup cost from O(n) to roughly O(log n).
2. **Context window in the generator.** The KMeans cluster count stays fixed (k=5), so the LLM prompt does not grow with the corpus. This scales without changes.

### Memory footprint

- FastEmbed model: about 50MB on disk, about 200MB loaded in RAM.
- Corpus embeddings: 384 floats x 4 bytes = about 1.5KB per question, so about 15MB at 10,000 questions.
- For 100,000+ questions, embeddings should move to a vector database instead of RAM.

### Cost per request

- LLM call (Groq): free tier, about 1-2 seconds per generation.
- Embedding and detection: no external API calls, everything runs locally.

---

## Performance

### Approximate latency (measured locally)

| Operation | Latency |
|---|---|
| `GET /api/health` | under 50ms |
| `GET /api/corpus` | under 100ms |
| `POST /api/detect` | 200-400ms (2 embeddings + 1 dot product) |
| `POST /api/evaluate` | 500-800ms (10 comparisons) |
| `POST /api/generate` | 2-4s (includes the LLM call) |

Cold starts on Render's free tier add 30-60s to the first request after idle time. The uptime monitor keeps this rare.

### Memory

- Backend process: about 250-300MB when running, within Render's 512MB free tier.
- The sentence-transformers alternative exceeded 512MB during model load and was replaced by FastEmbed.

### Accuracy

| Metric | Value |
|---|---|
| Precision | 1.000 |
| Recall | 1.000 |
| F1 | 1.000 |

These are on the 10-question test set with Q8 excluded as ambiguous. A larger benchmark is needed for production-grade confidence.

### How these were measured

- **Accuracy:** `python evaluate.py` over the provided Q1-Q10 corpus.
- **Latency:** repeated calls to each endpoint on a local machine.
- **Memory:** process monitoring locally and the Render dashboard.

---

## Running Locally

```bash
git clone https://github.com/Shlokverma0/intern-assignment-shlok.git
cd intern-assignment-shlok
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Get a Groq API key from https://console.groq.com and add it to a `.env` file:

```text
GROQ_API_KEY=your_key_here
```

Start the server:

```bash
uvicorn backend.main:app --reload --port 8000
```

Open:

- Web UI: http://localhost:8000/app/
- Swagger: http://localhost:8000/docs
- Health: http://localhost:8000/api/health

Optional CLI scripts:

```bash
python evaluate.py          # Detector metrics on Q1-Q10
python test_generator.py    # End-to-end generate + verify
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/corpus` | Q1-Q10 corpus with tags |
| POST | `/api/generate` | Generate and verify a question |
| POST | `/api/detect` | Classify a question pair |
| POST | `/api/evaluate` | Metrics over Q1 vs all |

### Example: detect

```bash
curl -X POST https://intern-assignment-shlok.onrender.com/api/detect \
  -H "Content-Type: application/json" \
  -d '{"question_a": "What is reflection of light?", "question_b": "Define the reflection of light."}'
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

| ID | Question | Expected |
|---|---|---|
| Q1 | What is reflection of light? | Original |
| Q2 | Define the reflection of light. | Semantic duplicate of Q1 |
| Q3 | What do you understand by the reflection of light? | Semantic duplicate of Q1 |
| Q4 | Explain the meaning of light reflection. | Semantic duplicate of Q1 |
| Q5 | State the laws of reflection of light. | Different |
| Q6 | What is the angle of reflection? | Different |
| Q7 | The angle of incidence is 30 degrees. What is the angle of reflection? | Different |
| Q8 | What is reflection? | Ambiguous |
| Q9 | What is the difference between regular and diffuse reflection? | Different |
| Q10 | Define the reflection of light. | Exact duplicate of Q2 |

---

## Results

### Detector evaluation (Q1 vs all)

| Comparison | Question | Detector | Exact | Semantic |
|---|---|---|---|---|
| Q1 vs Q2 | Define the reflection of light. | SEMANTIC_DUPLICATE | 73.68 | 94.66% |
| Q1 vs Q3 | What do you understand by the reflection of light? | SEMANTIC_DUPLICATE | 68.42 | 90.80% |
| Q1 vs Q4 | Explain the meaning of light reflection. | SEMANTIC_DUPLICATE | 48.48 | 94.26% |
| Q1 vs Q5 | State the laws of reflection of light. | DIFFERENT | 75.00 | 83.89% |
| Q1 vs Q6 | What is the angle of reflection? | DIFFERENT | 62.07 | 83.00% |
| Q1 vs Q7 | The angle of incidence is 30 degrees. What is the angle of reflection? | DIFFERENT | 37.89 | 73.19% |
| Q1 vs Q8 | What is reflection? | SEMANTIC_DUPLICATE | 80.00 | 87.02% (ambiguous) |
| Q1 vs Q9 | What is the difference between regular and diffuse reflection? | DIFFERENT | 43.18 | 75.78% |
| Q1 vs Q10 | Define the reflection of light. | SEMANTIC_DUPLICATE | 73.68 | 94.66% |
| Q2 vs Q10 | Define the reflection of light. | EXACT_DUPLICATE | 100.00 | 100.00% |
| Q5 vs Q9 | What is the difference between regular and diffuse reflection? | DIFFERENT | 44.90 | 69.09% |

Q8 is classified as SEMANTIC_DUPLICATE at 87.02%. The assignment marks it as potentially ambiguous, so it is excluded from the aggregate metrics but kept in the table above for transparency.

### Aggregate metrics (Q1 vs all, Q8 excluded as ambiguous)

| Metric | Value |
|---|---|
| True Positives | 4 |
| False Positives | 0 |
| False Negatives | 0 |
| True Negatives | 4 |
| Precision | 1.000 |
| Recall | 1.000 |
| F1 Score | 1.000 |

### Generation test

Example generated question:

> "How does a periscope use the principles of light reflection to enable an observer to see over an obstacle?"

It was verified against all 10 existing questions. All came back DIFFERENT, with semantic scores between 63% and 80%, so the avoidance worked.

---

## UI

A single-page application with three tabs:

- **Generate:** enter a topic, generate a question, verify it against the corpus.
- **Evaluate:** run the detector on Q1-Q10 and view the metrics.
- **Compare:** sanity-check two questions side by side.

No build step and no framework. Plain HTML, CSS and JavaScript talking to FastAPI over HTTP.

---

## Trade-offs and Limitations

1. **Topic drift.** A loose prompt sometimes drifts: "reflection of light" once produced a "plane mirrors" question. Both are optics, but the drift is real. It is a precision-vs-diversity trade-off: a strict prompt gives less drift but repetitive questions; a loose prompt gives variety but may stray. A post-generation topic classifier would fix this and is not implemented.
2. **Q8 is ambiguous.** "What is reflection?" could mean optical reflection or reflection in general. It is classified as a semantic duplicate (87.02%) and excluded from the metrics, as the assignment notes.
3. **Small evaluation set.** Only 10 questions on one topic, so the results are indicative, not conclusive. The pipeline scales (O(n) per detection), but proper benchmarking needs a larger corpus.
4. **Threshold tuning.** 0.87 was tuned for this corpus and FastEmbed. A different domain may need re-tuning.
5. **Deployment memory.** The switch to FastEmbed was forced by Render's 512MB limit. A paid tier with more RAM would allow sentence-transformers again (threshold around 0.80).
6. **English only.** The embedding model is English; multilingual questions are not evaluated.

---

## What I'd Add With More Time

- Vector database (FAISS) for fast approximate search at scale
- Few-shot prompts to reduce topic drift
- A larger evaluation set from real question banks
- A topic-adherence classifier
- Disk caching for corpus embeddings
- A Dockerfile
- CI on every commit

---

## Tech Stack

| Layer | Tools |
|---|---|
| Backend | FastAPI, Uvicorn, Pydantic |
| Embeddings | FastEmbed (ONNX, no torch) |
| ML | scikit-learn, NumPy, pandas |
| Similarity | RapidFuzz, cosine similarity |
| LLM | Groq (`openai/gpt-oss-120b`) |
| Frontend | HTML, CSS, JavaScript |
| Deploy | Vercel (frontend), Render (backend) |

---

## License

Personal assignment submission.
