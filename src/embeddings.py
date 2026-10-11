# Text ko vector mein convert karne ke liye sentence-transformers use kar rahe hain.
# Kai options the - OpenAI (paid), Gemini (rate limit), TF-IDF (semantic nahi samajhta).

from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

# Model ko lazy load kar rahe hain - pehli baar import pe load karega,
# taaki test karte waqt baar baar load na ho. Ek hi baar memory mein rehta hai.
_model = None


def _load_model():
    global _model
    if _model is None:
        print(f"Loading {MODEL_NAME}... (pehli baar 1-2 min lag sakta hai)")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_one(text):
    # Ek single text ka embedding
    model = _load_model()
    return model.encode(text, normalize_embeddings=True)


def embed_many(texts):
    # Batch mein embeddings - faster hai single se, especially 10+ texts ke liye
    model = _load_model()
    return model.encode(texts, normalize_embeddings=True, batch_size=32)