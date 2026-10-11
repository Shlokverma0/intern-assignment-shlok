# Duplicate detection ka kaam do layers mein kar rahe hain:
# 1. Exact/fuzzy match - rapidfuzz se (fast, sasta, punctuation handle karta hai)
# 2. Semantic match - embeddings + cosine similarity (meaning compare karta hai)
#
# Kyun dono? Kyunki ek akela kaafi nahi hai.
# - Sirf exact se "What is reflection?" vs "Define reflection" miss ho jayega
# - Sirf semantic se 100% identical wale pe bhi confidence kam rehta hai
# Dono layer lagane se reliable detection milti hai.

import re
import numpy as np
from rapidfuzz import fuzz


def clean(text):
    # Lowercase + punctuation hata do + extra spaces fix kar do.
    # Ye isliye zaroori hai kyunki "What is reflection?" aur "what is reflection"
    # bilkul same hone chahiye exact match ke liye.
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ---------- Layer 1: Exact / fuzzy match ----------

def exact_score(q1, q2):
    # 0-100 score deta hai. 100 = identical, 0 = kuch bhi match nahi.
    return fuzz.ratio(clean(q1), clean(q2))


def is_exact_dup(q1, q2, threshold=90):
    # 90 threshold rakha - isse upar basically same question hai.
    # Test kiya tha: Q2 vs Q10 = 100, Q1 vs Q5 = 75. 90 dono ko sahi classify karta hai.
    score = exact_score(q1, q2)
    return score >= threshold, score


# ---------- Layer 2: Semantic match ----------

def cosine_sim(emb1, emb2):
    # Embeddings already normalized hain (embeddings.py mein), isliye
    # cosine similarity = simple dot product. Fast hai.
    return float(np.dot(emb1, emb2))


def is_semantic_dup(emb1, emb2, threshold=0.80):
    # 0.80 threshold - isko test karke chuna.
    # 0.70 rakha to Q1 vs Q5 (0.79) ko bhi duplicate bol raha tha - galat
    # 0.90 rakha to Q1 vs Q2 (0.94) sahi, par Q8 wala case miss ho raha
    # 0.80 sweet spot hai test data pe.
    sim = cosine_sim(emb1, emb2)
    return sim >= threshold, sim


# ---------- Combine karke final verdict ----------

def classify(q1, q2, emb1, emb2):
    # Pehle exact check, phir semantic. Exact pehle kyunki wo zyada confident hai.
    exact_hit, ex_score = is_exact_dup(q1, q2)
    sem_hit, sem_score = is_semantic_dup(emb1, emb2)

    if exact_hit:
        label = "EXACT_DUPLICATE"
    elif sem_hit:
        label = "SEMANTIC_DUPLICATE"
    else:
        label = "DIFFERENT"

    return {
        "label": label,
        "exact_score": round(ex_score, 2),
        "semantic_score": round(sem_score, 4),
    }