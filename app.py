# Streamlit UI for the assignment.
# Three tabs:
#   1. Generate - LLM se naya question banwa, phir detector se check
#   2. Test Corpus - Q1-Q10 dikhao expected classifications ke saath
#   3. Compare Pair - do questions daalo, duplicate check karo
#
# Ye UI demo video ke liye hai taaki reviewer khud chala ke dekh sake.

import streamlit as st
import pandas as pd

from src.embeddings import embed_many, embed_one
from src.detector import classify
from src.generator import generate_question, pick_diverse_questions


st.set_page_config(page_title="Question Generator + Duplicate Detector", layout="wide")

st.title("AI Question Generation with Duplicate Prevention")
st.caption("LLM generates new questions · Detector catches duplicates")


# Data + embeddings ko cache karo taaki baar baar load na ho.
# Streamlit cache ka matlab: same function dobara call hui to cached result milega.
@st.cache_data
def load_corpus():
    df = pd.read_csv("data/questions.csv")
    return df


@st.cache_resource
def load_embeddings(questions_tuple):
    # cache_resource kyunki ye numpy array return karta hai (heavy object)
    return embed_many(list(questions_tuple))


df = load_corpus()
embs = load_embeddings(tuple(df["question"].tolist()))


# ---------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Generate & Verify", "Test Corpus", "Compare Two Questions"])


# ===============================================================
# Tab 1 - Generate new question + check for duplicates
# ===============================================================
with tab1:
    st.header("Generate a New Question")
    st.write(
        "Ye tab LLM se naya question generate karwata hai. Saath mein hum "
        "existing questions LLM ko dikhate hain taaki wo duplicate na banaye."
    )

    topic = st.text_input("Topic", value="reflection of light")

    col1, col2 = st.columns([1, 3])
    with col1:
        gen_btn = st.button("Generate", type="primary")

    if gen_btn:
        # Step 1 - context select karo
        context = pick_diverse_questions(df, embs, k=5)

        with st.expander("5 questions LLM ko context mein bhej rahe hain", expanded=False):
            for q in context:
                st.write(f"- {q}")

        # Step 2 - generate
        with st.spinner("Groq se question generate kar rahe hain..."):
            try:
                new_q = generate_question(topic, context)
            except Exception as e:
                st.error(f"Generation fail hui: {e}")
                st.stop()

        st.success("Question generate ho gaya:")
        st.markdown(f"### _{new_q}_")

        # Step 3 - detector se check
        st.subheader("Duplicate Check")
        st.write("Ab dekhte hain ki ye naya question kisi existing se match karta hai ya nahi.")

        new_emb = embed_one(new_q)
        rows = []
        is_dup = False
        for i in range(len(df)):
            res = classify(new_q, df.iloc[i]["question"], new_emb, embs[i])
            rows.append(
                {
                    "Existing ID": df.iloc[i]["id"],
                    "Existing Question": df.iloc[i]["question"],
                    "Label": res["label"],
                    "Exact Score": res["exact_score"],
                    "Semantic Score": res["semantic_score"],
                }
            )
            if res["label"] != "DIFFERENT":
                is_dup = True

        result_df = pd.DataFrame(rows)
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        if is_dup:
            st.error("Result: Generated question DUPLICATE hai - avoidance fail hui")
        else:
            st.success("Result: Generated question NAYA hai - avoidance kaam kiya")


# ===============================================================
# Tab 2 - Show test corpus with expected classifications
# ===============================================================
with tab2:
    st.header("Test Corpus (Q1 to Q10)")
    st.write(
        "Ye wahi dataset hai jo assignment ne diya tha. Expected classification "
        "kolumn mein likhi hai, aur detector ka actual output bhi generate kar sakte ho."
    )

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Run Detector on Q1 vs All")
    if st.button("Run Evaluation"):
        q1_text = df.iloc[0]["question"]
        q1_emb = embs[0]

        rows = []
        for i in range(1, len(df)):
            res = classify(q1_text, df.iloc[i]["question"], q1_emb, embs[i])
            rows.append(
                {
                    "Compare": f"Q1 vs {df.iloc[i]['id']}",
                    "Question": df.iloc[i]["question"],
                    "Expected": df.iloc[i]["expected"],
                    "Detector says": res["label"],
                    "Semantic": res["semantic_score"],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ===============================================================
# Tab 3 - Compare two custom questions
# ===============================================================
with tab3:
    st.header("Compare Two Questions")
    st.write(
        "Apne do questions daal ke check karo ki detector kya kehta hai. "
        "Quick sanity check ke liye useful hai."
    )

    q1 = st.text_input("Question 1", value="What is reflection of light?")
    q2 = st.text_input("Question 2", value="Define the reflection of light.")

    if st.button("Compare"):
        e1 = embed_one(q1)
        e2 = embed_one(q2)
        res = classify(q1, q2, e1, e2)

        st.markdown(f"**Label:** `{res['label']}`")
        col_a, col_b = st.columns(2)
        col_a.metric("Exact Score", f"{res['exact_score']:.2f}")
        col_b.metric("Semantic Score", f"{res['semantic_score']:.4f}")

        st.caption(
            "Exact Score 90+ ka matlab same wording. Semantic 0.80+ ka matlab same meaning."
        )