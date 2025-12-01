import streamlit as st
import numpy as np
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer
import os
import matplotlib.pyplot as plt
import io

# Utility imports
from utils.ranking import rank_resumes
from utils.bias import detect_bias_batch
from utils.neutralization import neutralize_batch
from utils.analysis import generate_suggestions
# from utils.feedback import generate_resume_feedback   # optional


# ------------------------
# Load Embedding Model
# ------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()


# ------------------------
# PDF → Text Extractor
# ------------------------
def extract_pdf_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


# ------------------------
# Reset Button
# ------------------------
if "reset" not in st.session_state:
    st.session_state.reset = False

def reset_dashboard():
    st.session_state.reset = True


st.set_page_config(page_title="AI Resume Ranker", layout="wide")
st.title("AI Resume Ranker")

st.button("🔄 Reset Dashboard", on_click=reset_dashboard)

if st.session_state.reset:
    st.session_state.clear()
    st.rerun()

st.write("Upload up to five resumes and one job description to generate rankings.")


# Upload Inputs
resumes = st.file_uploader("Upload 1–5 Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
job_desc = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

criteria = st.text_area(
    "Optional: Custom judging criteria",
    placeholder="Example: Focus more on project experience, GPA, AWS skills, etc."
)

run_button = st.button("Run Ranking")


# ------------------------
# When user clicks RUN
# ------------------------
if run_button:

    # Basic Validation
    if not resumes:
        st.error("Please upload at least one resume.")
        st.stop()
    if not job_desc:
        st.error("Please upload a job description.")
        st.stop()
    if len(resumes) > 5:
        st.error("You can upload up to 5 resumes.")
        st.stop()

    with st.spinner("Analyzing resumes... ⏳"):

        jd_text = extract_pdf_text(job_desc)

        resume_texts = []
        resume_names = []
        for file in resumes:
            resume_names.append(file.name)
            resume_texts.append(extract_pdf_text(file))

        # Ranking
        ranked = rank_resumes(resume_texts, jd_text, model)

        for i, r in enumerate(ranked):
            r["resume_name"] = resume_names[r["resume_index"]]
            r["resume_text"] = resume_texts[r["resume_index"]]

        ranking_df = pd.DataFrame({
            "Rank": [r["rank"] for r in ranked],
            "Resume": [r["resume_name"] for r in ranked],
            "Score": [round(r["similarity_score"], 4) for r in ranked],
        })


    # ------------------------
    # DISPLAY RANKINGS
    # ------------------------
    st.subheader("🏆 Resume Rankings")
    st.table(ranking_df)


    # ------------------------
    # BIAS METRICS
    # ------------------------
    st.subheader("⚖️ Bias & Fairness Metrics")

    bias_scores = detect_bias_batch([r["resume_text"] for r in ranked])

    # 🔥 FIX: Create bias_df here so charts can use it
    bias_df = pd.DataFrame({
        "Resume": [r["resume_name"] for r in ranked],
        "Bias Score": bias_scores
    })

    for i, score in enumerate(bias_scores):

        if score < 2:
            badge = "🟢 Low Bias"
        elif score < 5:
            badge = "🟡 Medium Bias"
        else:
            badge = "🔴 High Bias"

        st.write(f"**{ranked[i]['resume_name']}** — Score: `{round(score, 2)}` {badge}")



    # ------------------------
    # FAIRNESS INDEX CALCULATION
    # ------------------------
    st.subheader("📊 Fairness Index")

    neutralized_texts = neutralize_batch([r["resume_text"] for r in ranked])
    neutralized_ranked = rank_resumes(neutralized_texts, jd_text, model)

    original_ranks = np.array([r["rank"] for r in ranked])
    new_ranks = np.array([r["rank"] for r in neutralized_ranked])

    fairness_index = 1 - (np.abs(original_ranks - new_ranks).mean() / len(ranked))

    st.metric(label="Fairness Index (0–1)", value=round(float(fairness_index), 3))


    # ------------------------
    # DISPLAY NEUTRALIZED RANKINGS
    # ------------------------
    st.subheader("🔍 Neutralized Ranking Comparison")

    neutralized_df = pd.DataFrame({
        "Rank (Neutralized)": [r["rank"] for r in neutralized_ranked],
        "Resume": [resume_names[r["resume_index"]] for r in neutralized_ranked],
        "Score": [round(r["similarity_score"], 4) for r in neutralized_ranked],
    })

    st.table(neutralized_df)

    # ------------------------
    # CHARTS
    # ------------------------
    st.subheader("📈 Charts & Analytics")

    # 2 rows × 2 columns layout
    col1, col2 = st.columns(2)

    # -------------------------------
    # Histogram: Similarity Scores
    # -------------------------------
    with col1:
        st.markdown("### 📊 Similarity Score Distribution (Histogram)")
        fig, ax = plt.subplots()
        ax.hist(
            ranking_df["Score"],
            bins=len(ranking_df["Score"]),
            color="skyblue",
            edgecolor="black"
        )
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # -------------------------------
    # Histogram: Bias Scores
    # -------------------------------
    with col2:
        st.markdown("### 📊 Bias Score Distribution (Histogram)")
        fig, ax = plt.subplots()
        ax.hist(
            bias_scores,
            bins=len(bias_scores),
            color="salmon",
            edgecolor="black"
        )
        ax.set_xlabel("Bias Score")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # ===============================
    # 2nd Row (Bar Charts)
    # ===============================
    col3, col4 = st.columns(2)

    # -------------------------------
    # Bar Chart: Similarity
    # -------------------------------
    with col3:
        st.markdown("### 📊 Similarity Scores (Bar Chart)")
        fig, ax = plt.subplots()
        ax.bar(
            ranking_df["Resume"],
            ranking_df["Score"],
            color="skyblue",
            edgecolor="black"
        )
        ax.set_ylabel("Score")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    # -------------------------------
    # Bar Chart: Bias
    # -------------------------------
    with col4:
        st.markdown("### 📊 Bias Scores (Bar Chart)")
        fig, ax = plt.subplots()
        ax.bar(
            bias_df["Resume"],
            bias_df["Bias Score"],
            color="salmon",
            edgecolor="black"
        )
        ax.set_ylabel("Bias Score")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)



    # ------------------------
    # STRENGTH & WEAKNESSES
    # ------------------------
    st.subheader("💡 Strengths, Weaknesses, and Suggestions")

    for r in ranked:
        st.write(f"### {r['resume_name']}")

        suggestions = generate_suggestions(r["resume_text"], jd_text)

        st.write("**Strengths:**", suggestions["strengths"])
        st.write("**Weaknesses:**", suggestions["weaknesses"])
        st.write("**Suggestions:**")
        for s in suggestions["suggestions"]:
            st.write(f"- {s}")


    # ------------------------
    # DOWNLOAD CSV REPORTS
    # ------------------------
    st.subheader("📥 Download Reports")

    st.download_button(
        label="⬇️ Download Original Ranking CSV",
        data=ranking_df.to_csv(index=False),
        file_name="ranking_results.csv",
        mime="text/csv"
    )

    st.download_button(
        label="⬇️ Download Neutralized Ranking CSV",
        data=neutralized_df.to_csv(index=False),
        file_name="neutralized_ranking.csv",
        mime="text/csv"
    )


    st.download_button(
        label="⬇️ Download Bias Scores CSV",
        data=bias_df.to_csv(index=False),
        file_name="bias_scores.csv",
        mime="text/csv"
    )


    # ------------------------
    # RAW TEXT VIEW
    # ------------------------
    with st.expander("📄 View Extracted Resume Text"):
        for r in ranked:
            st.write(f"### {r['resume_name']}")
            st.write(r["resume_text"])
