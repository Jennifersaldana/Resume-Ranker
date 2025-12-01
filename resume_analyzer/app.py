import streamlit as st
import numpy as np
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt

# Custom utility modules
from utils.embedding import load_embedding_model
from utils.ranking import rank_resumes
from utils.bias import detect_bias_batch
from utils.neutralization import neutralize_batch
from utils.suggestions import generate_suggestions 
from utils.feedback import generate_resume_feedback

# ===========================================
# PAGE CONFIG
# ===========================================
st.set_page_config(page_title="AI Resume Ranker", layout="wide")
st.title("Resume Ranker Dashboard")

# ===========================================
# CACHE MODEL LOADING
# ===========================================
@st.cache_resource
def load_model():
    return load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

# ===========================================
# PDF Extraction
# ===========================================
def extract_pdf_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


# ===========================================
# SESSION RESET
# ===========================================
if "reset" not in st.session_state:
    st.session_state.reset = False

def reset_dashboard():
    st.session_state.reset = True

if st.session_state.reset:
    st.session_state.clear()
    st.rerun()

# ===========================================
# USER INPUT SECTION
# ===========================================
st.write("Upload up to five resumes and one job description. The AI model will rank resumes, detect bias, and compute fairness metrics.")

resumes = st.file_uploader("Upload 1–5 Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
job_desc = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
criteria = st.text_area(
    "Optional custom judging criteria:",
    placeholder="Example: Focus on AWS skills, internships, and GPA..."
)

run_button = st.button("Run Ranking")
st.button("Reset Dashboard", on_click=reset_dashboard)

# ===========================================
# MAIN PIPELINE
# ===========================================
if run_button:

    # --- Validate inputs ---
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

        # ----------------------------------------------------------------------------------
        # 1) Extract PDF text
        # ----------------------------------------------------------------------------------
        jd_text = extract_pdf_text(job_desc)
        resume_texts = []
        resume_names = []

        for f in resumes:
            resume_names.append(f.name)
            resume_texts.append(extract_pdf_text(f))

        # ----------------------------------------------------------------------------------
        # 2) Phase 11: Ranking Algorithm
        # ----------------------------------------------------------------------------------
        ranked = rank_resumes(resume_texts, jd_text, model)

        # Add name + full text back to each result
        for item in ranked:
            idx = item["resume_index"]
            item["resume_name"] = resume_names[idx]
            item["resume_text"] = resume_texts[idx]

        ranking_df = pd.DataFrame({
            "Rank": [r["rank"] for r in ranked],
            "Resume": [r["resume_name"] for r in ranked],
            "Score": [round(r["similarity_score"], 4) for r in ranked],
        })

    st.success("Analysis complete!")

    # --------------------------------------------------------------------------------------
    # RANKINGS TABLE
    # --------------------------------------------------------------------------------------
    st.header("Resume Rankings")
    st.table(ranking_df)

    # --------------------------------------------------------------------------------------
    # 3) Phase 6: Bias Detection
    # --------------------------------------------------------------------------------------
    st.header("Bias & Fairness Metrics")
    st.write("🟢 Low Bias, 🟡 Medium Bias, 🔴 High Bias")

    bias_scores = detect_bias_batch([r["resume_text"] for r in ranked])

    bias_df = pd.DataFrame({
        "Resume": [r["resume_name"] for r in ranked],
        "Bias Score": bias_scores
    })

    # Display per-resume bias
    for i, score in enumerate(bias_scores):
        if score < 2:
            badge = "🟢 Low Bias"
        elif score < 5:
            badge = "🟡 Medium Bias"
        else:
            badge = "🔴 High Bias"

        st.write(f"**{ranked[i]['resume_name']}** — Score: `{round(score, 2)}` {badge}")

    # --------------------------------------------------------------------------------------
    # 4) Neutralization (Fairness Evaluation)
    # --------------------------------------------------------------------------------------
    st.header("Fairness Index")

    # Create neutralized versions
    neutralized_texts = neutralize_batch([r["resume_text"] for r in ranked])

    # Re-rank with neutralized text
    neutralized_ranked = rank_resumes(neutralized_texts, jd_text, model)

    original_ranks = np.array([r["rank"] for r in ranked])
    new_ranks = np.array([r["rank"] for r in neutralized_ranked])

    fairness_index = 1 - (np.abs(original_ranks - new_ranks).mean() / len(ranked))

    st.metric(label="Fairness Index (0–1)", value=round(float(fairness_index), 3))

    # --------------------------------------------------------------------------------------
    # 5) Neutralized Ranking Table
    # --------------------------------------------------------------------------------------
    st.header("Neutralized Ranking Comparison")

    neutralized_df = pd.DataFrame({
        "Rank (Neutralized)": [r["rank"] for r in neutralized_ranked],
        "Resume": [resume_names[r["resume_index"]] for r in neutralized_ranked],
        "Score": [round(r["similarity_score"], 4) for r in neutralized_ranked],
    })

    st.table(neutralized_df)

    # --------------------------------------------------------------------------------------
    # 6) Charts & Visualizations
    # --------------------------------------------------------------------------------------
    st.header("Charts & Analytics")

    col1, col2 = st.columns(2)

    # Histogram — Similarity Scores
    with col1:
        st.markdown("### Similarity Score Distribution")
        fig, ax = plt.subplots()
        ax.hist(ranking_df["Score"], bins=len(ranking_df["Score"]), edgecolor="black")
        ax.set_xlabel("Similarity Score")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # Histogram — Bias Scores
    with col2:
        st.markdown("### Bias Score Distribution")
        fig, ax = plt.subplots()
        ax.hist(bias_scores, bins=len(bias_scores), edgecolor="black")
        ax.set_xlabel("Bias Score")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    # Bar Chart — Similarity
    with col3:
        st.markdown("### Similarity Scores (Bar Chart)")
        fig, ax = plt.subplots()
        ax.bar(ranking_df["Resume"], ranking_df["Score"], edgecolor="black")
        plt.xticks(rotation=45, ha='right')
        ax.set_ylabel("Score")
        st.pyplot(fig)

    # Bar Chart — Bias
    with col4:
        st.markdown("### Bias Scores (Bar Chart)")
        fig, ax = plt.subplots()
        ax.bar(bias_df["Resume"], bias_df["Bias Score"], edgecolor="black")
        plt.xticks(rotation=45, ha='right')
        ax.set_ylabel("Bias Score")
        st.pyplot(fig)

    # --------------------------------------------------------------------------------------
    # 7) Suggestions (Strength, Weaknesses)
    # --------------------------------------------------------------------------------------
    st.header("Strengths, Weaknesses, Suggestions")

    for r in ranked:
        st.write(f"### {r['resume_name']}")
        suggestions = generate_suggestions(r["resume_text"], jd_text)

        st.write("**Strengths:** ", suggestions["strengths"])
        st.write("**Weaknesses:** ", suggestions["weaknesses"])
        st.write("**Suggestions:**")
        for s in suggestions["suggestions"]:
            st.write(f"- {s}")

    # --------------------------------------------------------------------------------------
    # 8) Feedback with a LLM
    # --------------------------------------------------------------------------------------
    with st.expander(f"AI Reviewer Feedback for {r['resume_name']}"):
        feedback = generate_resume_feedback(r["resume_text"], jd_text)
        st.write(feedback)


    # --------------------------------------------------------------------------------------
    # 9) CSV Downloads
    # --------------------------------------------------------------------------------------
    st.subheader("Download Reports")

    st.download_button(
        label="Download Original Ranking CSV",
        data=ranking_df.to_csv(index=False),
        file_name="ranking_results.csv",
        mime="text/csv"
    )

    st.download_button(
        label="Download Neutralized Ranking CSV",
        data=neutralized_df.to_csv(index=False),
        file_name="neutralized_ranking.csv",
        mime="text/csv"
    )

    st.download_button(
        label="Download Bias Scores CSV",
        data=bias_df.to_csv(index=False),
        file_name="bias_scores.csv",
        mime="text/csv"
    )

    # --------------------------------------------------------------------------------------
    # 10) View Extracted Resume Text
    # --------------------------------------------------------------------------------------
    with st.expander("View Extracted Resume Text"):
        for r in ranked:
            st.write(f"### {r['resume_name']}")
            st.write(r["resume_text"])
