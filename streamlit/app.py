import streamlit as st
import numpy as np
import pandas as pd
import pdfplumber
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import subprocess

# Suppress sklearn version warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(page_title="AI Resume Ranker", layout="wide")
st.title("AI Resume Ranker Dashboard")
st.write("Upload 1–5 resumes and a job description. The model predicts a match score for each resume.")

# =============================================================
# LOAD TRAINED MODEL
# =============================================================
@st.cache_resource
def load_model():
    try:
        return joblib.load("best_matching_model.pkl")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# =============================================================
# LOAD EMBEDDING MODEL
# =============================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# =============================================================
# PDF TEXT EXTRACTION
# =============================================================
def extract_pdf_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.strip()

# =============================================================
# RESET SESSION
# =============================================================
if "reset" not in st.session_state:
    st.session_state.reset = False

def reset_dashboard():
    st.session_state.reset = True

if st.session_state.reset:
    st.session_state.clear()
    st.rerun()

# =============================================================
# OLLAMA FEEDBACK
# =============================================================
def get_feedback_from_ollama(prompt, model_name="llama3"):
    """
    Runs a local Ollama model and sends a prompt to it.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.stdout.decode("utf-8").strip()

    except Exception as e:
        return f"Ollama error: {e}"

# =============================================================
# USER INPUT SECTION
# =============================================================
st.subheader("Upload Your Files")

resumes = st.file_uploader(
    "Upload 1–5 Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)
job_desc = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])


run_button = st.button("Run Ranking")
st.button("Reset Dashboard", on_click=reset_dashboard)

# =============================================================
# MAIN RANKING PIPELINE
# =============================================================
if run_button:

    if not resumes:
        st.error("Please upload at least one resume.")
        st.stop()

    if not job_desc:
        st.error("Please upload a job description.")
        st.stop()

    if len(resumes) > 5:
        st.error("You can upload up to 5 resumes.")
        st.stop()

    with st.spinner("Analyzing resumes... This may take a moment ⏳"):

        job_text = extract_pdf_text(job_desc)
        job_emb = embedder.encode(job_text)

        ranking_results = []

        for r in resumes:
            resume_text = extract_pdf_text(r)
            res_emb = embedder.encode(resume_text)

            cos_sim = cosine_similarity([res_emb], [job_emb])[0][0]

            features = np.concatenate([res_emb, job_emb, [cos_sim]])
            score = float(model.predict([features])[0])

            ranking_results.append({
                "Resume": r.name,
                "Predicted Score": score,
                "Cosine Similarity": cos_sim
            })

        ranking_results = sorted(
            ranking_results,
            key=lambda x: x["Predicted Score"],
            reverse=True
        )

    # =============================================================
    # DISPLAY RESULTS
    # =============================================================
    st.subheader("Ranking Results (Best → Worst Match)")

    df_results = pd.DataFrame(ranking_results)
    st.dataframe(df_results, use_container_width=True)

    st.subheader("Score Visualization")
    st.bar_chart(df_results.set_index("Resume")["Predicted Score"])

    st.success("Ranking complete!")


    # =============================================================
    # DISPLAY LLM FEEDBACK
    # =============================================================
    st.subheader("Personalized Resume Feedback (Local LLM)")
    total_candidates = len(ranking_results)

    for position, result in enumerate(ranking_results, start=1):
        resume_name = result["Resume"]
        match_score = result["Predicted Score"]

        # Determine tier
        if match_score >= 0.7:
            tier = "High"
        elif match_score >= 0.50:
            tier = "Medium"
        else:
            tier = "Low"

        # Tier tone
        if tier == "High":
            tier_context = (
                "This is a TOP-TIER candidate. Focus on what makes them stand out "
                "and offer minor polish suggestions."
            )
        elif tier == "Medium":
            tier_context = (
                "This is a MID-TIER candidate. Highlight their potential and the "
                "specific improvements needed to reach top tier."
            )
        else:
            tier_context = (
                "This is a LOWER-TIER candidate. Be constructive and identify the "
                "key gaps preventing a better match."
            )

        position_context = f"Ranked {position} out of {total_candidates} candidates."

        resume_obj = next(r for r in resumes if r.name == resume_name)
        resume_text = extract_pdf_text(resume_obj)

        # Final prompt
        prompt = f"""
    You are an expert HR recruiter giving personalized feedback on a resume.

    CANDIDATE RATING:
    Tier: {tier}
    Match Score: {match_score:.1%}
    Position: {position_context}

    FEEDBACK TONE: {tier_context}

    JOB DESCRIPTION:
    {job_text[:1200]}

    RESUME:
    {resume_text[:1200]}

    Give personalized feedback in 3–4 sentences based on their tier rating, match score, and ranking.
    You MUST explicitly mention the candidate's tier in your feedback 
    (e.g., “As a High-tier candidate…” or “Given your Medium-tier rating…”).

    Be specific to THIS resume and job. 
    No bullet points — write naturally.
    """

        feedback = get_feedback_from_ollama(prompt)

        st.markdown(f"### Feedback for **{resume_name}**")
        st.write(feedback)
        st.markdown("---")
