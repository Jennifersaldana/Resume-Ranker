import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
import warnings

from features.text_utils import extract_pdf_text, tokenize_words, extract_skills
from features.bias_utils import load_bias_indicators, detect_bias_in_text
from features.scoring_utils import get_match_tier, tier_badge_html, compute_confidences
from features.visuals import plot_score_histogram, plot_wordcount_vs_score, plot_similarity_heatmap
from features.llm_feedback import get_feedback_from_ollama, build_feedback_prompt

from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# =============================================================
# PAGE CONFIG & STYLES
# =============================================================
st.set_page_config(page_title="AI Resume Ranker", layout="wide")

st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #4B0082;
        margin-bottom: 0px;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .subtitle {
        font-size: 16px;
        color: #555555;
        margin-bottom: 20px;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .metric-card {
        background-color: #f7f2ff;
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid #e0d4ff;
        margin-bottom: 8px;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .tier-high {
        background-color: #e6ffed;
        color: #046c4e;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }
    .tier-medium {
        background-color: #fff7e6;
        color: #8a5a00;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }
    .tier-low {
        background-color: #ffecec;
        color: #a10000;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }
    .skill-chip {
        display: inline-block;
        background-color: #eef2ff;
        color: #312e81;
        padding: 3px 10px;
        border-radius: 999px;
        margin: 2px 4px 4px 0;
        font-size: 11px;
        font-weight: 500;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .skill-chip-missing {
        display: inline-block;
        background-color: #fef2f2;
        color: #991b1b;
        padding: 3px 10px;
        border-radius: 999px;
        margin: 2px 4px 4px 0;
        font-size: 11px;
        font-weight: 500;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='main-title'>AI Resume Ranker Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Upload 1–5 resumes and a job description. The system ranks candidates, "
    "extracts skills, highlights missing skills, estimates bias indicators, and visualizes metrics.</div>",
    unsafe_allow_html=True,
)

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
# USER INPUTS
# =============================================================
st.subheader("Upload Files")

col_u1, col_u2 = st.columns([2, 1])
with col_u1:
    resumes = st.file_uploader(
        "Upload 1–5 Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )
with col_u2:
    job_desc = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

run_button = st.button("Run Ranking")
st.button("Reset Dashboard", on_click=reset_dashboard)

# =============================================================
# MAIN PIPELINE
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

    with st.spinner("Analyzing resumes and building dashboard…"):

        job_text = extract_pdf_text(job_desc)
        if not job_text.strip():
            st.error("Could not extract text from job description PDF.")
            st.stop()

        job_emb = embedder.encode(job_text)

        jd_tokens = tokenize_words(job_text)
        jd_keyword_set = set(jd_tokens)
        jd_skill_set = extract_skills(job_text)

        indicators = load_bias_indicators()

        ranking_rows = []
        resume_text_map = {}
        resume_skill_map = {}
        resume_bias_map = {}
        resume_embeddings = []

        for r in resumes:
            resume_text = extract_pdf_text(r)
            if not resume_text.strip():
                continue

            resume_text_map[r.name] = resume_text

            res_emb = embedder.encode(resume_text)
            resume_embeddings.append(res_emb)

            cos_sim = cosine_similarity([res_emb], [job_emb])[0][0]
            features = np.concatenate([res_emb, job_emb, [cos_sim]])
            score = float(model.predict([features])[0])

            tokens = tokenize_words(resume_text)
            word_count = len(tokens)

            resume_skills = extract_skills(resume_text)
            resume_skill_map[r.name] = resume_skills

            if len(jd_keyword_set) > 0:
                keyword_matches = len(set(tokens) & jd_keyword_set)
                keyword_match_ratio = keyword_matches / len(jd_keyword_set)
            else:
                keyword_match_ratio = 0.0

            if len(jd_skill_set) > 0:
                skills_matched = len(resume_skills & jd_skill_set)
                missing_skills_count = len(jd_skill_set - resume_skills)
            else:
                skills_matched = 0
                missing_skills_count = 0

            bias_counts = detect_bias_in_text(resume_text, indicators)
            resume_bias_map[r.name] = bias_counts

            ranking_rows.append({
                "Resume": r.name,
                "Predicted Score": score,
                "Cosine Similarity": cos_sim,
                "Word Count": word_count,
                "Keyword Match Ratio": keyword_match_ratio,
                "Skills Matched": skills_matched,
                "Missing Skills Count": missing_skills_count,
                "Bias Score": bias_counts["bias_score"],
                "Prestige Institutions": bias_counts["prestige_institutions"],
                "Prestige Companies": bias_counts["prestige_companies"],
                "Prestige Locations": bias_counts["prestige_locations"],
                "Gender Terms": bias_counts["gender_terms"],
            })

        if not ranking_rows:
            st.error("No readable resumes were processed. Check your PDFs.")
            st.stop()

        ranking_rows = sorted(ranking_rows, key=lambda x: x["Predicted Score"], reverse=True)

        scores = [row["Predicted Score"] for row in ranking_rows]
        confidences = compute_confidences(scores)

        for row, conf in zip(ranking_rows, confidences):
            row["Confidence"] = conf
            row["Match Tier"] = get_match_tier(row["Predicted Score"])

        df_results = pd.DataFrame(ranking_rows)

        if len(resume_embeddings) > 1:
            emb_matrix = np.vstack(resume_embeddings)
            sim_matrix = cosine_similarity(emb_matrix, emb_matrix)
        else:
            sim_matrix = None

    # =========================================================
    # TABS
    # =========================================================
    tab_rank, tab_feedback, tab_skills, tab_bias, tab_metrics = st.tabs(
        ["Ranking Overview", "LLM Feedback", "Skills Analysis", "Bias Indicators", "Metrics & Visuals"]
    )

    # ---------------------------------------------------------
    # TAB 1: RANKING OVERVIEW
    # ---------------------------------------------------------
    with tab_rank:
        st.markdown("### Overall Ranking")

        top_row = df_results.iloc[0]
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.markdown(
                f"<div class='metric-card'>Top Resume<br><b>{top_row['Resume']}</b></div>",
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f"<div class='metric-card'>Best Match Score<br><b>{top_row['Predicted Score']:.2f}</b></div>",
                unsafe_allow_html=True,
            )
        with col_m3:
            st.markdown(
                f"<div class='metric-card'>Top Similarity<br><b>{top_row['Cosine Similarity']:.2f}</b></div>",
                unsafe_allow_html=True,
            )
        with col_m4:
            st.markdown(
                f"<div class='metric-card'>Tier<br>{tier_badge_html(top_row['Match Tier'])}</div>",
                unsafe_allow_html=True,
            )

        df_display = df_results.copy()
        df_display["Confidence (%)"] = (df_display["Confidence"] * 100).round(1)
        df_display["Keyword Match (%)"] = (df_display["Keyword Match Ratio"] * 100).round(1)
        df_display = df_display[
            [
                "Resume",
                "Match Tier",
                "Predicted Score",
                "Cosine Similarity",
                "Confidence (%)",
                "Keyword Match (%)",
                "Word Count",
                "Skills Matched",
                "Missing Skills Count",
            ]
        ]

        st.dataframe(df_display, use_container_width=True)

        st.markdown("#### Score Bar Chart")
        st.bar_chart(df_results.set_index("Resume")["Predicted Score"])

    # ---------------------------------------------------------
    # TAB 2: LLM FEEDBACK
    # ---------------------------------------------------------
    with tab_feedback:
        st.markdown("### Personalized Resume Feedback (Local LLM)")
        total_candidates = len(ranking_rows)

        for position, row in enumerate(ranking_rows, start=1):
            resume_name = row["Resume"]
            match_score = row["Predicted Score"]
            tier = row["Match Tier"]
            confidence = row["Confidence"]
            resume_text = resume_text_map.get(resume_name, "")

            prompt = build_feedback_prompt(
                tier=tier,
                match_score=match_score,
                position=position,
                total_candidates=total_candidates,
                confidence=confidence,
                job_text=job_text,
                resume_text=resume_text,
            )

            feedback = get_feedback_from_ollama(prompt)

            st.markdown(f"#### Feedback for **{resume_name}**")
            st.write(feedback)
            st.markdown("---")

    # ---------------------------------------------------------
    # TAB 3: SKILLS ANALYSIS
    # ---------------------------------------------------------
    with tab_skills:
        st.markdown("### Skills and Missing Skills")

        st.markdown("**Skills extracted from job description:**")
        if jd_skill_set:
            jd_skill_html = "".join(
                f"<span class='skill-chip'>{s}</span>" for s in sorted(jd_skill_set)
            )
            st.markdown(jd_skill_html, unsafe_allow_html=True)
        else:
            st.write("No clear skills extracted from job description with the current heuristic.")

        st.markdown("---")

        for row in ranking_rows:
            resume_name = row["Resume"]
            resume_skills = resume_skill_map.get(resume_name, set())
            missing_skills = jd_skill_set - resume_skills if jd_skill_set else set()

            st.markdown(f"#### {resume_name}")
            st.markdown(tier_badge_html(row["Match Tier"]), unsafe_allow_html=True)

            st.markdown("**Detected skills in this resume:**")
            if resume_skills:
                html_skills = "".join(
                    f"<span class='skill-chip'>{s}</span>" for s in sorted(resume_skills)
                )
                st.markdown(html_skills, unsafe_allow_html=True)
            else:
                st.write("No distinct skills detected with the current heuristic.")

            st.markdown("**Skills from job description missing in this resume:**")
            if missing_skills:
                html_missing = "".join(
                    f"<span class='skill-chip-missing'>{s}</span>" for s in sorted(missing_skills)
                )
                st.markdown(html_missing, unsafe_allow_html=True)
            else:
                st.write("No missing skills identified relative to the job description.")
            st.markdown("---")

    # ---------------------------------------------------------
    # TAB 4: BIAS INDICATORS
    # ---------------------------------------------------------
    with tab_bias:
        st.markdown("### Bias Indicator Summary")

        bias_cols = [
            "Resume",
            "Bias Score",
            "Prestige Institutions",
            "Prestige Companies",
            "Prestige Locations",
            "Gender Terms",
        ]
        bias_df = df_results[bias_cols].copy()

        col_b1, col_b2, col_b3 = st.columns(3)
        avg_bias = bias_df["Bias Score"].mean()
        max_bias = bias_df["Bias Score"].max()
        any_gender = (bias_df["Gender Terms"] > 0).sum()

        with col_b1:
            st.markdown(
                f"<div class='metric-card'>Average Bias Score<br><b>{avg_bias:.2f}</b></div>",
                unsafe_allow_html=True,
            )
        with col_b2:
            st.markdown(
                f"<div class='metric-card'>Maximum Bias Score<br><b>{max_bias:.2f}</b></div>",
                unsafe_allow_html=True,
            )
        with col_b3:
            st.markdown(
                f"<div class='metric-card'>Resumes with gendered terms<br><b>{any_gender}</b></div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### Bias Indicators per Resume")
        st.dataframe(bias_df, use_container_width=True)

        st.info(
            "These indicators are heuristic and highlight possible sources of bias, "
            "such as prestige-heavy backgrounds or gendered language."
        )

    # ---------------------------------------------------------
    # TAB 5: METRICS & VISUALS
    # ---------------------------------------------------------
    with tab_metrics:
        st.markdown("### Score and Similarity Metrics")

        col_v1, col_v2 = st.columns(2)

        with col_v1:
            st.markdown("#### Predicted Score Distribution")
            fig1 = plot_score_histogram(df_results)
            st.pyplot(fig1)

        with col_v2:
            st.markdown("#### Word Count vs Predicted Score")
            fig2 = plot_wordcount_vs_score(df_results)
            st.pyplot(fig2)

        st.markdown("---")
        st.markdown("#### Resume-to-Resume Similarity Heatmap")

        if sim_matrix is not None and len(df_results) > 1:
            fig3 = plot_similarity_heatmap(df_results["Resume"].tolist(), sim_matrix)
            st.pyplot(fig3)
        else:
            st.write("Not enough resumes to show a similarity heatmap.")
