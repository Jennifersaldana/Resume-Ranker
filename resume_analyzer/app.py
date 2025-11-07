import streamlit as st
from utils.pdf_reader import extract_text_from_pdf

# --- App Configuration ---
st.set_page_config(page_title="AI Resume Ranker", layout="wide")

st.title("AI Resume Ranker")
st.write("Upload multiple resumes and one job description. The LLM will evaluate and rank them based on your chosen criteria.")

# --- File Uploaders ---
resume_files = st.file_uploader(
    "Upload up to 5 resumes (PDF format)",
    type=["pdf"],
    accept_multiple_files=True
)

job_file = st.file_uploader(
    "Upload the Job Description (PDF format)",
    type=["pdf"]
)

# --- Optional Instructions ---
custom_instruction = st.text_area(
    "Optional: Add judging instructions (e.g., 'Focus on GPA and technical skills', 'Prefer LSU graduates', etc.)",
    height=120
)

# --- Analyze Button ---
if st.button("Analyze Resumes"):

    if not job_file or not resume_files:
        st.warning("Please upload both the job description and at least one resume.")
    else:
        with st.spinner("Analyzing resumes... ⏳"):
            job_text = extract_text_from_pdf(job_file)
            results = []

            # Fake AI analysis (replace this later)
            for file in resume_files:
                resume_text = extract_text_from_pdf(file)
                # Placeholder for LLM logic 
                fake_feedback = f"""
                **Analysis for {file.name}:**
                - ✅ Strengths: Strong academic background, relevant coursework.
                - ⚠️ Weaknesses: Missing specific keywords from the job description.
                - 💡 Suggestion: Add measurable achievements and tailor your skills section.
                """
                results.append((file.name, fake_feedback))

        st.success("Analysis complete!")

        # --- Display Results ---
        for name, feedback in results:
            st.markdown(f"###  {name}")
            st.markdown(feedback)
            st.divider()
