# utils/feedback.py

import json
from openai import OpenAI

# ---------------------------------------------
# Generate LLM-based resume feedback
# ---------------------------------------------
def generate_resume_feedback(resume_text: str, job_text: str, api_key=None):
    """
    Uses an LLM to provide personalized resume improvement feedback.
    Returns:
        - missing_skills
        - weaknesses
        - improvement_tips
    """

    # Create client *inside* the function to prevent import-time API key error
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        client = OpenAI()  # Will use OPENAI_API_KEY env variable

    prompt = f"""
You are an expert technical recruiter evaluating resumes.

Compare the following resume and job description.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_text}

Identify the top missing skills AND give specific suggestions.

Return ONLY valid JSON in this format:
{{
  "missing_skills": ["skill1", "skill2", "skill3"],
  "weaknesses": ["issue1", "issue2"],
  "improvement_tips": ["tip1", "tip2", "tip3"]
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
    except Exception as e:
        # If user has no API key, return safe fallback
        return {
            "missing_skills": [],
            "weaknesses": [],
            "improvement_tips": [f"⚠️ Error: {e}. Add your OPENAI_API_KEY to enable LLM feedback."]
        }

    raw_output = response.choices[0].message.content

    try:
        return json.loads(raw_output)
    except:
        return {
            "missing_skills": [],
            "weaknesses": [],
            "improvement_tips": [raw_output]
        }
