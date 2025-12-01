import openai

# Make sure you set your environment variable:
# export OPENAI_API_KEY="yourkey"

def generate_resume_feedback(resume_text, job_text):
    """
    Uses an LLM to generate professional, human-like feedback for a resume.
    More detailed than suggestions.py.
    """

    prompt = f"""
You are a cybersecurity hiring manager reviewing a resume.

Job Description:
----------------
{job_text}

Candidate Resume:
-----------------
{resume_text}

TASKS:
1. Provide a **professional evaluation** of the resume.
2. Identify strengths relevant to cybersecurity or IT.
3. Identify missing skills or gaps specific to the job.
4. Suggest **clear improvements**, including:
   - what to add
   - what to rewrite
   - what to emphasize
   - formatting or structure tips
5. Keep the tone supportive and professional.

Return your response in the following format:

### Overall Assessment
(text)

### Strengths
- (bullet points)

### Areas for Improvement
- (bullet points)

### Actionable Recommendations
- (bullet points)

Go ahead and produce the feedback now.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",   # or gpt-4.1, etc.
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=600,
    )

    return response["choices"][0]["message"]["content"]
