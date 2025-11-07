from openai import OpenAI

client = OpenAI()

def analyze_resume(resume_text, job_text):
    prompt = f"""
    Compare this resume to the job description.
    Highlight missing skills, weak bullet points, and how to improve.
    Give a match percentage and specific feedback.

    Resume:
    {resume_text}

    Job Description:
    {job_text}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
