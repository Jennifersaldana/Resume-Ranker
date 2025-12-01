def generate_suggestions(resume_text, job_text):
    """
    Very simple evaluation:
        - If the resume contains job keywords → strength
        - Missing keywords → weakness
    """

    job_keywords = set(job_text.lower().split())
    resume_words = set(resume_text.lower().split())

    strengths = list(resume_words.intersection(job_keywords))
    weaknesses = list(job_keywords - resume_words)

    return {
        "strengths": strengths[:10],
        "weaknesses": weaknesses[:10],
        "suggestions": [
            f"Add more emphasis on: {', '.join(weaknesses[:5])}",
            "Consider quantifying achievements.",
            "Highlight technical skills required by job description."
        ]
    }
