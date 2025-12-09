import subprocess

def get_feedback_from_ollama(prompt: str, model_name: str = "llama3") -> str:
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

def build_feedback_prompt(
    tier: str,
    match_score: float,
    position: int,
    total_candidates: int,
    confidence: float,
    job_text: str,
    resume_text: str,
) -> str:
    position_context = f"Ranked {position} out of {total_candidates} candidates."
    return f"""
You are an experienced recruiter giving personalized feedback on a resume.

CANDIDATE SUMMARY:
Tier: {tier}
Match Score: {match_score:.1%}
Relative Position: {position_context}
Confidence: {confidence:.1%}

TONE GUIDANCE:
{"This is a strong candidate. Focus on what makes them stand out and offer minor polish suggestions."
 if tier == "High" else
 "This candidate is in the middle range. Highlight their potential and the specific improvements needed to reach the top tier."
 if tier == "Medium" else
 "This candidate is a lower match. Be constructive and identify the key gaps preventing a better match."
}

JOB DESCRIPTION:
{job_text[:1400]}

RESUME:
{resume_text[:1400]}

Write 3–5 sentences of feedback.
You must explicitly mention the candidate's tier (for example, "As a High-tier candidate…", "Given your Medium-tier rating…", or "As a Low-tier candidate…").
Be concrete about strengths, weaknesses, and next steps to better align with the job.
""".strip()
