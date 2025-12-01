import re
import string

# -----------------------------
# STOPWORDS
# -----------------------------
STOPWORDS = {
    "the","a","an","and","or","of","to","for","with","on","in","as","is","be","are",
    "that","this","these","those","by","from","it","its","at","into","your","you",
    "their","they","we","our","ours","i"
}

# -----------------------------
# SKILL CATEGORIES
# -----------------------------
SKILL_CATEGORIES = {
    "Cloud": {"aws", "azure", "gcp", "cloud"},
    "Scripting": {"python", "powershell", "bash", "kql"},
    "Security Tools": {"splunk", "wireshark", "siem", "edr"},
    "Offensive Security": {"penetration", "testing", "vulnerability", "red", "exploit"},
    "Defensive Security": {"incident", "response", "detect", "analysis", "monitoring"},
    "Networking": {"networking", "tcp", "udp", "dns", "firewall"},
    "General Cybersecurity": {"cybersecurity", "security", "risk"},
}

# Flatten for lookup
ALL_SKILLS = {skill for cat in SKILL_CATEGORIES.values() for skill in cat}

# -----------------------------
# BASIC KEYWORD EXTRACTION
# -----------------------------
def extract_keywords(text):
    if not text:
        return set()

    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    return set(w for w in words if w not in STOPWORDS and len(w) > 2)

def extract_skills(text):
    words = extract_keywords(text)
    return set(w for w in words if w in ALL_SKILLS)

# -----------------------------
# CATEGORY DETECTION
# -----------------------------
def categorize_skills(skillset):
    categories = set()
    for cat, skills in SKILL_CATEGORIES.items():
        if skillset.intersection(skills):
            categories.add(cat)
    return categories

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def generate_suggestions(resume_text, job_text):

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)

    strengths = sorted(list(resume_skills.intersection(job_skills)))
    weaknesses = sorted(list(job_skills - resume_skills))

    # Skill categories
    resume_cats = categorize_skills(resume_skills)
    job_cats = categorize_skills(job_skills)

    missing_cats = job_cats - resume_cats

    suggestions = []

    # ------------------------------------------
    # CATEGORY-BASED SUGGESTIONS (NEW)
    # ------------------------------------------
    if missing_cats:
        cat_list = ", ".join(missing_cats)
        suggestions.append(f"Consider strengthening these areas emphasized in the job: {cat_list}.")

    # ------------------------------------------
    # SKILL-LEVEL SUGGESTIONS
    # ------------------------------------------
    if weaknesses:
        suggestions.append(f"Add or emphasize missing job-required skills: {', '.join(weaknesses)}.")

    # ------------------------------------------
    # STRENGTH-BASED SUGGESTIONS
    # ------------------------------------------
    if strengths:
        suggestions.append(f"Good alignment in: {', '.join(strengths[:5])}. Highlight these in your summary section.")

    # ------------------------------------------
    # DEPTH & EXPERIENCE ANALYSIS (NEW)
    # ------------------------------------------
    # simple heuristic: count how often skill appears
    depth_score = len(resume_skills)

    if depth_score < 5:
        suggestions.append("Increase technical depth by adding more detailed experience, tools, and frameworks used.")

    if len(resume_text) < 1200:
        suggestions.append("Consider expanding your experience descriptions with measurable impact and outcomes.")

    # ------------------------------------------
    # GENERAL SUGGESTIONS
    # ------------------------------------------
    suggestions.append("Quantify your achievements (%, time saved, incidents resolved).")
    suggestions.append("Use strong action verbs: led, implemented, automated, reduced, improved.")

    return {
        "strengths": strengths[:10],
        "weaknesses": weaknesses[:10],
        "missing_categories": list(missing_cats),
        "suggestions": suggestions[:5]  # limit length
    }
