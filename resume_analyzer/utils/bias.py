# Phase 6
import pandas as pd
import re
from collections import Counter

# ------------------------------------------------------------
# Load Bias Dictionaries
# ------------------------------------------------------------
def load_bias_indicators():
    return {
        "prestige_institutions": [
            'harvard','stanford','mit','princeton','yale','columbia',
            'university of chicago','caltech','duke','brown'
        ],
        "prestige_companies": [
            'google','amazon','meta','facebook','apple','microsoft','tesla'
        ],
        "prestige_locations": [
            'silicon valley','nyc','boston','san francisco','palo alto'
        ],
        "gender_indicators": {
            "pronouns": [" he ", " she ", " his ", " her "],
            "titles": [" mr ", " ms ", " mrs ", " miss "],
            "explicit_gender": [" male ", " female ", " man ", " woman "],
            "protected_characteristics": [
                "hispanic", "latino", "disabled", "veteran"
            ]
        }
    }

# ------------------------------------------------------------
# Bias Detection for One Text
# ------------------------------------------------------------
def detect_bias_in_text(text, indicators):
    if not text:
        return {"total_bias_score": 0}

    text_lower = " " + text.lower() + " "

    score = 0

    # Prestige institutions
    for inst in indicators["prestige_institutions"]:
        if inst in text_lower:
            score += 2

    # Prestige companies
    for comp in indicators["prestige_companies"]:
        if comp in text_lower:
            score += 1.5

    # Prestige locations
    for loc in indicators["prestige_locations"]:
        if loc in text_lower:
            score += 1

    # Gender + protected indicators
    for cat, terms in indicators["gender_indicators"].items():
        for t in terms:
            if t in text_lower:
                score += 1

    return {"total_bias_score": score}


# ------------------------------------------------------------
# Batch Processing for Resumes
# ------------------------------------------------------------
def detect_bias_batch(resume_texts):
    indicators = load_bias_indicators()
    results = []

    for text in resume_texts:
        score = detect_bias_in_text(text, indicators)
        results.append(score["total_bias_score"])

    return results
