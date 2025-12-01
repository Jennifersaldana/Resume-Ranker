import re
import numpy as np

# ---------------------------------------------
# Load Bias Indicators (same as notebook)
# ---------------------------------------------
def load_bias_indicators():
    return {
        "prestige_institutions": [
            "harvard", "stanford", "mit", "princeton", "yale",
            "columbia", "upenn", "cornell", "dartmouth", "brown",
            "uchicago", "caltech", "duke"
        ],
        "prestige_companies": [
            "google", "microsoft", "amazon", "meta", "facebook",
            "apple", "netflix", "uber", "airbnb", "tesla"
        ],
        "prestige_locations": [
            "silicon valley", "palo alto", "menlo park",
            "new york city", "nyc", "manhattan",
            "london", "paris", "tokyo", "singapore", "sf"
        ],
        "gender_terms": [
            " he ", " she ", " his ", " her ", " him ", " hers ",
            " mr ", " mrs ", " ms ", " miss "
        ]
    }

# ---------------------------------------------
# Detect bias score for a single document
# ---------------------------------------------
def detect_bias(text, indicators):
    if not text:
        return 0

    t = " " + text.lower() + " "

    score = 0

    for inst in indicators["prestige_institutions"]:
        if inst in t:
            score += 2.0

    for comp in indicators["prestige_companies"]:
        if comp in t:
            score += 1.5

    for loc in indicators["prestige_locations"]:
        if loc in t:
            score += 1.0

    for g in indicators["gender_terms"]:
        if g in t:
            score += 1.0

    return score

# ---------------------------------------------
# Batch processing for Streamlit
# ---------------------------------------------
def detect_bias_batch(text_list):
    indicators = load_bias_indicators()
    return [detect_bias(t, indicators) for t in text_list]
