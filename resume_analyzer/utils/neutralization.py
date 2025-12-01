import re
from utils.bias import load_bias_indicators

def neutralize_text(text, reduction_factor=0.3):
    if not text:
        return text

    indicators = load_bias_indicators()
    out = text.lower()

    # Replace prestige names with generic placeholders
    for inst in indicators["prestige_institutions"]:
        out = out.replace(inst, "university")

    for comp in indicators["prestige_companies"]:
        out = out.replace(comp, "company")

    for loc in indicators["prestige_locations"]:
        out = out.replace(loc, "location")

    # Remove gender terms
    for g in indicators["gender_terms"]:
        out = out.replace(g.strip(), "")

    return out

def neutralize_batch(text_list):
    return [neutralize_text(t) for t in text_list]
