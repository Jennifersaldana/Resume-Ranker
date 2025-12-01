import re

def neutralize_text(text, indicators, reduction_factor=0.3):
    """
    Reduce prestige indicators + remove protected attributes.
    """
    if not text:
        return text

    new_text = text.lower()

    # Replace prestigious terms with generic substitutes
    for inst in indicators["prestige_institutions"]:
        new_text = new_text.replace(inst, "university")

    for comp in indicators["prestige_companies"]:
        new_text = new_text.replace(comp, "company")

    for loc in indicators["prestige_locations"]:
        new_text = new_text.replace(loc, "location")

    # Remove gender / protected terms
    for cat, terms in indicators["gender_indicators"].items():
        for t in terms:
            new_text = new_text.replace(t.strip(), "")

    return new_text


def neutralize_batch(resume_texts):
    from utils.bias import load_bias_indicators
    indicators = load_bias_indicators()

    neutralized = []
    for text in resume_texts:
        neutralized.append(neutralize_text(text, indicators))
    return neutralized
