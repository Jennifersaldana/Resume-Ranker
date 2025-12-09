def load_bias_indicators():
    prestige_institutions = [
        "harvard", "stanford", "mit", "massachusetts institute of technology",
        "princeton", "yale", "columbia", "university of pennsylvania", "upenn",
        "cornell", "dartmouth", "brown", "university of chicago", "uchicago",
        "caltech", "duke", "oxford", "cambridge", "ivy league"
    ]
    prestige_companies = [
        "google", "microsoft", "apple", "amazon", "meta", "facebook",
        "goldman sachs", "mckinsey", "bain", "bcg", "netflix", "uber",
        "airbnb", "tesla", "spacex", "stripe", "palantir"
    ]
    prestige_locations = [
        "silicon valley", "palo alto", "menlo park", "mountain view",
        "new york city", "nyc", "manhattan", "london", "san francisco", "sf"
    ]
    gender_terms = [
        " he ", " she ", " his ", " her ", " him ", " hers ",
        " mr ", " mrs ", " ms ", " miss ", " gentleman ", " lady ",
        " male ", " female ", " men ", " women "
    ]
    return {
        "prestige_institutions": prestige_institutions,
        "prestige_companies": prestige_companies,
        "prestige_locations": prestige_locations,
        "gender_terms": gender_terms,
    }

def detect_bias_in_text(text: str, indicators: dict):
    if not text:
        return {
            "prestige_institutions": 0,
            "prestige_companies": 0,
            "prestige_locations": 0,
            "gender_terms": 0,
            "bias_score": 0.0,
        }

    lower = text.lower()
    padded = " " + lower + " "

    counts = {
        "prestige_institutions": 0,
        "prestige_companies": 0,
        "prestige_locations": 0,
        "gender_terms": 0,
    }

    for word in indicators["prestige_institutions"]:
        if word in lower:
            counts["prestige_institutions"] += 1

    for word in indicators["prestige_companies"]:
        if word in lower:
            counts["prestige_companies"] += 1

    for word in indicators["prestige_locations"]:
        if word in lower:
            counts["prestige_locations"] += 1

    for term in indicators["gender_terms"]:
        if term in padded:
            counts["gender_terms"] += 1

    weights = {
        "prestige_institutions": 2.0,
        "prestige_companies": 1.5,
        "prestige_locations": 1.0,
        "gender_terms": 1.0,
    }

    bias_score = sum(counts[k] * weights[k] for k in counts)
    counts["bias_score"] = bias_score
    return counts
