import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# Rank resumes based on JD + optional custom criteria
# ---------------------------------------------------
def rank_resumes(resume_texts, jd_text, model):
    """
    Computes cosine similarity between each resume and the job description.
    Returns sorted ranking dictionaries.
    """

    if not resume_texts:
        raise ValueError("resume_texts must be a non-empty list.")

    # Embed resumes
    resume_emb = model.encode(resume_texts, show_progress_bar=False)
    jd_emb = model.encode([jd_text], show_progress_bar=False)[0].reshape(1, -1)

    # Compute similarity for each resume
    similarities = []
    for i, vec in enumerate(resume_emb):
        score = float(cosine_similarity(vec.reshape(1, -1), jd_emb)[0][0])
        similarities.append({
            "resume_index": i,
            "similarity_score": score
        })

    # Sort highest → lowest
    sorted_ranked = sorted(similarities, key=lambda x: x["similarity_score"], reverse=True)

    # Add ranking numbers
    for rank, item in enumerate(sorted_ranked, start=1):
        item["rank"] = rank

    return sorted_ranked
