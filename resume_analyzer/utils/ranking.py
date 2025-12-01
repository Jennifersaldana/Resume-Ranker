import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(resume_embeddings, jd_embeddings):
    """
    Compute cosine similarity between each resume and job description.
    """
    similarities = []
    for i in range(len(resume_embeddings)):
        sim = cosine_similarity(
            resume_embeddings[i:i+1],
            jd_embeddings
        )[0][0]
        similarities.append(sim)
    return similarities


def rank_resumes(resume_texts, jd_text, model):
    """
    Full ranking pipeline:
        - Embed resumes + JD
        - Compute similarities
        - Sort by similarity
    Returns:
        list of dict { 'resume', 'score', 'rank' }
    """

    # Embeddings
    resume_emb = model.encode(resume_texts, batch_size=32)
    jd_emb = model.encode([jd_text])[0].reshape(1, -1)

    similarities = compute_similarity(resume_emb, jd_emb)

    # Prepare ranked output
    results = []
    for i, score in enumerate(similarities):
        results.append({
            "resume_index": i,
            "similarity_score": float(score)
        })

    # Sort highest → lowest
    results_sorted = sorted(
        results,
        key=lambda x: x["similarity_score"],
        reverse=True
    )

    # Assign ranks
    for r, item in enumerate(results_sorted, start=1):
        item["rank"] = r

    return results_sorted
