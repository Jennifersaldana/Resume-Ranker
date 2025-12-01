import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load Embedding Model
# -----------------------------
def load_embedding_model(model_path="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Loads MiniLM model (local or huggingface).
    """
    return SentenceTransformer(model_path)

# -----------------------------
# Embed Text Batch
# -----------------------------
def embed_text_list(text_list, model):
    """
    Returns a NumPy matrix (N x D) of embeddings for a list of strings.
    """
    if not isinstance(text_list, list):
        raise ValueError("embed_text_list expects a Python list of strings.")

    return model.encode(text_list, batch_size=32, show_progress_bar=False)
