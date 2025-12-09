import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_score_histogram(df_results: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.hist(df_results["Predicted Score"], bins=10, edgecolor="black")
    ax.set_xlabel("Predicted Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Predicted Match Scores")
    fig.tight_layout()
    return fig

def plot_wordcount_vs_score(df_results: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.scatter(df_results["Word Count"], df_results["Predicted Score"])
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Predicted Score")
    ax.set_title("Word Count vs Match Score")
    fig.tight_layout()
    return fig

def plot_similarity_heatmap(names, sim_matrix: np.ndarray):
    fig, ax = plt.subplots()
    im = ax.imshow(sim_matrix, cmap="Blues")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Cosine Similarity Between Resumes")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
