import numpy as np
import pandas as pd
import torch
import os

from numpy import ndarray

import constants as c
import matplotlib
from matplotlib import pyplot as plt
# matplotlib.use("Qt5Agg") # uncomment to pop charts out into seperate window on mac
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from corpus_utils import get_everything

# ---------------------------------------------------------------------
# Model and Tokenizer Setup
# ---------------------------------------------------------------------
# Load pre - trained tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# Load pre - trained model
model = BertModel.from_pretrained("bert-base-uncased")
model.eval() # since wea re not training BERT to be used later, we want to turn off dropout by turning on eval mode

# Function to get embeddings as seen in 10_Embeddings.ipynb
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use [CLS] token embedding as sentence representation
    return outputs.last_hidden_state[0][0].numpy()  # [CLS] token


# perform and return bert embeddings
# performed on whole episodes worth of dialogs at once
def perform_bert_embedding(dialogs: list[str]) -> list[np.ndarray]:
    embeddings = []

    for dialog_index in tqdm(
        range(0, len(dialogs)),
        desc="Performing BERT Embedding",
        unit=" dialogs",
        colour="blue",
    ):
        bert_embedding = get_embedding(dialogs[dialog_index])
        embeddings.append(bert_embedding)

    return embeddings


# perform PCA and return principle vectors on BERT vectors
def perform_pca(bert_vectors: list[np.ndarray], num_components=2) -> np.ndarray:
    arr_bert_vectors = np.vstack(bert_vectors)
    pca = PCA(n_components=num_components)
    bert_pca = pca.fit_transform(arr_bert_vectors)
    return bert_pca


# helper function to save plots to folder
def save_figure(name: str):
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{name}.png", dpi=300, bbox_inches="tight")


# plot 2d representation of bert embeddings
def plot_2d_pca(bert_pca_primary, ground_truth_list: list[int]):

    humorous_pca_x = []
    humorous_pca_y = []

    non_humorous_pca_x = []
    non_humorous_pca_y = []

    # split the PCA into humorous and non-humorous sections (for matplotlib labels)
    for i in range(0, len(ground_truth_list)):
        if ground_truth_list[i] == c.GT_HUMOUR:
            humorous_pca_x.append(bert_pca_primary[i, 0])
            humorous_pca_y.append(bert_pca_primary[i, 1])
        elif ground_truth_list[i] == c.GT_NON_HUMOUR:
            non_humorous_pca_x.append(bert_pca_primary[i, 0])
            non_humorous_pca_y.append(bert_pca_primary[i, 1])

    plt.figure(figsize=(8, 6))
    plt.scatter(humorous_pca_x, humorous_pca_y, s=5, c="b", label="Humorous Dialogs")
    plt.scatter(
        non_humorous_pca_x,
        non_humorous_pca_y,
        s=5,
        c="r",
        label="Non-Humorous Dialogs",
    )
    plt.title(
        "Big Bang Theory Humorous and Non-Humorous Dialogs\n Principle Component Analysis",
        fontsize=14,
    )
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.legend()
    save_figure("pca2d")
    plt.show()


# plot 2d representation of bert embeddings
def plot_3d_pca(bert_pca_primary, ground_truth_list: list[int]):

    humorous_pca_x = []
    humorous_pca_y = []
    humorous_pca_z = []

    non_humorous_pca_x = []
    non_humorous_pca_y = []
    non_humorous_pca_z = []

    # split the PCA into humorous and non-humorous sections (for matplotlib labels)
    for i in range(0, len(ground_truth_list)):
        if ground_truth_list[i] == c.GT_HUMOUR:
            humorous_pca_x.append(bert_pca_primary[i, 0])
            humorous_pca_y.append(bert_pca_primary[i, 1])
            humorous_pca_z.append(bert_pca_primary[i, 2])

        elif ground_truth_list[i] == c.GT_NON_HUMOUR:
            non_humorous_pca_x.append(bert_pca_primary[i, 0])
            non_humorous_pca_y.append(bert_pca_primary[i, 1])
            non_humorous_pca_z.append(bert_pca_primary[i, 2])

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(humorous_pca_x, humorous_pca_y, humorous_pca_z, s=5, linewidths=0, edgecolors='none', c="b",
               label="Humorous Dialogs")
    ax.scatter(
        non_humorous_pca_x,
        non_humorous_pca_y,
        non_humorous_pca_z,
        s=3, linewidths=0, edgecolors='none',
        c="r",
        label="Non-Humorous Dialogs"
    )
    ax.set_title(
        "Big Bang Theory Humorous and Non-Humorous Dialogs\n Principle Component Analysis",
        fontsize=14,
    )
    ax.set_xlabel("Principal Component 1", fontsize=14)
    ax.set_ylabel("Principal Component 2", fontsize=14)
    ax.set_zlabel("Principal Component 3", fontsize=14)
    ax.legend()
    save_figure("pca3d")
    plt.show()


def perform_and_plot_pca(full_corpus_df: pd.DataFrame,
                         embeddings: list[ndarray],
                         show_and_save_plots: bool = True):
    # PCA + Graph embeddgiuns (2d and 3d)
    pca = perform_pca(embeddings)
    pca_3d = perform_pca(embeddings, num_components=3)

    print(f"Len pca embeddings: {len(pca)}")
    if show_and_save_plots:
        gt_labels_list = full_corpus_df["GT"].astype(int).tolist()
        plot_2d_pca(pca, gt_labels_list)
        plot_3d_pca(pca_3d,gt_labels_list)


# Perform PCA analysis and optionally save plots of PCA to /plots directory
def perform_fresh_embeddings_and_pca(show_and_save_plots: bool = True):
    print("Starting")

    # Get all dialogs
    full_corpus_df = get_everything()
    print(f"Len of dataframe: {len(full_corpus_df)}")

    # Iterate through and get all embeddings
    dialog_list = full_corpus_df["Full_Conversation"].values.tolist()
    embeddings = perform_bert_embedding(dialog_list)
    print(f"Len of embeddings: {len(embeddings)}")

    # Save to a file
    np.save(c.SAVED_EMBEDDINGS_DIR, embeddings)

    # PCA + Graph embeddgiuns (2d and 3d)
    perform_and_plot_pca(full_corpus_df, embeddings, show_and_save_plots=show_and_save_plots)


# Perform and Plot PCA from already saved embeddings
def perform_cached_embeddings_and_pca(show_and_save_plots: bool = True):
    # Code for PCA
    with open(c.SAVED_EMBEDDINGS_DIR, "rb") as infile:
        embeddings = np.load(infile)
        full_corpus_df = get_everything()
        perform_and_plot_pca(full_corpus_df, embeddings, show_and_save_plots=show_and_save_plots)



# Perform BERT embeddings and PCA, and optionally plot PCA
# embeddings can be used from cache (use_cached_embeddings=true) or re-run
def orchestrate_pca(use_cached_embeddings: bool = True, show_and_save_plots: bool = True):
    cached_embeddings_present = os.stat(c.SAVED_EMBEDDINGS_DIR).st_size != 0
    # If embeddings are not present or we want to override cache
    if (not cached_embeddings_present) or (not use_cached_embeddings):
        perform_fresh_embeddings_and_pca(show_and_save_plots = show_and_save_plots)
    else:
        perform_cached_embeddings_and_pca(show_and_save_plots = show_and_save_plots)


if __name__ == "__main__":
    orchestrate_pca(use_cached_embeddings=True, show_and_save_plots=True)