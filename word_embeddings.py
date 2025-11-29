import numpy as np
import pandas as pd
import torch
import os

from numpy import ndarray

import constants as c
from constants import BERT_MODEL

import matplotlib
from matplotlib import pyplot as plt
# matplotlib.use("Qt5Agg") # uncomment to pop charts out into seperate window on mac
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
from corpus_utils import get_everything

# BERT and SBERT Model Initilization
# if initalzied outside of method they only need to be loaded into memory once
# Load pretrained tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# Load pretrained model
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

# Load pre-trained model
s_bert_model = SentenceTransformer("all-MiniLM-L6-v2")
s_bert_model.eval()

# if file or directories do not exist make both directories and empty file
def make_empty_file_if_not_exists(path: str):
    # Make parent directories
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Create the file if missing
    if not os.path.isfile(path):
        with open(path, "w") as f:
            pass


# ---------------------------------------------------------------------
# Model and Tokenizer Setup
# ---------------------------------------------------------------------

# Function to get embeddings as seen in 10_Embeddings.ipynb
def get_classic_bert_embedding(text, tokenizer: BertTokenizer, model: BertModel):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use [CLS] token embedding as sentence representation
    return outputs.last_hidden_state[0][0].numpy().astype(np.float32)  # [CLS] token, force 32 bit


# perform and return bert embeddings
# performed on whole episodes worth of dialogs at once
def perform_classic_bert_embedding(dialogs: list[str]) -> list[np.ndarray]:
    embeddings = []
    # Run BERT separately for every dialog
    for dialog_index in tqdm(range(0, len(dialogs)),
                             desc="Performing BERT Embedding",
                             unit=" dialogs",
                             colour="blue"):

        bert_embedding = get_classic_bert_embedding(dialogs[dialog_index], bert_tokenizer, bert_model)
        embeddings.append(bert_embedding)

    return embeddings


# Return embeddings for SBert
def get_sbert_embeddings(texts: list[str], model: SentenceTransformer, print_stats: bool = False):
    embeddings = model.encode(texts, convert_to_numpy=True).astype(np.float32) # Should be default, but forcing anyway

    if print_stats:
        print(f"Embedding shape: {embeddings.shape}")

    return embeddings


# perform and return SBERT embeddings (batched)
def perform_sentence_bert_embedding(dialogs: list[str], batch_size: int = 32) -> list[np.ndarray]:
    embeddings = []

    for start_idx in tqdm(
        range(0, len(dialogs), batch_size),
        desc="Performing Sentence BERT Embedding",
        unit=" dialogs",
        colour="blue",
    ):
        # Batch embeddings for speed
        batch = dialogs[start_idx : start_idx + batch_size]
        # encode entire batch at once
        batch_embeddings = get_sbert_embeddings(batch, s_bert_model)

        for emb in batch_embeddings:
            embeddings.append(emb)

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
def plot_2d_pca(bert_pca_primary, ground_truth_list: list[int], figure_name: str):

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
    save_figure(figure_name)
    plt.show()


# plot 2d representation of bert embeddings
def plot_3d_pca(bert_pca_primary, ground_truth_list: list[int], figure_name: str):

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
    save_figure(figure_name)
    plt.show()


# Orchestrate performing PCA and Plotting 2d and 3d plots
# performed on new embeddings
def perform_and_plot_pca(full_corpus_df: pd.DataFrame,
                         embeddings: list[ndarray],
                         model_selection: BERT_MODEL,
                         show_and_save_plots: bool = True):
    # PCA + Graph embeddgiuns (2d and 3d)
    pca = perform_pca(embeddings)
    pca_3d = perform_pca(embeddings, num_components=3)

    print(f"Len pca embeddings: {len(pca)}")
    if show_and_save_plots:
        gt_labels_list = full_corpus_df["GT"].astype(int).tolist()
        plot_2d_pca(pca, gt_labels_list, model_selection.name + "_2d_pca")
        plot_3d_pca(pca_3d,gt_labels_list, model_selection.name + "_3d_pca")


# Perform PCA analysis and optionally save plots of PCA to /plots directory
def perform_fresh_embeddings_and_pca(model_selection: BERT_MODEL,
                                     show_and_save_plots: bool = True):
    print("Starting")

    # Get all dialogs
    full_corpus_df = get_everything()
    print(f"Len of dataframe: {len(full_corpus_df)}")

    # Iterate through and get all embeddings
    dialog_list = full_corpus_df["Full_Conversation"].values.tolist()

    # Pick appropriate model and perform embeddings
    if model_selection is BERT_MODEL.BERT:
        embeddings = perform_classic_bert_embedding(dialog_list)
        np.save(model_selection.value, embeddings)
        print(f"Embeddings Saved To: {model_selection.value}")
        print(f"Len of embeddings: {len(embeddings)}")

    elif model_selection is BERT_MODEL.S_BERT:
        embeddings = perform_sentence_bert_embedding(dialog_list)
        np.save(model_selection.value, embeddings)
        print(f"Embeddings Saved To: {model_selection.value}")
        print(f"Len of embeddings: {len(embeddings)}")

    # PCA + Graph embeddgiuns (2d and 3d)
    perform_and_plot_pca(full_corpus_df,
                         embeddings,
                         model_selection = model_selection,
                         show_and_save_plots = show_and_save_plots)


# Perform and Plot PCA from already saved embeddings
def perform_cached_embeddings_and_pca(model_selection: BERT_MODEL,
                                      show_and_save_plots: bool = True):
    # Code for PCA
    with open(model_selection.value, "rb") as infile:
        embeddings = np.load(infile)
        full_corpus_df = get_everything()

        perform_and_plot_pca(full_corpus_df,
                             embeddings,
                             model_selection = model_selection,
                             show_and_save_plots=show_and_save_plots)



# Perform BERT embeddings and PCA, and optionally plot PCA
# embeddings can be used from cache (use_cached_embeddings=true) or re-run
# models can be selected from by specifying BERT_MODEL.BERT or BERT_MODEL.S_BERT
def orchestrate_embeddings_and_pca(model_selection: BERT_MODEL,
                                   use_cached_embeddings: bool = True,
                                   show_and_save_plots: bool = True):
    # Make bert_embeddings file if not already present
    make_empty_file_if_not_exists(model_selection.value)
    # Check if bert_embeddings file is empty (if cached data not present)
    cached_embeddings_present = os.stat(model_selection.value).st_size != 0
    # If embeddings are not present or we want to override cache
    if (not cached_embeddings_present) or (not use_cached_embeddings):
        perform_fresh_embeddings_and_pca(model_selection = model_selection,
                                         show_and_save_plots = show_and_save_plots)
    else:
        perform_cached_embeddings_and_pca(model_selection = model_selection,
                                         show_and_save_plots = show_and_save_plots)



# Return Dataframe of Ground Truth labels and Embeddings
# Uses cached emebddings if available, can also force a cache refresh
def get_embeddings_and_labels_for_model(model_selection: BERT_MODEL,
                                        use_cached_embeddings: bool = True):
    # Check if Cached embeddings exist, and if not re-run them before returning them
    make_empty_file_if_not_exists(model_selection.value)
    cached_embeddings_present = os.stat(model_selection.value).st_size != 0
    if (not cached_embeddings_present) or (not use_cached_embeddings):
        perform_fresh_embeddings_and_pca(model_selection = model_selection,
                                         show_and_save_plots = False)

    with open(model_selection.value, "rb") as infile:
        embeddings = np.load(infile)

    # Convert embeddings to list[np.ndarray]
    embeddings_list = [np.asarray(e, dtype=np.float32) for e in embeddings] # Force Typing

    # Load ground-truth labels
    full_corpus_df = get_everything()
    gt_list = full_corpus_df["GT"].astype(int).tolist()

    # Build the DataFrame
    df = pd.DataFrame({
        "GT": gt_list,
        "Embedding": embeddings_list
    })

    return df


# Refresh BERT and S BERT embeddings
# WARNING - VERY SLOW
# ONLY DO IF NEEDED
def refresh_model_embeddings():
    get_embeddings_and_labels_for_model(model_selection=BERT_MODEL.S_BERT,
                                                        use_cached_embeddings=False)

    get_embeddings_and_labels_for_model(model_selection=BERT_MODEL.BERT,
                                                        use_cached_embeddings=False)


if __name__ == "__main__":
    refresh_model_embeddings()

    orchestrate_embeddings_and_pca(model_selection = BERT_MODEL.S_BERT,
                                   use_cached_embeddings=False,
                                   show_and_save_plots=True)

    orchestrate_embeddings_and_pca(model_selection = BERT_MODEL.BERT,
                                   use_cached_embeddings=False,
                                   show_and_save_plots=True)