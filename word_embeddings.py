import numpy as np
import torch
import os
import constants as c
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from corpus_utils import get_everything

# Load pre - trained tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load pre - trained model
model = BertModel.from_pretrained("bert-base-uncased")


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
    plt.scatter(humorous_pca_x, humorous_pca_y, s=40, c="b", label="Humorous Dialogs")
    plt.scatter(
        non_humorous_pca_x,
        non_humorous_pca_y,
        s=40,
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
    plt.show()


def driver():
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

    # PCA + Graph embeddgiuns
    pca = perform_pca(embeddings)
    print(f"Len pca embeddings: {len(pca)}")

    gt_labels_list = full_corpus_df["GT"].astype(int).tolist()
    plot_2d_pca(pca, gt_labels_list)


def run_already_saved():
    # Code for PCA
    with open(c.SAVED_EMBEDDINGS_DIR, "r") as infile:
        bert_vectors = np.load(infile)
        primary_components = perform_pca(bert_vectors, c.NUM_PRIMARY_COMPONENTS)
        full_corpus_df = get_everything()
        gt_labels_list = full_corpus_df["GT"].astype(int).tolist()
        plot_2d_pca(primary_components, gt_labels_list)


if __name__ == "__main__":

    # Check if the embeddings file has data saved to it
    if os.stat(c.SAVED_EMBEDDINGS_DIR).st_size == 0:
        driver()

    else:
        run_already_saved()
