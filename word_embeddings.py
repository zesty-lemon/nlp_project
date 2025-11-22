import numpy as np
import torch
import os
import constants as c
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

    for poem in tqdm(
        range(0, len(dialogs)),
        desc="Performing BERT Embedding",
        unit="poem",
        colour="blue",
    ):
        bert_embedding = get_embedding(dialogs[poem])
        embeddings.append(bert_embedding)

    return embeddings


# perform PCA and return principle vectors on BERT vectors
def perform_pca(bert_vectors: list[np.ndarray], num_components=2) -> np.ndarray:
    arr_bert_vectors = np.vstack(bert_vectors)
    pca = PCA(n_components=num_components)
    bert_pca = pca.fit_transform(arr_bert_vectors)
    return bert_pca


def driver():
    print("Starting")

    # Get all dialogs
    full_corpus_df = get_everything()
    print(f"Len of dataframe: {len(full_corpus_df)}")

    # Iterate through and get all embeddings
    dialog_list = full_corpus_df["Full_Conversation"].values.tolist()
    embeddings = perform_bert_embedding(dialog_list)
    print(f"Len of embeddings: {len(embeddings)}")

    # Save

    # PCA + Graph embeddgiuns

    #


if __name__ == "__main__":

    # Check if embeddings have already been created and then graph
    if os.path.exists(c.SAVED_EMBEDDINGS_DIR):
        driver()

    else:
        # Code for PCA. TODO: Need to fix this to reflect actual file type.
        with open(c.SAVED_EMBEDDINGS_DIR, "r") as infile:
            bert_vectors = infile.read()
            primary_components = perform_pca(bert_vectors, c.NUM_PRIMARY_COMPONENTS)
