import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel

# Load pre - trained tokenizer
tokenizer = BertTokenizer.from_pretrained("bert - base - uncased")

# Load pre - trained model
model = BertModel.from_pretrained("bert - base - uncased")


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

    for poem in tqdm(range(0, len(dialogs)), desc="Performing BERT Embedding", unit="poem", colour="blue"):
        bert_embedding = get_embedding(dialogs[poem])
        embeddings.append(bert_embedding)

    return embeddings


# perform PCA and return principle vectors on BERT vectors
def perform_pca(bert_vectors: list[np.ndarray], num_components = 2) -> np.ndarray:
    arr_bert_vectors = np.vstack(bert_vectors)
    pca = PCA(n_components=num_components)
    bert_pca = pca.fit_transform(arr_bert_vectors)
    return bert_pca