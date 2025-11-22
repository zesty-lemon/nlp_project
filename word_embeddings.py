import torch
from transformers import BertModel, BertTokenizer

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
