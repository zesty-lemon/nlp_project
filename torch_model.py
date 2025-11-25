import torch
import constants as c
import numpy as np


def setup():
    # Check if CUDA is available. CUDA will not be availalble on Mac.
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        # Get the number of available CUDA devices
        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices: {num_devices}")

        # Get the name of the first CUDA device
        device_name = torch.cuda.get_device_name(0)
        print(f"Device name: {device_name}")

        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


def load_data(bert_version="base"):

    # Determine which version of BERT we want to collect the embeddings from
    if bert_version == "base":
        embedding_path = c.BASE_BERT_EMBEDDINGS
    else:
        embedding_path = c.SENTENCE_BERT_EMBEDDINGS

    # Load vector embeddings from embeddings.npy file
    with open(embedding_path, "rb") as infile:
        embeddings = np.load(infile)

    return embeddings


# Split the embeddings into train test split validation sets
print


# Create the model

# Train for n epochs on the train and test data

# Check final validation accuracy on validation data

if __name__ == "__main__":

    device = setup()
    embeddings = load_data()
