import torch
from torch.utils.data import random_split, DataLoader
import pandas as pd
import numpy as np
from constants import BERT_MODEL, BATCH_SIZE
from corpus_utils import get_everything
from dialog_dataloader import DialogDataset


def setup(verbose=False):
    # Check if CUDA is available. CUDA will not be availalble on Mac.
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        # Get the number of available CUDA devices
        num_devices = torch.cuda.device_count()

        # Get the name of the first CUDA device
        device_name = torch.cuda.get_device_name(0)

        if verbose:
            print(f"Number of CUDA devices: {num_devices}")
            print(f"Device name: {device_name}")

        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


def load_split_data(bert_version: BERT_MODEL, verbose=False):

    # Determine which version of BERT we want to collect the embeddings from
    if bert_version is BERT_MODEL.BERT:
        embedding_path = bert_version.value
    elif bert_version is BERT_MODEL.S_BERT:
        embedding_path = bert_version.value

    # Get data for X_FEATURES
    # Load vector embeddings from embeddings.npy file
    with open(embedding_path, "rb") as infile:
        embeddings = np.load(infile)

    # Get data for Y_LABELS
    full_corpus_df = get_everything()
    gt_labels_list = full_corpus_df["GT"].astype(int).tolist()

    if verbose:
        print(f"X feature shape: {embeddings.shape}")
        print(f"Y label shape: {len(gt_labels_list)}")

    dataset = DialogDataset(embeddings, gt_labels_list)

    if verbose:
        print(f"Total len of dataset: {len(dataset)}")
        item_1_label = dataset[0]["label"]
        item_1_features = dataset[0]["features"]

        print(f"First item's label: {item_1_label}")
        print(f"First item's initial features: {item_1_features[:5]} ...")

    # Split data into train, test, and validation sets (70%, 15%, 15%)
    num_items = len(dataset)
    num_train_data = round(num_items * 0.7)
    num_val_data = round(num_items * 0.15)
    num_test_data = num_items - num_train_data - num_val_data

    train_dataset, test_dataset, val_dataset = random_split(
        dataset, [num_train_data, num_test_data, num_val_data]
    )

    # Create the dataloaders for each split
    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    validation_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# Create the model

# Train for n epochs on the train and test data

# Check final validation accuracy on validation data

if __name__ == "__main__":

    device = setup()

    version = BERT_MODEL.BERT
    load_split_data(version)
