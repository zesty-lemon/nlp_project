import torch
import os
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchsummary import summary
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

    return train_dl, test_dl, validation_dl


# Create the model
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(768, 512), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        self.flatten = nn.Flatten()
        self.layer3 = nn.Sequential(nn.Linear(128, 1))
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.layer3(x)
        x = self.softmax(x)

        # Return confidence
        return x


# Training function for the model
def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x_features, y_labels) in enumerate(dataloader):
        x_features = x_features.to(device)
        y_labels = y_labels.to(device)

        # Compute prediction error
        output = model(x_features)
        print(output)
        print(output.shape)
        pred = round(output)
        loss = loss_fn(pred, y_labels)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x_features)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Testing function
def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x_features, y_labels in dataloader:
            x_features, y_labels = x_features.to(device), y_labels.to(device)
            output = model(x_features)
            pred = round(output)
            test_loss += loss_fn(pred, y_labels).item()
            correct += (pred.argmax(1) == y_labels).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# Validation function
def evaluate(data_loader, model, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        # loop through validation data points and pass them into the model
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            prediction = round(output)
            correct_predictions += (prediction == labels).sum().item()
            total_predictions += labels.size(0)

    print(f"Validation Accuracy: {correct_predictions / total_predictions}")


if __name__ == "__main__":

    debug = False
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    device = setup(verbose=debug)

    version = BERT_MODEL.BERT
    train_dl, test_dl, validation_dl = load_split_data(version, verbose=debug)

    model = NN().to(device)
    # print(f"Model summary : \n{summary(model, (64, 768))}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Train for n epochs on the train and test data
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dl, model, loss_fn, optimizer, device)
        test(test_dl, model, loss_fn, device)
    print("Done!")

    # Check final validation accuracy on validation data
    evaluate(validation_dl, model, device)
