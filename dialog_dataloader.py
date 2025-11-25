from torch.utils.data import Dataset


class DialogDataset(Dataset):
    """
    This is a custom dataloader for our dialog embeddings dataset

    Args:
        x_features: a numpy array of the dialog embedding features
        y_labels: an array of binary ground truth labels for humourous or non-humourous
    """

    def __init__(self, x_features, y_labels):
        self.x_features = x_features
        self.y_labels = y_labels

    def __len__(self):
        return len(self.x_features)

    def __getitem__(self):
        return NotImplementedError
