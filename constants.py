"""
Reusable Constants File
"""

from enum import Enum
from pathlib import Path

# File Operations
BIG_BANG_THEORY_DIR = Path("data/big_bang_theory")
RANDOM_FOREST_TRAINED_MODEL_DIR_PREFIX = "trained_models/random_forest/"

NUM_SEASONS = 5

# Runtime Parameters
NUM_DIALOG_TURNS = 5
NUM_PRIMARY_COMPONENTS = 2

# Can theoretically change this to be [SEP] or something different
SEPARATOR = "<s>"

# Ground truth labels: Humour = 1, Non-Humour = 0
GT_HUMOUR = 1
GT_NON_HUMOUR = 0
GT_HUMOR_CLASSNAME = "humour"
GT_NON_HUMOR_CLASSNAME = "non_humour"
CLASSES = {
    1: GT_HUMOR_CLASSNAME,
    0: GT_NON_HUMOR_CLASSNAME,
}  # source of truth for classname -> val mappings

# Pytorch Neural Network Constants
BATCH_SIZE = 64
RANDOM_SEED = 21


# enum to store different BERT model choices
# value = bert_embeddings directory of saved embeddings
class BERT_MODEL(Enum):
    BERT = "bert_embeddings/bert_embeddings.npy"
    S_BERT = "bert_embeddings/s_bert_embeddings.npy"
