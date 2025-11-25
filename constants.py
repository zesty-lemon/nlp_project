"""
Reusable Constants File
"""
from enum import Enum
from pathlib import Path

# File Operations
BIG_BANG_THEORY_DIR = Path("data/big_bang_theory")
NUM_SEASONS = 5
BERT_EMBEDDINGS_DIR = Path("output/bert_embeddings.npy")
S_BERT_EMBEDDINGS_DIR = Path("output/s_bert_embeddings.npy")

# Runtime Parameters
NUM_DIALOG_TURNS = 5
NUM_PRIMARY_COMPONENTS = 2

# Can theoretically change this to be [SEP] or something different
SEPARATOR = "<s>"

# ground truth labels: Humour = 1, Non-Humour = 0
GT_HUMOUR = 1
GT_NON_HUMOUR = 0

# enum to store different BERT model choices
# value = output directory of saved embeddings
class BERT_MODEL(Enum):
    BERT = "output/bert_embeddings.npy"
    S_BERT = "output/s_bert_embeddings.npy"
