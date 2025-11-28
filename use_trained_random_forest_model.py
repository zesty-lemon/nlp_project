import joblib
import numpy as np
from numpy import ndarray

import word_embeddings

from sklearn.ensemble import RandomForestClassifier

# Paths to trained models (trained_models directory)
# if not present check filepath or re-run create_train_random_forest.py
path_to_s_bert_model = "trained_models/random_forest/final/S_BERT_26_11_19_03_59/random_search_model/random_forest_model.joblib"
path_to_bert_model = "trained_models/random_forest/final/BERT_26_11_15_21_35/random_search_model/random_forest_model.joblib"


# Unpack trained classifier from trained_models/random_forest directory
def get_trained_model(path_to_model: str) -> RandomForestClassifier:
    # model retrival inside method to force typing hints to work correctly
    trained_classifier = joblib.load(path_to_model)
    return trained_classifier

# Load classifiers into memory
print("BEGIN loading BERT and SBERT classifiers into memory")
bert_classifier = get_trained_model(path_to_bert_model)
s_bert_classifier = get_trained_model(path_to_s_bert_model)
print("END loading BERT and SBERT classifiers into memory")


# Load appropriate BERT model and get embeddings for dialogs as input
def run_burt_model(model_path: str, dialogs: list[str]) -> list[ndarray]:
    embedding = []
    if model_path is path_to_bert_model:
        embedding = word_embeddings.perform_classic_bert_embedding(dialogs)
    elif model_path is path_to_s_bert_model:
        embedding = word_embeddings.perform_sentence_bert_embedding(dialogs)
    return embedding


# Pick appropriate model to return based on filepath
def pick_bert_model_classifier(path_to_model: str):
    if path_to_model is path_to_bert_model:
        return bert_classifier
    elif path_to_model is path_to_s_bert_model:
        return s_bert_classifier


# Example of usage to take a string input, get its BERT (or SBERT) embedding, and classify it
def example_usage(model_path: str):
    # Step 1: Get the Classifier
    classifier = pick_bert_model_classifier(model_path)
    # Step 2: Example input
    example_text = ("Look at me i'm sheldon i'm so nerdy and smart."
                    "That's so true! I also struggle around women in a mildly creepy way"
                    "should we ask a real live girl out using a Star Trek reference?"
                    "As Spock says - Live Long and Prosper"
                    "Can a date be part of the prospering?")
    # Step 3: Convert Input to list
    example_text_as_list = [example_text]

    # Step 4: Get embedding of input
    example_embedding_array = run_burt_model(model_path, example_text_as_list)
    example_embedding = np.vstack(example_embedding_array).astype(np.float32)
    example_embedding = np.array(example_embedding)

    # Step 5: Classify Input
    is_example_funny = classifier.predict(example_embedding) == 1

    print(f"Is Example Funny? {is_example_funny}")
    # running with example_text and BERT: {ValueError}ValueError('X has 384 features, but RandomForestClassifier is expecting 768 features as input.')

if __name__ == "__main__":
    # Running with S BERT
    print("Running with S Bert")
    example_usage(path_to_s_bert_model)
    # RUnning with Bert
    print("Running with Normal Bert")
    example_usage(path_to_bert_model)

# TODO: this can be faster.  Right now we are re-loading the classifier from a file every time we want
# to do a classification.  Very slow.  We should do it once for each classifier (bert/sbert)
# and then hit the same classifier every time
# move to top of class maybe?  Or run outside of main?