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


# Switch the embeddings used (BERT or S-BERT by specifying different model path)
def predict_humour_of_text(model_path: str, input_text: str) -> bool:
    # Step 1: Get the Classifier
    classifier = pick_bert_model_classifier(model_path)

    # Step 2: Convert Input to list
    input_text_as_list = [input_text]

    # Step 3: Get embedding of input
    input_embedding_array = run_burt_model(model_path, input_text_as_list)

    # Step 4: Convert the BERT embedding output to the input expected by the classifier
    input_embedding = np.vstack(input_embedding_array).astype(np.float32)
    input_embedding = np.array(input_embedding)

    # Step 5: Classify Input
    is_input_funny = classifier.predict(input_embedding) == 1

    print(f"Is Input Funny? {is_input_funny}")
    proba = classifier.predict_proba(input_embedding)[0]
    print("      Classes:", classifier.classes_)
    print("Probabilities:  ", proba)

    return is_input_funny


if __name__ == "__main__":
    # ---------- Example Usage ----------
    # Take some input text, and classify it as humourous or non-humourous

    example_text = ("Look at me i'm sheldon i'm so nerdy and smart."
                    "That's so true! I also struggle around women in a mildly creepy way."
                    "should we ask a real live girl out using a Star Trek reference?"
                    "As Spock says - Live Long and Prosper."
                    "Can a date be part of the prospering?")

    # Example 1: Running with S BERT
    print("Running with S Bert")
    predict_humour_of_text(path_to_s_bert_model, example_text)

    # Example 2: Running with Bert
    print("Running with Normal Bert")
    predict_humour_of_text(path_to_bert_model, example_text)