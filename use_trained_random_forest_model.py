import joblib
import corpus_utils

from sklearn.ensemble import RandomForestClassifier

path_to_model = "trained_models/random_forest/final/S_BERT_26_11_14_20_26/random_search_model/random_forest_model.joblib"

def get_trained_model(path_to_model: str) -> RandomForestClassifier:
    # model retrival inside method to force typing hints to work correctly
    trained_classifier = joblib.load(path_to_model)
    return trained_classifier

classifier = get_trained_model(path_to_model)

# todo: convert string input to list of just that one string
# use corpus utils to get embedding of that one string
# get output
# maybe just sbert - or do the ANNOYING thing of switching between them on demand
print("Loaded Model")

