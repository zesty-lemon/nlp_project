"""
Utility for reusable operations on corpus
"""

import json
import os
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

import constants as c
import pandas as pd

from word_embeddings import perform_classic_bert_embedding, perform_sentence_bert_embedding


# open json file at filepath as json object
def open_file_as_json(file_path: str):
    with open(file_path, "r") as f:
        data = json.load(f)
        return data


# get full filepaths for all episode data for a given number of turns and season
# returns full filepaths for all episode in a given season as a list
def get_episodes_for_dt_and_season(dialog_turns: int, season_num: int) -> list[str]:
    file_path = f"{c.BIG_BANG_THEORY_DIR}/DT_{dialog_turns}/Raw/S{season_num}"
    episode_filepaths = get_episode_list_from_directory(file_path)
    return episode_filepaths


def get_episode_list_from_directory(path: str) -> list[str]:
    entries = os.listdir(path)
    files = []
    for entry in entries:
        full_path = os.path.join(path, entry)
        if os.path.isfile(full_path):
            files.append(full_path)
    files.sort()  # because file names are consistently "The Big bang_S0X0X" it is safe to do this
    return files


# join all turns of dialog into one string of unified text
def get_and_combine_all_dialog_turns(dialog_id: str, dialog) -> str:
    # a turn of dialog is 1 person speaking.  I.E "it's Giles turn to speak"
    conversation = ""  # all turns of dialog (2-7 turns, depending on lookahead)
    dialog_keys = []
    # get the keys corresponding to dialog turns only
    for k, v in dialog.items():
        if k.startswith("Dialog Turns"):
            dialog_keys.append(k)
    dialog_keys.sort()
    # for every turn of dialog (in order, since they are sorted), concat the line and add a separator

    for k in dialog_keys:
        line = dialog[k]["Dialog"]
        # May need to add an optional space after the seperator.
        # Ask cooper about this if confused.
        conversation += f"{line}<s>"

    return conversation


# get all dialogs from episode stored as a dataframe
# ground truth label is preserved, all turns of dialogs are appended together
def get_all_dialogs_from_episode_df(episode_dialogs):
    dialogs = []
    # loop through each dialog (top level of JSON)
    for dialog_id, dialog in episode_dialogs.items():

        # full_conversation = create_convo_from_dialog(dialog_id, dialog)
        full_conversation = get_and_combine_all_dialog_turns(dialog_id, dialog)
        # pull specific top level metadata fields out of each dialog object
        dialogs.append(
            {
                "GT": int(dialog.get("GT")),  # the ground truth
                "Full_Conversation": full_conversation,  # combined dialog from all turns for given scene
            }
        )

    df_dialogs = pd.DataFrame(dialogs)
    return df_dialogs


# Get All Episode Filepaths for Entire Corpus
# in episode order
def get_all_episode_filepaths() -> np.ndarray:
    all_episode_filepaths = []

    for season in range(1, c.NUM_SEASONS + 1):
        season_filepaths = get_episodes_for_dt_and_season(
            dialog_turns=c.NUM_DIALOG_TURNS,
            season_num=season
        )

        all_episode_filepaths.extend(season_filepaths)

    all_episode_filepaths = np.array(sorted(all_episode_filepaths))

    return all_episode_filepaths


# Get labels and embeddings for each episode in a list of episodes
def get_labels_embeddings_for_episode_list(filepaths: list[str],
                                           model_selection: c.BERT_MODEL) -> Tuple[np.ndarray, np.ndarray]:
    X_embeddings = []
    y_labels = []

    for filepath in filepaths:
        # Open epsisode JSON file as DF
        episode_dialogs = open_file_as_json(filepath)
        # Pull all dialogs out from the episode, stored as a dataframe. Dialog is all turns added together
        df_dialogs = get_all_dialogs_from_episode_df(episode_dialogs)
        # ----- Get embeddings for dialogs -----
        dialog_list = df_dialogs["Full_Conversation"].values.tolist()
        gt_labels = df_dialogs["GT"].values.tolist()
        # Pick appropriate model and perform embeddings
        embeddings = []
        if model_selection is c.BERT_MODEL.BERT:
            embeddings = perform_classic_bert_embedding(dialog_list)
        elif model_selection is c.BERT_MODEL.S_BERT:
            embeddings = perform_sentence_bert_embedding(dialog_list)

        # Flatten so we get a list of all embeddings and labels for training set
        X_embeddings.extend(embeddings)
        y_labels.extend(gt_labels)

    # Convert to correct format
    X = np.vstack(X_embeddings)
    y = np.array(y_labels, dtype=int)
    return X,y

def corpus_test_train_split_by_episode(test_size = 0.3,
                                       random_state = 39,
                                       model_selection = c.BERT_MODEL.S_BERT) -> Tuple[np.ndarray,np.ndarray, np.ndarray, np.ndarray]:
    # Get all episode filepaths for corpus as a list
    all_episodes_in_corpus_filepaths = get_all_episode_filepaths()

    # Split the filepath list into test/training set
    Xtrain_filepaths, Xtest_filepaths = train_test_split(all_episodes_in_corpus_filepaths,
                                                         test_size=test_size,
                                                         random_state=random_state)

    # Get embeddings for Test and Training split
    X_train_embeddings, y_train_labels = get_labels_embeddings_for_episode_list(Xtrain_filepaths,
                                                                                model_selection=model_selection)

    X_test_embeddings, y_test_labels = get_labels_embeddings_for_episode_list(Xtest_filepaths,
                                                                              model_selection=model_selection)

    return X_train_embeddings,y_train_labels,X_test_embeddings,y_test_labels

def main():
    # Example Usage with S-BERT:
    Xtrain, Ytrain, Xtest, Ytest = corpus_test_train_split_by_episode(test_size = 0.3,
                                                                      model_selection=c.BERT_MODEL.S_BERT)

    # Example Usage with BERT:
    Xtrain, Ytrain, Xtest, Ytest = corpus_test_train_split_by_episode(test_size = 0.3,
                                                                      model_selection=c.BERT_MODEL.BERT)







if __name__ == "__main__":
    main()
