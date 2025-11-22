"""
Utility for reusable operations on corpus
"""

import json
import os
import constants as c
import pandas as pd
import csv


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


# Similar to get_and_combine_all_dialog_turns() but leaves them as a list representing a convo
def create_convo_from_dialog(dialog_id: str, dialog) -> str:
    # a turn of dialog is 1 person speaking.  I.E "it's Giles turn to speak"
    conversation = ""
    dialog_keys = []
    # get the keys corresponding to dialog turns only
    for k, v in dialog.items():
        if k.startswith("Dialog Turns"):
            dialog_keys.append(k)
    dialog_keys.sort()
    # for every turn of dialog (in order, since they are sorted), append the text corresponding to the dialog
    for k in dialog_keys:
        conversation.append(dialog[k]["Dialog"])

    return conversation


def get_all_dialogs_from_episode_df(episode_dialogs):
    dialogs = []
    # loop through each dialog (top level of JSON)
    for dialog_id, dialog in episode_dialogs.items():

        # full_conversation = create_convo_from_dialog(dialog_id, dialog)
        full_conversation = get_and_combine_all_dialog_turns(dialog_id, dialog)
        # pull specific top level metadata fields out of each dialog object
        dialogs.append(
            {
                "GT": dialog.get("GT"),  # the ground truth
                "Full_Conversation": full_conversation,  # combined dialog from all turns for given scene
            }
        )

    df_dialogs = pd.DataFrame(dialogs)
    return df_dialogs


def get_everything() -> pd.DataFrame:

    full_corpus_df = pd.DataFrame()

    for season in range(1, c.NUM_SEASONS + 1):

        all_episode_filepaths = get_episodes_for_dt_and_season(
            dialog_turns=c.NUM_DIALOG_TURNS, season_num=season
        )

        for episode_filepath in all_episode_filepaths:
            # open filepath to episode as Json object (dictionary)
            episode_dialogs = open_file_as_json(episode_filepath)
            # pull all dialogs out from the episode, stored as a dataframe. Dialog is all turns added together
            df_dialogs = get_all_dialogs_from_episode_df(episode_dialogs)

            # Concat dataframe to overall show dataframe

            full_corpus_df = pd.concat([full_corpus_df, df_dialogs], axis=0)

    return full_corpus_df


def main():
    all_episode_filepaths = get_episodes_for_dt_and_season(dialog_turns=5, season_num=1)

    for episode_filepath in all_episode_filepaths:
        # open filepath to episode as Json object (dictionary)
        episode_dialogs = open_file_as_json(episode_filepath)
        # pull all dialogs out from the episode, stored as a dataframe. Dialog is all turns added together
        df_dialogs = get_all_dialogs_from_episode_df(episode_dialogs)
        # print(df_dialogs.head())

    # Access first sentence of first conversation
    print(df_dialogs["Full_Conversation"][0])
    print("done")


if __name__ == "__main__":
    main()
