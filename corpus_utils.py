'''
Utility for reusable operations on corpus
'''

import json
import os
import constants as c
import pandas as pd


# open json file at filepath as json object
def open_file_as_json(file_path: str):
    with open(file_path, 'r') as f:
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
    files.sort() # because file names are consistantly "The Big bang_S0X0X" it is safe to do this
    return files


# join all turns of dialog into one paragraph of unified text
def get_and_combine_all_dialog_turns(dialog_id: str, dialog) -> str:
    # a turn of dialog is 1 person speaking.  I.E "it's Giles turn to speak"
    dialog_turns = []  # all turns of dialog (2-7 turns, depending on lookahead)
    dialog_keys = []
    # get the keys corresponding to dialog turns only
    for k, v in dialog.items():
        if k.startswith("Dialog Turns"):
            dialog_keys.append(k)
    dialog_keys.sort()
    # for every turn of dialog (in order, since they are sorted), append the text corresponding to the dialog
    for k in dialog_keys:
        dialog_turns.append(dialog[k]["Dialog"])

    full_dialog = " ".join(dialog_turns)
    return full_dialog

def get_all_dialogs_from_episode_df(episode_dialogs):
    dialogs = []
    # loop through each dialog (top level of JSON)
    for dialog_id, dialog in episode_dialogs.items():

        full_dialog = get_and_combine_all_dialog_turns(dialog_id, dialog)

        dialogs.append(
            {
                "Dialog_ID": dialog_id, # numerical identifier.  Sequential, counts up as episode progresses
                "Full_Dialog": full_dialog, # combined dialog from all turns for given scene
                "Scene": dialog.get("Scene"), # setting for the secene ("the stairwell"
                "Participants": dialog.get("Participant"), # who is involved in dialog
                "AV_ID": dialog.get("AV_ID"), # the episode & season number
            }
        )

    df_dialogs = pd.DataFrame(dialogs)
    return df_dialogs

all_episode_filepaths = get_episodes_for_dt_and_season(dialog_turns=5, season_num=1)

for episode_filepath in all_episode_filepaths:
    # open filepath to episode as Json object (dictionary)
    episode_dialogs = open_file_as_json(episode_filepath)
    # pull all dialogs out from the episode, stored as a dataframe. Dialog is all turns added together
    df_dialogs = get_all_dialogs_from_episode_df(episode_dialogs)
    print(df_dialogs.head())

print("done")