import corpus_utils as c_u
import constants as c
import pandas as pd
import numpy as np

# File to preprocess our data
DIALOG_TURNS = 5

def main():
    all_dialogs = []
    # Iterate through all 5 seasons
    for season in range(1, 6):

        all_episode_filepaths = c_u.get_episodes_for_dt_and_season(dialog_turns=DIALOG_TURNS, season_num=season)

        for episode_filepath in all_episode_filepaths:
            # open filepath to episode as Json object (dictionary)
            episode_dialogs = c_u.open_file_as_json(episode_filepath)
            # pull all dialogs out from the episode, stored as a dataframe. Dialog is all turns added together
            df_dialogs = c_u.get_all_dialogs_from_episode_df(episode_dialogs)
            all_dialogs.append(df_dialogs)
            # sanity check to see correct number of dialogs per episode
            print("Dialogs per episode:")
            print(f"{episode_filepath}: {len(df_dialogs)} dialogs")

    df_all_dialogs = (pd.concat(all_dialogs, ignore_index=True))
    print(df_all_dialogs.head())


if __name__ == "__main__":
    print('Starting:')
    main()