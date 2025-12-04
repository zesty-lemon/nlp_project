import corpus_utils as c_u
from pathlib import Path
import constants as c
import pandas as pd
import numpy as np
from use_trained_random_forest_model import predict_humour_of_text

# File to preprocess our data
DIALOG_TURNS = 5


def main():

    # Get cleaned dialogs

    # file_name = "young_sheldon_s4_e16"
    file_name = "honeymooners_s1_e37"
    dir_path = Path("data/subtitles/")
    file_path = dir_path / f"{file_name}.txt"

    with open(file_path, "r", encoding="utf-8") as script:

        lines = script.read()
        lines = lines.splitlines()

        dialog_list = []
        output_script = []
        for line in lines:

            dialog_list.append(line)
            output_script.append(line)

            # If we have a fully formed dialog
            if len(dialog_list) == 5:

                # Concat all lines with seperator
                dialog_str = " <s>".join(dialog_list)

                # Remove first and oldest entry
                dialog_list = dialog_list[1:]

                # Pass dialog_str to processing function
                humorous = predict_humour_of_text(dialog_str)

                if humorous == True:
                    output_script.append("\n[HILARIOUS AUDIENCE LAUGHTER]\n")

                elif humorous == False:
                    output_script.append("\n")


if __name__ == "__main__":
    print("Starting:")
    main()
