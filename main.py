from pathlib import Path
import torch
from torch_model import NN, pred_dialog
import use_trained_random_forest_model as random_forest


def main():

    print("Starting:")

    model_name = "pytorch"

    if model_name == "pytorch":
        model_path = Path("trained_models/pytorch_model/trained_weights.pt")
        model = NN()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

    # file_name = "young_sheldon_s4_e16_cleaned"
    script_name = "honeymooners_s1_e37_cleaned"
    data_path = Path("data/subtitles/")
    script_path = data_path / f"{script_name}.txt"

    with open(script_path, "r", encoding="utf-8") as script:

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

                match model_name:
                    case "random_forest":
                        humorous = random_forest.predict_humour_of_text(
                            use_bert=True, input_text=dialog_str
                        )
                    case "pytorch":
                        humorous = pred_dialog(model, dialog_str)

                if humorous == True:
                    output_script.append("\n[HILARIOUS AUDIENCE LAUGHTER]\n")

                elif humorous == False:
                    output_script.append("\n")

    finalized_script = "\n".join(output_script)

    script_output_path = data_path / f"{script_name}_funny.txt"
    with open(script_output_path, "w", encoding="utf-8") as file:
        file.write(finalized_script)

    print("Done Writing File.")


if __name__ == "__main__":
    main()
