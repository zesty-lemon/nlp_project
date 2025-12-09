import re
from pathlib import Path


# Strip Timestamps
def strip_header_footer(text: str) -> str:
    srt_pattern = r"^\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$"
    text = re.sub(srt_pattern, "", text, flags=re.MULTILINE)

    srt_pattern = r"\n\n"
    text = re.sub(srt_pattern, "\n", text, flags=re.MULTILINE)
    return text


def greedy_parse(episode_lines: list[str]) -> list[str]:
    """Iterates through unconcated lines and greedily concats based on punctuation.
    Args:
        episode_lines (list[str]): The list of unconcatenated lines taken from a subtitle txt file
    Returns:
        list[str]: A finalized list where lines are concatenated together
    """

    temp_line = ""
    punctuation = {".", "!", "?"}
    parsed_lines = []

    # Loop through all lines of episode
    for line in episode_lines:

        # If line exists and has punctuation
        if line and (line.strip()[-1] in punctuation):

            # Add it to the list of dialogs
            temp_line += line
            parsed_lines.append(temp_line)

            # Reset temp line for next line of dialog
            temp_line = ""

        # Else if its not the end of a dialog line
        elif line:

            # We can add it to temp constructor with a space to just keep going
            temp_line += f"{line} "

    return parsed_lines


def competent_parse(episode_lines: list[str]) -> list[str]:

    temp_line = ""
    parsed_lines = []
    punctuation = {".", "!", "?"}

    # Loop through all lines of episode
    for index, line in enumerate(episode_lines):

        try:

            current_line = line.strip()
            next_line = episode_lines[index + 1]

            # If line exists, isn't followed immediately by another line, and has punctuation
            if line and not next_line and (line[-1] in punctuation):

                # Add it to the list of dialogs
                temp_line += line
                parsed_lines.append(temp_line)

                # Reset temp line for next line of dialog
                temp_line = ""

            # Else add it to temp if it exists
            elif line:

                temp_line += f"{line} "

        except IndexError:
            # Reached end of episode lines
            # Concat final line and write output.
            parsed_lines.append(current_line)
            pass

    return parsed_lines


def main():

    # file_name = "young_sheldon_s4_e16"
    # file_name = "honeymooners_s1_e37"
    file_name = "The_Big_Bang_Theory_S10_E13"
    dir_path = Path("data/subtitles/")
    file_path = dir_path / f"{file_name}.srt"

    with open(file_path, "r", encoding="utf-8") as subtitles:

        episode = subtitles.read()

        episode = strip_header_footer(episode)

        episode_lines = episode.splitlines()

        # Uncomment depending on what option we want =====================

        greedy_lines = greedy_parse(episode_lines)
        competent_lines = competent_parse(episode_lines)
        # dialog_lines = giles_parse(episode_lines)

        # ================================================================

        # Print first lines to compare
        print(f"1st: {greedy_lines[:3]}")
        print(f"2nd: {competent_lines[:3]}")

        processed_episode = "\n".join(competent_lines)

        output_path = dir_path / f"{file_name}_cleaned.txt"
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(processed_episode)

        print("Done Processing")


if __name__ == "__main__":

    main()
