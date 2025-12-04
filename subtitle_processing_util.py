import re

# Strip Timestamps
def strip_header_footer(text:str) -> str:
    srt_pattern = r'^\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$'
    text = re.sub(srt_pattern, '', text, flags=re.MULTILINE)

    srt_pattern = r'\n\n'
    text = re.sub(srt_pattern, '\n', text, flags=re.MULTILINE)
    return text

if __name__ == "__main__":
    file = open("data/subtitles/young_sheldon_s4_e16.txt", "r", encoding="utf-8")
    episode = file.read()

    episode = strip_header_footer(episode)

    episode_lines = episode.splitlines()

# if the next one is not an empty string, loop until len(episode_lines).  Loop till you find the next one
# then concatonate i (original) to index(after search) and increment i

    # # deos not wor, going 2 bed
    # cleaned_episode = []
    # for i in range(0, len(episode_lines)-1):
    #     if (episode_lines[i+1] != ""):
    #         if i+1 <= len(episode_lines): # just in case
    #             for j in range(i+1, len(episode_lines) - 1):
    #
    #         else:
    #             cleaned_episode.append(episode_lines[i])
    #
    #         else:
    #             cleaned_episode.append(episode_lines[i])
    #     else:
    #         pass

    # processed_episode = "\n".join(cleaned_episode)
    # print(processed_episode)
    with open("data/subtitles/young_sheldon_s4_e16_cleaned.txt", "w", encoding="utf-8") as file:
        file.write(episode)

    print("Done Processing")

