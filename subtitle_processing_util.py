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

    cleaned_episode = []
    for i in range(0, len(episode_lines)):
        if not episode_lines[i].endswith((".", "!", "?")):
            if i+1 <= len(episode_lines):
                cleaned_episode.append(episode_lines[i]+episode_lines[i+1])
            else:
                cleaned_episode.append(episode_lines[i])
        else:
            pass

    processed_episode = "\n".join(cleaned_episode)

    with open("data/subtitles/young_sheldon_s4_e16_cleaned.txt", "w", encoding="utf-8") as file:
        file.write(episode)

    print("Done Processing")

