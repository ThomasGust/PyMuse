import numpy as np
import os
import re

def get_vocab(joined_string):
    return sorted(set(joined_string))


def vectorize_vocab(vocab):
    return {u: i for i, u in enumerate(vocab)}, np.array(vocab)


def vectorize_string(char2idx, string):
    vectorized_output = np.array([char2idx[char] for char in string])
    return vectorized_output

def load_abc_data(name):
    with open(os.path.join("TrainingData", name), "r") as f:
        text = f.read()
    songs = extract_song_snippet(text)
    return songs

def extract_song_snippet(text):
    pattern = '(^|\n\n)(.*?)\n\n'
    search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
    songs = [song[1] for song in search_results]
    print("Found {} songs in text".format(len(songs)))
    return songs

def save_song_to_abc(song, filename="tmp"):
    save_name = "{}.abc".format(filename)
    with open(save_name, "w") as f:
        f.write(song)
    return filename

def clean_abc(in_path, out_path):
    file1 = open(in_path,
                'r')

    file2 = open(out_path,
                'w')
    for line in file1.readlines():
        if not (line.startswith('%')) and not (line.startswith("Z")) and not (line.startswith("N")):
            print(line)
            file2.write(line)
    file2.close()
    file1.close()