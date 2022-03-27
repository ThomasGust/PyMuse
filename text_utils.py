import numpy as np

def get_vocab(joined_string):
    return sorted(set(joined_string))


def vectorize_vocab(vocab):
    return {u: i for i, u in enumerate(vocab)}, np.array(vocab)


def vectorize_string(char2idx, string):
    vectorized_output = np.array([char2idx[char] for char in string])
    return vectorized_output