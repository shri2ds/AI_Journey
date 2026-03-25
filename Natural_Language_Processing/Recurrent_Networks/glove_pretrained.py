import os
import torch
import torch.nn as nn
import numpy as np

def load_glove_file(path):
    """
    Parses the actual 100MB+ GloVe file.
    Returns: Dict { "word": np.array([0.1, ...]) }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Could not find GloVe file at: {path}")

    embeddings_index = {}

    # encoding='utf-8' is mandatory for GloVe to avoid crashing on weird chars
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except ValueError:
                continue

    print(f"✅ Loaded {len(embeddings_index)} word vectors.")
    return embeddings_index


def create_embedding_matrix(word2idx, embeddings_index, embedding_dim):
    """
    Creates the weight matrix for nn.Embedding.
    Row 0: <PAD> (All Zeros)
    Row 1: <UNK> (Random Noise or Average)
    Row i: The GloVe vector for word i
    """
    vocab_size = len(word2idx)
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim))

    # Force <PAD> to be zero
    if "<PAD>" in word2idx:
        embedding_matrix[word2idx["<PAD>"]] = np.zeros(embedding_dim)

    found_count = 0
    for word, i in word2idx.items():
        # Check if the word exists in the downloaded GloVe dictionary
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            found_count += 1

    print(f"Stats: Found {found_count} / {vocab_size} words in GloVe.")
    return torch.tensor(embedding_matrix, dtype=torch.float32)


if __name__ == "__main__":

    glove_path = "glove.6B.50d.txt"  # make sure the glove.txt file exists in the current path
    EMBED_DIM = 50
    glove_index = load_glove_file(glove_path)

    #  Real Vocab
    my_vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "king": 2,
        "queen": 3,
        "apple": 4,
        "microsoft": 5,
        "supercalifragilistic": 6  # This should NOT be in GloVe -> Random Noise
    }

    #  Create the Weight Matrix
    weights = create_embedding_matrix(my_vocab, glove_index, EMBED_DIM)

    #  Initialize PyTorch Layer
    embedding_layer = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)

    print("\n--- Reality Check ---")

    # Test 1: "King" (Should match the file)
    idx_king = torch.tensor([2])
    vec_king = embedding_layer(idx_king)
    print(f"Vector 'king' (First 5 dims): {vec_king[0][:5]}")

    # Test 2: Unknown Word
    idx_unk = torch.tensor([6])
    vec_unk = embedding_layer(idx_unk)
    print(f"Vector 'supercalifragilistic' (First 5 dims - Random): {vec_unk[0][:5]}")
