import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re

# --- The Vocabulary Builder ---
class Vocabulary:
    def __init__(self, tokenizer_func=None):
        self.tokenizer = tokenizer_func if tokenizer_func else self.simple_tokenizer
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        # Add Special Tokens immediately
        self.add_word('<PAD>')  # Index 0
        self.add_word('<UNK>')  # Index 1

    def simple_tokenizer(self, text):
        # A quick regex to keep only alphanumeric chars and lowercase them
        return re.findall(r"\w+", text.lower())

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def build_vocab(self, sentence_list, min_freq=1):
        """
        Scans the entire dataset and builds the map.
        min_freq: Ignore words that appear less than X times (rare words).
        """
        counter = Counter()
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                counter[word] += 1

        for word, count in counter.items():
            if count >= min_freq:
                self.add_word(word)

    def encode(self, text):
        """ Converts 'I love AI' -> [23, 4, 99] """
        tokens = self.tokenizer(text)
        encoded = []
        for token in tokens:
            if token in self.word2idx:
                encoded.append(self.word2idx[token])
            else:
                encoded.append(self.word2idx['<UNK>'])

        return encoded


# --- The Dataset Class ---
class SentimentDataset(Dataset):
    def __init__(self, sentences, labels, vocab=None):
        self.sentences = sentences
        self.labels = labels

        # If no vocab provided, build one from these sentences
        if vocab is None:
            self.vocab = Vocabulary()
            self.vocab.build_vocab(sentences)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        Returns: (vectorized_sentence, label)
        """
        txt = self.sentences[idx]
        label = self.labels[idx]

        # Convert text to numbers
        numerical_sentence = self.vocab.encode(txt)

        return torch.tensor(numerical_sentence), torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    """
    This function runs AFTER __getitem__ but BEFORE the batch enters the model.
    It receives a list of tuples: [(tensor([1,2]), 0), (tensor([1,2,3,4]), 1)]
    It must pad them to the same length.
    """
    sequences, labels = zip(*batch)

    # Pad sequences to the length of the longest sentence in this batch
    # batch_first=True -> Output shape: (Batch, Max_Seq_Len)
    # padding_value=0 -> Fills with <PAD> token
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)

    # Stack labels
    labels = torch.stack(labels)

    return padded_sequences, labels

if __name__ == "__main__":
    # Mock Data
    raw_data = [
        "I love this movie",           # Len 4
        "This is the worst film ever", # Len 6
        "Great acting",                # Len 2
        "Terrible plot"                # Len 2
    ]
    raw_labels = [1, 0, 1, 0]  # 1=Pos, 0=Neg

    # Setup Dataset
    dataset = SentimentDataset(raw_data, raw_labels)

    print(f"Vocab Size: {len(dataset.vocab.word2idx)}")
    print(f"Word 'movie' is at index: {dataset.vocab.word2idx.get('movie')}")

    # Setup DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    print("\n--- Batching Simulation ---")
    for batch_idx, (inputs, targets) in enumerate(loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"Input Shape: {inputs.shape} (Batch, Max_Seq_Len)")
        print(f"Input Matrix:\n{inputs}")
        print(f"Labels: {targets}")
