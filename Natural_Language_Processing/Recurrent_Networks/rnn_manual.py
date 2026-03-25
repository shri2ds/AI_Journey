import numpy as np

# Activation Function (Tanh squashes values between -1 and 1)
def tanh(x):
    return np.tanh(x)

class simpleRNN:
    def __init__(self, input_size, hidden_size):
        # Weights for the Input (Current Word)
        self.Wx = np.random.rand(hidden_size, input_size)

        # Weights for the Hidden State (Past Memory)
        self.Wh = np.random.rand(hidden_size, hidden_size)

        # Bias
        self.bias = np.zeros((hidden_size, 1))

    def forward(self, inputs):
        """
            inputs: List of word vectors (e.g., shape [3, 1] each)
        """
        # Initialize Memory (Hidden State) as zeros
        h = np.zeros((self.Wh.shape[0], 1))

        # Store states to visualize later
        hidden_states = []

        print("--- Processing Sequence ---")
        for i, x in enumerate(inputs):
            # The Core RNN Equation:
            # New_Memory = tanh( (Old_Memory * Weight_Mem) + (Current_Input * Weight_Input) + Bias )
            h = tanh(np.dot(self.Wh, h) + np.dot(self.Wx, x) + self.bias)

            hidden_states.append(h)
            print(f"Step {i + 1}: Updated Memory. First 3 values: {h.flatten()[:3]}")

        return hidden_states

# And we want a memory of size 3
rnn = simpleRNN(input_size=4, hidden_size=3)

# Mock Embeddings: Simulating a batch of 3 words.
# Shape: (Sequence_Length, Embedding_Dim, 1) -> (3, 4, 1)
# In production, these would come from model.wv['word']
input_sentence = [
    np.random.randn(4, 1), # "The"
    np.random.randn(4, 1), # "cat"
    np.random.randn(4, 1)  # "sat"
]

final_memory = rnn.forward(input_sentence)

print("\nFinal Context Vector (Summary of 'The cat sat'):")
print(final_memory[-1])
