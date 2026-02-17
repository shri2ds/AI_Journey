import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        concat_size = input_size + hidden_size

        # 1. Forget Gate Weights
        self.Wf = np.random.rand(hidden_size, concat_size)
        self.bf = np.zeros((hidden_size,1))

        # 2. Input Gate Weights
        self.Wi = np.random.randn(hidden_size, concat_size)
        self.bi = np.zeros((hidden_size, 1))

        # 3. Candidate Memory Weights (Cell Gate)
        self.Wc = np.random.randn(hidden_size, concat_size)
        self.bc = np.zeros((hidden_size, 1))

        # 4. Output Gate Weights
        self.Wo = np.random.randn(hidden_size, concat_size)
        self.bo = np.zeros((hidden_size, 1))

    def forward_step(self, x, h_prev, c_prev):
        """
        Processes a SINGLE time step.
        x: Current Input Vector (input_size, 1)
        h_prev: Previous Hidden State (hidden_size, 1)
        c_prev: Previous Cell State (hidden_size, 1)
        """

        # Step 0: Concatenate h_prev and x
        # Shape: (hidden_size + input_size, 1)
        combined = np.vstack((h_prev, x))

        # Step 1: Forget Gate (f_t) - "What do I delete?"
        # Logic: f_t will be a vector of 0s and 1s (mostly)
        f_t = sigmoid(np.dot(self.Wf, combined) + self.bf)

        # Step 2: Input Gate (i_t) - "What do I pay attention to?"
        i_t = sigmoid(np.dot(self.Wi, combined) + self.bi)

        # Step 3: Candidate Memory (C_tilde) - "What is the NEW info?"
        # Note: Uses Tanh because values can be negative (e.g. "Sentiment went DOWN")
        c_tilde = tanh(np.dot(self.Wc, combined) + self.bc)

        # Step 4: Update Cell State (C_t) - THE SUPERHIGHWAY
        # Logic: (Old * Forget) + (New * Input)
        c_next = (f_t * c_prev) + (i_t * c_tilde)

        # Step 5: Output Gate (o_t) - "What do I reveal?"
        o_t = sigmoid(np.dot(self.Wo, combined) + self.bo)

        # Step 6: Update Hidden State (h_t) - The Working Memory
        h_next = o_t * tanh(c_next)

        print(f"Forget Gate: {f_t.flatten()}")

        return h_next, c_next

# --- Verification Simulation ---
input_dim = 4
hidden_dim = 3
lstm = LSTM(input_dim, hidden_dim)


x_input = np.random.randn(input_dim, 1)  # Word: "cat"
h_prev = np.zeros((hidden_dim, 1))  # Initial Hidden
c_prev = np.zeros((hidden_dim, 1))  # Initial Cell

print("--- Step 1 Input ---")
print(f"Word Vector: {x_input.flatten()}")

h_new, c_new = lstm.forward_step(x_input, h_prev, c_prev)

print("\n--- Step 1 Output ---")
print(f"New Hidden State (Short Term): {h_new.flatten()}")
print(f"New Cell State   (Long Term):  {c_new.flatten()}")

# Pass the NEW states back in
x_input_2 = np.random.randn(input_dim, 1)  # Word: "sat"
h_final, c_final = lstm.forward_step(x_input_2, h_new, c_new)

print("\n--- Step 2 Output ---")
print(f"Final Hidden State: {h_final.flatten()}")

