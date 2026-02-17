import torch
import torch.nn as nn

# --- Configuration ---
BATCH_SIZE = 1      # 1 Sentence
SEQ_LEN = 5         # 5 Words ("The cat sat on mat")
INPUT_DIM = 10      # Word Vector Size (e.g. tiny Word2Vec)
HIDDEN_DIM = 20     # Memory Size

print(f"--- LSTM Configuration ---")
print(f"Input: (Batch={BATCH_SIZE}, Seq={SEQ_LEN}, Dim={INPUT_DIM})")

# --- 1. Define The Models ---
# A. Standard RNN
simple_rnn = nn.RNN(input_size=INPUT_DIM, hidden_size=HIDDEN_DIM, batch_first=True)

# B. LSTM (The one that remembers)
lstm = nn.LSTM(input_size=INPUT_DIM, hidden_size=HIDDEN_DIM, batch_first=True)

# C. GRU (The Efficient One: Returns output, hidden)
gru = nn.GRU(input_size=INPUT_DIM, hidden_size=HIDDEN_DIM, batch_first=True)

# --- 2. Create Fake Data ---
inputs = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)

# --- 3. The Forward Pass ---
print("\n--- 1. Simple RNN Output ---")
rnn_out, rnn_h = simple_rnn(inputs)
print(f"RNN Hidden Shape: {rnn_h.shape}")

print("\n--- 2. LSTM Output ---")
lstm_out, (lstm_h, lstm_c) = lstm(inputs)

print(f"LSTM Hidden State (Short-Term): {lstm_h.shape}")
print(f"LSTM Cell State   (Long-Term):  {lstm_c.shape}")

# Logic Check
print("\n--- Logic Check ---")
print("Does the Output match the Hidden State?")
# In LSTM, the 'output' at the last step should equal the 'hidden' state (h), NOT the cell state (c).
last_step_out = lstm_out[0, -1, :]
final_short_term = lstm_h[0, 0, :]

if torch.allclose(last_step_out, final_short_term):
    print("✅ YES: The LSTM output is just the Short-Term Memory exposed.")
else:
    print("❌ NO: Something is wrong.")

# --- 4. The GRU Pass ---
print("\n--- 3. GRU Mechanics ---")
# Returns: Output (History), Hidden State
gru_out, gru_h = gru(inputs)

print(f"GRU Output (Timeline):    {gru_out.shape}   -> (Batch, Seq, Hidden)")
print(f"GRU Hidden (Summary):     {gru_h.shape}     -> (Layers, Batch, Hidden)")

# Logic Check for GRU
last_step_gru = gru_out[0, -1, :]    # Batch 0, Last Time Step
final_gru_h   = gru_h[0, 0, :]       # Layer 0, Batch 0

if torch.allclose(last_step_gru, final_gru_h):
    print("✅ GRU Check Passed: Last time step equals final hidden state.")
else:
    print("❌ GRU Logic Failed.")

print("\n--- Summary ---")
print("LSTM: Keeps a separate 'Cell State' (Long-Term Memory).")
print("GRU:  Merges everything into one 'Hidden State'. Simpler, faster, often just as good.")
