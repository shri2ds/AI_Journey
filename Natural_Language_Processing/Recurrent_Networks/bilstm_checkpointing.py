import torch
import torch.nn as nn
import os

# --- The Upgraded Model ---
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # bidirectional=True
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        # The input to the Linear layer is now hidden_dim * 2
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)

        output, (hidden, cell) = self.lstm(embedded)

        # hidden shape for BiLSTM: [num_layers * 2, batch, hidden_dim]
        hidden_forward = hidden[-2]     # hidden[-2] is the last forward state
        hidden_backward = hidden[-1]    # hidden[-1] is the last backward state

        # Glue them together along the feature dimension (dim=1)
        final_hidden = torch.cat((hidden_forward, hidden_backward), dim=1)

        return self.fc(final_hidden)

# --- Checkpointing Logic ---
def save_model(model, filepath):
    print(f"--- Saving Model to {filepath} ---")
    torch.save(model.state_dict(), filepath)
    print("✅ Save Complete.")

def load_model(model_class_instance, filepath):
    print(f"--- Loading Model from {filepath} ---")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ No checkpoint found at {filepath}")

    model_class_instance.load_state_dict(torch.load(filepath, map_location=torch.device('mps')))

    # CRITICAL: Always set to evaluation mode after loading for inference
    # This disables Dropout and Batch Normalization layers
    model_class_instance.eval()
    print("✅ Load Complete.")
    return model_class_instance


# --- Execution Simulation ---
if __name__ == "__main__":
    VOCAB_SIZE = 100
    EMBED_DIM = 10
    HIDDEN_DIM = 16
    OUTPUT_DIM = 1

    # 1. Initialize fresh model
    model = BiLSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)

    # 2. Simulate training (Fake input data: Batch of 2, Length 5)
    dummy_input = torch.randint(1, VOCAB_SIZE, (2, 5))
    initial_prediction = model(dummy_input)
    print(f"Initial Prediction (Untrained): {initial_prediction.detach().numpy()}")

    # 3. Save to disk
    CHECKPOINT_PATH = "bilstm_v1.pth"
    save_model(model, CHECKPOINT_PATH)

    # 4. Create a completely new, blank model
    new_model = BiLSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)

    # 5. Load the saved weights into the new model
    new_model = load_model(new_model, CHECKPOINT_PATH)

    # 6. Verify identical outputs
    loaded_prediction = new_model(dummy_input)
    print(f"Loaded Prediction (Restored):   {loaded_prediction.detach().numpy()}")
