import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()

        #   Create a matrix of [max_len, embed_dim] filled with zeros
        pe = torch.zeros(max_len, embed_dim)

        #   Create a vector representing positions [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(-1)

        #   Calculate the division term for the sine/cosine arguments
        #   We use the log space for numerical stability
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        #   Fill even indices (0, 2, 4...) with sine
        pe[:, 0::2] = torch.sin(position * div_term)

        #   Fill odd indices (1, 3, 5...) with cosine
        pe[:, 1::2] = torch.cos(position * div_term)

        #   Add a batch dimension and register as a buffer (stays on GPU but not trained)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Embed_Dim]
        # We add the positional encoding to the word embeddings
        x = x + self.pe[:, :x.size(1), :]
        return x


if __name__ == "__main__":
    EMBED_DIM = 128
    SEQ_LEN = 10

    # Simulating word embeddings
    x = torch.randn(1, SEQ_LEN, EMBED_DIM)

    pos_encoder = PositionalEncoding(EMBED_DIM)
    output = pos_encoder(x)

    print("--- Positional Encoding Test ---")
    print(f"Input Shape:  {x.shape}")
    print(f"Output Shape: {output.shape}")
    print("\n✅ Order has been mathematically injected into the embeddings.")
