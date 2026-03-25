import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        # Embed_Dim must be divisible by Num_Heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear Layers for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Final Linear Layer to mix the heads back together
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Embed_Dim]
        batch_size, seq_len, _ = x.shape

        # Step A: Linear Projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Step B: Split the Heads
        # We reshape Embed_Dim into [Num_Heads, Head_Dim]
        # New Shape: [Batch, Seq_Len, Num_Heads, Head_Dim]
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Step C: Transpose for Attention Calculation
        # We want the Num_Heads dimension to be treated like a batch dimension
        # Swap dimension 1 (Seq_Len) and 2 (Num_Heads)
        # New Shape: [Batch, Num_Heads, Seq_Len, Head_Dim]
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        # Step D: Scaled Dot-Product Attention (Logic from Day 12)
        # [Batch, Heads, Seq_Len, Head_Dim] * [Batch, Heads, Head_Dim, Seq_Len]
        # Output: [Batch, Heads, Seq_Len, Seq_Len]
        scores = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply by Values
        # Output: [Batch, Heads, Seq_Len, Head_Dim]
        out = torch.matmul(attention_weights, V)

        # Step E: Concatenate Heads
        # Undo the transpose: [Batch, Seq_Len, Heads, Head_Dim]
        out = out.transpose(1, 2)

        # Flatten the last two dimensions back into Embed_Dim
        # [Batch, Seq_Len, Embed_Dim]
        out = out.reshape(batch_size, seq_len, self.embed_dim)

        return self.fc_out(out)


if __name__ == "__main__":
    # Test Parameters
    BATCH_SIZE = 2
    SEQ_LEN = 5
    EMBED_DIM = 64  # Must be divisible by 8
    NUM_HEADS = 8

    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    mha = MultiHeadAttention(EMBED_DIM, NUM_HEADS)

    output = mha(x)

    print("--- Multi-Head Attention Shapes ---")
    print(f"Input Shape:  {x.shape}")
    print(f"Output Shape: {output.shape}")
    print("\n✅ If Input == Output shape, the block is preserving dimensions correctly.")
