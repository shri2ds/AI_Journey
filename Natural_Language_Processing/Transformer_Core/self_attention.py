import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # The three distinct linear projections
        # Input: embed_dim -> Output: embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Embed_Dim]

        # Create Q, K, V from the exact same input
        Q = self.W_q(x)     # [Batch, Seq_Len, Embed_Dim]
        K = self.W_k(x)
        V = self.W_v(x)

        K_transposed = K.transpose(1,2)
        raw_scores = torch.bmm(Q, K_transposed)

        # Scale the scores
        scaled_scores = raw_scores/np.sqrt(self.embed_dim)

        # We apply softmax over the last dimension & build context vector
        attention_weights = torch.softmax(scaled_scores, dim=-1)
        context_vector = torch.bmm(attention_weights, V)

        return context_vector, attention_weights


if __name__ == "__main__":
    BATCH_SIZE = 2
    SEQ_LEN = 4  # e.g., "The bank of river"
    EMBED_DIM = 16
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)


    attention_layer = SelfAttention(EMBED_DIM)
    context, weights = attention_layer(x)

    print("--- Self-Attention Execution ---")
    print(f"Input Shape:   {x.shape}")
    print(f"Context Shape: {context.shape} (Notice it matches the input!)")
    print(f"\nAttention Weights for Batch 0 (Shape: {weights[0].shape}):\n{weights[0].detach().numpy()}")
