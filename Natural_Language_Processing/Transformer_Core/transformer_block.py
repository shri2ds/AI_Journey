import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.2):
        super().__init__()

        # 1. Multi-Head Attention Layer
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # 2. Normalization Layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 3. Feed-Forward Network
        # It typically expands the dimension (ff_dim) and then shrinks it back
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Step 1: Attention + Residual + Norm
        # We apply attention, add the original x (Residual), then Normalize
        attn_output = self.attention(x)
        x = self.norm1(attn_output + x)

        # Step 2: Feed-Forward + Residual + Norm
        ffn_output = self.ffn(x)
        x = self.norm2(ffn_output + x)

        return x


if __name__ == "__main__":
    # Test Parameters
    BATCH_SIZE = 2
    SEQ_LEN = 10
    EMBED_DIM = 128
    NUM_HEADS = 8
    FF_DIM = 512  # The "Expansion" dimension for the FFN

    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    block = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM)

    output = block(x)

    print("--- Transformer Block Execution ---")
    print(f"Input Shape:  {x.shape}")
    print(f"Output Shape: {output.shape}")
    print("\n✅ The Transformer Block successfully processed the sequence while maintaining dimensions.")
