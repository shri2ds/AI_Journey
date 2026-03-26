import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from NLP.Transformer_Core.transformer_block import TransformerBlock
from NLP.Transformer_Core.positional_encoding import PositionalEncoding

class LogSentinel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, max_len=5):
        super().__init__()

        # 1. Embedding Layer (Vocab -> Vector)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 2. Positional Encoding (Inject Order)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)

        # 3. Transformer Encoder Block (The Brain)
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim)

        # 4. Classification Head (Is it an attack?)
        self.fc_out = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [Batch, Seq_Len]
        x = self.embedding(x)       # [Batch, Seq_Len, Embed_Dim]
        x = self.pos_encoder(x)
        x = self.transformer(x)     # [Batch, Seq_Len, Embed_Dim]

        # For classification, we often take the mean of the sequence
        x = x.mean(dim=1)           # [Batch, Embed_Dim]
        return self.sigmoid(self.fc_out(x))





