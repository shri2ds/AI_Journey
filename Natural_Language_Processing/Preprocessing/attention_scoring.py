import torch
import torch.nn.functional as F


def calculate_attention(query, keys, values):
    """
    Computes standard Dot-Product Attention.

    Args:
        query:  [Batch, 1, Hidden_Dim]       (The Decoder's current state)
        keys:   [Batch, Seq_Len, Hidden_Dim] (The Encoder's states)
        values: [Batch, Seq_Len, Hidden_Dim] (Often the exact same tensor as keys)
    """
    print(f"Query shape:  {query.shape}")
    print(f"Keys shape:   {keys.shape}")

    # Calculate Scores: Q * K^T
    keys_transposed = keys.transpose(1, 2)
    print(f"Keys Transposed shape: {keys_transposed.shape}")

    # torch.bmm ==> [Batch, 1, Hidden] x [Batch, Hidden, Seq_Len] -> [Batch, 1, Seq_Len]
    raw_scores = torch.bmm(query, keys_transposed)
    print(f"Raw Scores shape: {raw_scores.shape}")

    # Softmax to get Attention Weights
    attention_weights = F.softmax(raw_scores, dim=-1)

    # Calculate the Context Vector: Weights * Values
    # [Batch, 1, Seq_Len] x [Batch, Seq_Len, Hidden] -> [Batch, 1, Hidden]
    context_vector = torch.bmm(attention_weights, values)
    print(f"Context Vector shape: {context_vector.shape}")

    return context_vector, attention_weights


if __name__ == "__main__":
    BATCH_SIZE = 2
    SEQ_LEN = 5  
    HIDDEN_DIM = 16  

    # Simulating the Decoder's current thought process
    q = torch.randn(BATCH_SIZE, 1, HIDDEN_DIM)

    # Simulating the Encoder's memory of the 5 input words
    k = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    v = k  # Init v=k for sample testing purpose

    print("--- Executing Attention Mechanism ---\n")
    context, weights = calculate_attention(q, k, v)

    print("\n--- Results ---")
    print(f"Attention Weights for Batch 0:\n{weights[0].detach().numpy()}")
    # Notice how the weights for the 5 words sum exactly to 1.0!
