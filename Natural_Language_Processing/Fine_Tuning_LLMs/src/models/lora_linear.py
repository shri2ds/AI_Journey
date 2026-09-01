"""
Bare-Metal PyTorch Implementation of Low-Rank Adaptation (LoRA).
Reference: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)

This module builds LoRA from first principles without using the Hugging Face `peft` library,
demonstrating base weight freezing, low-rank factorization, forward scaling,
and zero-latency inference weight merging.
"""

import math
import torch
import torch.nn as nn
from typing import Optional

class LoRALinear(nn.Module):
    """
    Custom Linear layer wrapping a standard frozen nn.Linear layer with
    parallel trainable low-rank decomposition matrices A and B.

    Mathematical Forward Pass:
        h = W_0 * x + (alpha / r) * (x * A^T) * B^T
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.05,
                 bias: bool = False
                 ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        # 1. Base Pre-trained Linear Layer (FROZEN)
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        self.base_layer.weight.requires_grad = False
        if bias:
            self.base_layer.bias.requires_grad = False

        # 2. Trainable Low-Rank Adapter Matrices
        if r > 0:
            # Down-projection matrix A: R^(r x in_features)
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            # Up-projection matrix B: R^(out_features x r)
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))

            # Dropout applied to input activations before down-projection
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

            # State tracker for merged weights
            self.merged = False

            # 3. Initialize weights according to Hu et al.
            self.reset_parameters()

        else:
            self.lora_A = None
            self.lora_B = None
            self.merged = False

    def reset_parameters(self):
        """
        Initialization Rule (Hu et al., Section 4.1):
        - Matrix A: Kaiming Uniform / Gaussian distribution
        - Matrix B: Strictly ZEROS
        Guarantees that delta_W = B * A = 0 at step 0 (zero initial noise).
        """

        if self.r > 0:
            # Initialize A with Kaiming uniform (scaled by sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

            # Initialize B to exact zeros
            nn.init.zeros_(self.lora_B)

    def merge_weights(self):
        """
        Zero-Inference Latency Weight Folding:
        Fuses the adapter weights directly into the base layer weights:
            W_serving = W_0 + (alpha / r) * (B * A)
        """

        if self.r > 0 and not self.merged:
            # Compute low-rank delta: (out_features x in_features)
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            # Add directly to frozen base weights in-place
            self.base_layer.weight.data += delta_w
            self.merged = True
            print("✅ Weights successfully merged into base layer. LoRA branch is now fused.")

    def unmerge_weights(self):
        """
        Unfolds the adapter weights from the base layer to resume training:
            W_0 = W_serving - (alpha / r) * (B * A)
        """
        if self.r > 0 and self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta_w
            self.merged = False
            print("🔄 Weights unmerged. Layer restored to dual-branch training state.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass computation:
        - If merged (production inference): h = x * W_serving^T
        - If unmerged (training): h = x * W_0^T + (alpha / r) * (dropout(x) * A^T) * B^T
        """

        # Base linear transformation
        result = self.base_layer(x)

        # If not merged and LoRA is active, compute low-rank residual path
        if self.r > 0 and not self.merged:
            # Step 1: Apply input dropout and project down via A
            # x: [Batch, Seq_Len, in_features] -> [Batch, Seq_Len, r]
            lora_down = self.lora_dropout(x) @ self.lora_A.T

            # Step 2: Project back up via B and apply scaling factor
            # [Batch, Seq_Len, r] -> [Batch, Seq_Len, out_features]
            lora_up = (lora_down @ self.lora_B.T) * self.scaling

            # Step 3: Additive residual fusion
            result = result + lora_up

        return result

# -----------------------------------------------------------------------------
# Verification Harness: Testing LoRA Properties & Gradient Isolation
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 4
    d_in = 4096
    d_out = 4096
    rank = 16
    alpha = 32

    print("=" * 70)
    print("🧠 BARE-METAL PYTORCH LoRA LAYER VERIFICATION")
    print("=" * 70)

    # 1. Instantiate Custom LoRA Layer
    layer = LoRALinear(in_features=d_in, out_features=d_out, r=rank, lora_alpha=alpha)

    # 2. Inspect Parameter Counts and Gradient Tracking
    total_params = sum(p.numel() for p in layer.parameters())
    trainable_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in layer.parameters() if not p.requires_grad)

    print("\n[STEP 1] Parameter Allocation Audit:")
    print(f"  Base Layer (Frozen) : {frozen_params:,} params ({frozen_params * 4 / (1024 ** 2):.2f} MB in FP32)")
    print(f"  LoRA Adapter A      : {layer.lora_A.numel():,} params")
    print(f"  LoRA Adapter B      : {layer.lora_B.numel():,} params")
    print(
        f"  Total Trainable     : {trainable_params:,} params ({trainable_params * 4 / (1024 ** 2):.2f} MB in FP32)")
    print(f"  Parameter Reduction : {100 * (1 - trainable_params / total_params):.2f}% reduction!")

    # 3. Verify Step 0 Output Equivalence (Zero Initial Noise)
    dummy_input = torch.randn(batch_size, seq_len, d_in)
    layer.eval()  # Set to evaluation mode to disable dropout during verification
    with torch.no_grad():
        output_lora_init = layer(dummy_input)
        output_base_only = layer.base_layer(dummy_input)

    diff_step_0 = torch.max(torch.abs(output_lora_init - output_base_only)).item()
    print("\n" + "-" * 70)
    print(f"[STEP 2] Initial Noise Verification (t=0):")
    print(f"  Max Absolute Difference between Base and LoRA output: {diff_step_0:.8f}")
    assert diff_step_0 == 0.0, "❌ Error: LoRA output at step 0 must match base output exactly."
    print("  ✅ PASS: LoRA output is mathematically identical to pre-trained base model at step 0.")

    # 4. Verify Backward Pass & Gradient Flow Isolation
    print("\n" + "-" * 70)
    print("[STEP 3] Gradient Backpropagation Isolation Test:")
    layer.train()  # Switch back to train mode for backprop
    output = layer(dummy_input)
    dummy_loss = output.sum()
    dummy_loss.backward()

    print(f"  Base Layer Weight Grad is None? : {layer.base_layer.weight.grad is None} (Gradients FROZEN)")
    print(f"  LoRA Matrix A Grad is Computed? : {layer.lora_A.grad is not None} (Shape: {layer.lora_A.grad.shape})")
    print(f"  LoRA Matrix B Grad is Computed? : {layer.lora_B.grad is not None} (Shape: {layer.lora_B.grad.shape})")
    assert layer.base_layer.weight.grad is None, "Base weight must not track gradients!"
    assert layer.lora_A.grad is not None and layer.lora_B.grad is not None, "Adapters must track gradients!"
    print("  ✅ PASS: Only low-rank matrices A and B receive gradient updates.")

    # 5. Verify Zero-Inference Latency Weight Merging
    print("\n" + "-" * 70)
    print("[STEP 4] Zero-Latency Weight Merging Test:")
    # Switch to eval mode so dropout does not introduce random variance during inference comparison
    layer.eval()

    # Simulate an update to adapter weights
    with torch.no_grad():
        layer.lora_B.data.fill_(0.01)

    # Output before merging
    output_unmerged = layer(dummy_input)

    # Merge weights into base layer
    layer.merge_weights()

    # Output after merging
    output_merged = layer(dummy_input)

    diff_merge = torch.max(torch.abs(output_unmerged - output_merged)).item()
    print(f"  Max Difference between Unmerged and Merged execution: {diff_merge:.8f}")
    assert diff_merge < 1e-5, "Merged layer output must match unmerged dual-branch computation."
    print("  ✅ PASS: Fused weight matrix matches dual-branch computation perfectly.")

    print("\n" + "=" * 70)
    print("🎉 ALL PYTORCH LoRA FIRST-PRINCIPLES TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
