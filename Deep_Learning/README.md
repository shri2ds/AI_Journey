# 🧠 Deep Learning Engineering

> **From Mathematical Internals to Production Architectures.**

This module documents my progression from understanding the low-level physics of PyTorch tensors to building advanced Neural Network architectures. Unlike standard tutorials, this section focuses on **failure modes**, **memory internals**, and **mathematical verification** of Deep Learning concepts.

---

## 📂 Module Structure

### 1. [Framework Basics (MNIST)](./Framework_Basics_MNIST)
**"The Hello World of Deep Learning"**
* **Goal:** A baseline implementation of a Feed-Forward Neural Network (MLP) to classify handwritten digits.
* **Key Components:**
    * Custom `nn.Module` class design.
    * Manual Training Loop (Forward -> Loss -> Backward -> Step).
    * Evaluation metrics and tensor reshaping.

### 2. [PyTorch Tensor Basics](./PYTorch_Tensor_Basics)
**"The Physics of Data"**
* **Goal:** Understanding how PyTorch actually manages memory, preventing silent bugs in production.
* **Key Concepts:**
    * **Storage vs. View:** Proving that `reshape()` does not move data, and how `stride()` math works.
    * **Contiguity:** diagnosing and fixing `RuntimeError: view size is not compatible with input tensor's size and stride`.
    * **Broadcasting:** Visualizing how dimensions "stretch" during arithmetic operations.

### 3. [Initialization & The Dying ReLU](./Initialization_Relu)
**"Debugging Neural Network Failures"**
* **Goal:** A practical experiment demonstrating why deep networks stop learning (The "Dying ReLU" problem).
* **The Experiment:**
    * Intentionally initialized a network with negative weights and positive inputs.
    * **Result:** Proved that gradients become `0.0`, causing the network to "die" (stop updating).
* **The Fix:** Implemented **He Initialization (Kaiming Init)** to mathematically stabilize variance across layers.

---

## 🛠️ Key Technical Takeaways
* **Debugging:** I don't just call `fit()`. I inspect gradients (`tensor.grad`) to diagnose vanishing/exploding gradients.
* **Optimization:** Understanding why specific initializations (Xavier/He) are mathematically required for specific activation functions (Tanh/ReLU).
* **Internals:** Leveraging `strides` and `views` for memory-efficient data manipulation.

---
*Next Steps: CNN Architectures (ResNet), Batch Normalization, and Object Detection.*
