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

### 4. [🚀 Modern Network Architecture](/.NN_Architecture) (Adam, BatchNorm, Dropout)
**This project builds a **Production-Grade Image Classifier** on MNIST.**
**Unlike the naive implementation in `Framework_Basics`, this version implements modern deep learning best practices to achieve **98% accuracy** and stability.**

#### 🔧 Key Improvements
   1.  **Batch Normalization:** Added `nn.BatchNorm1d` to stabilize layer inputs and prevent vanishing gradients.
   2.  **Dropout:** Added `nn.Dropout(0.2)` to force redundant feature learning (regularization).
   3.  **Adam Optimizer:** Switched from SGD to Adam (Adaptive Learning Rates) for 5x faster convergence.
   4.  **Mode Switching:** Explicit handling of `model.train()` vs `model.eval()`.

#### 🧪 The "Single Image" Stress Test
   I implemented a stress test to demonstrate why `model.eval()` is mandatory for inference.

#### The Experiment
   We fed a **single image** (Batch Size = 1) to the model in both modes.

#### The Results
   * **✅ EVAL Mode:**
       * **Behavior:** Uses global moving averages for normalization. Dropout is OFF.
       * **Result:** Correct Prediction (High Confidence).
   * **❌ TRAIN Mode:**
       * **Behavior:** Tries to calculate Mean/Std of the *current* batch (1 image).
       * **Result:** `ValueError` (Crash) or Garbage Output.
       * **Why:** Batch Norm cannot calculate Standard Deviation on a single value (Division by Zero).

#### 📂 Files
   * `ModrnMNIST.py`: The complete training pipeline with the stress test appended.

### 5. [CNN Mechanics](./CNN_Mechanics)
**[Beyond the Black Box](./CNN_Mechanics/ManualConv.py)**
* **Goal:** Manually implementing 2D Convolution to understand Feature Extraction.
* **The Experiment:** Created a manual "Sobel Filter" kernel to detect vertical edges without training.
* **Key Takeaway:** Proved how `Kernel Size` and `Stride` dictate the output feature map dimensions.

**[Dissecting a Production Model - RESNET](./CNN_Mechanics/Inspect_RESNET.py)**
* **Goal:** Understanding Downsampling and preparing for Transfer Learning.
* **The Concept:** Implemented **Max Pooling** manually to visualize how it reduces spatial dimensions while preserving dominant features.
* **The Experiment:**
    * Loaded a pre-trained **ResNet18** (trained on ImageNet).
    * Inspected the `state_dict` to see the raw learned weights.
    * **"Chopped the Head":** Programmatically replaced the final 1000-class `fc` layer with a custom 2-class layer, the foundational step for Transfer Learning.

---

## 🛠️ Key Technical Takeaways
* **Debugging:** I don't just call `fit()`. I inspect gradients (`tensor.grad`) to diagnose vanishing/exploding gradients.
* **Optimization:** Understanding why specific initializations (Xavier/He) are mathematically required for specific activation functions (Tanh/ReLU).
* **Internals:** Leveraging `strides` and `views` for memory-efficient data manipulation.

---
*Next Steps: CNN Architectures (ResNet), Batch Normalization, and Object Detection.*
