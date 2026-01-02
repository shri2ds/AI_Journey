# Hand-Coded Neural Network (MNIST) 🧠

> **Project Goal:** Build a Deep Learning framework from scratch using only **NumPy** to master the mathematics behind Backpropagation, Gradient Descent, and Matrix Calculus.

## 🚀 Overview
This repository contains a raw implementation of a Multi-Layer Perceptron (MLP) designed to classify handwritten digits from the **MNIST dataset**. 

Unlike standard implementations that rely on high-level frameworks like PyTorch or TensorFlow, this project builds the **computational graph manually**. Every forward pass calculation, activation function derivative, and weight update is mathematically derived and implemented via vectorization.

## 📊 Performance
* **Accuracy:** ~84.4% (on Test Set)
* **Training Time:** < 30 seconds (500 Iterations)
* **Architecture:** 2-Layer MLP (784 Input $\to$ 10 Hidden $\to$ 10 Output)

## 🛠️ Tech Stack & Concepts
* **Language:** Python
* **Core Library:** NumPy (Linear Algebra & Matrix Operations)
* **Data Loading:** Keras (Used *only* to download the dataset)
* **Key Concepts:** * Matrix Calculus (Chain Rule)
    * Vectorization (No `for` loops in training)
    * ReLU & Softmax Activations
    * Categorical Cross-Entropy Loss

---

## 📐 The Architecture & Math

### 1. Forward Propagation
The network consists of an input layer, one hidden layer with **ReLU** activation, and an output layer with **Softmax** activation.

* **Layer 1 (Hidden):**
$$Z^{[1]} = W^{[1]} X + b^{[1]}$$
$$A^{[1]} = \text{ReLU}(Z^{[1]})$$

* **Layer 2 (Output):**
$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$
$$A^{[2]} = \text{softmax}(Z^{[2]})$$

### 2. Backward Propagation (The Learning Engine)
Deriving gradients manually using the Chain Rule to minimize the Cross-Entropy Loss.

* **Output Error:**
$$dZ^{[2]} = A^{[2]} - Y$$

* **Hidden Layer Error (Reflected):**
$$dZ^{[1]} = W^{[2]T} dZ^{[2]} \cdot g'(Z^{[1]})$$
*(Where $g'$ is the derivative of ReLU)*

* **Gradients:**
$$dW^{[l]} = \frac{1}{m} dZ^{[l]} A^{[l-1]T}$$
$$db^{[l]} = \frac{1}{m} \sum dZ^{[l]}$$

---

## 💻 Code Highlight: Vectorization
Instead of iterating over 60,000 images, the implementation uses NumPy broadcasting to process the entire batch simultaneously.

```python
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    
    # Layer 2 Gradients
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * np.dot(dZ2, A1.T)  # Matrix Multiplication
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    
    # Layer 1 Gradients (The Chain Rule)
    dZ1 = np.dot(W2.T, dZ2) * deriv_ReLU(Z1) 
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2
```

---
# PyTorch Implementation (AutoGrad) 🔥

> **Project Goal:** Re-implement the MNIST classifier using **PyTorch** to leverage Automatic Differentiation (AutoGrad) and GPU acceleration.

## 🚀 Key Differences from "From Scratch"
This implementation reduces the code complexity by 90% by abstracting the backward pass.

| Feature | NumPy Implementation | PyTorch Implementation |
| :--- | :--- | :--- |
| **Gradients** | Manually derived (Calculus) | `loss.backward()` (AutoGrad) |
| **Weights** | Dictionary of Arrays | `nn.Linear` Layers |
| **Device** | CPU Only | CPU / GPU / MPS |
| **Optimization** | Manual Update Rule | `optim.SGD` / `optim.Adam` |

## 💻 Code Snippet
The entire training loop is simplified to:

```python
# The "Magic" Step
optimizer.zero_grad()   # Clear previous gradients
loss.backward()         # Calculate gradients automatically
optimizer.step()        # Update weights
```
---
