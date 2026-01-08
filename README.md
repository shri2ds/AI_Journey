# AI & Machine Learning Engineering Journey 🚀

> **From Mathematical Foundations to Production Deployment.**

This repository documents my intensive journey transitioning into AI Engineering. It focuses on three core pillars:
1.  **Mathematics:** Implementing algorithms (Linear Regression, Backprop) from scratch without libraries.
2.  **Engineering:** Deploying models using Docker, FastAPI, and Cloud Platforms.
3.  **Deep Learning:** Building Neural Networks, CNNs, and Transformers using PyTorch.

---

## 📂 Repository Structure

### 🔹 [Classical Machine Learning](./Classical_Machine_Learning)
Bridging the gap between theory and software engineering.
* **Algorithms from Scratch:** Python implementations of Linear/Logistic Regression and Neural Networks to understand the calculus of Gradient Descent.
* **Production Deployment:** A full end-to-end MLOps pipeline for the Titanic Survival model.
    * **Tech Stack:** `XGBoost`, `FastAPI`, `Docker` and `Render`.

### 🔹 [Deep_Learning](./Deep_Learning)
**The internals of PyTorch and Modern Architectures.**

| Module | Status | Key Concepts |
| :--- | :--- | :--- |
| **1. Tensor Physics** | ✅ Completed | Strides, Views, Broadcasting, Contiguity. |
| **2. Optimization** | ✅ Completed | Autograd, He Initialization, The "Dying ReLU" Experiment. |
| **3. Modern MLP** | ✅ Completed | Batch Normalization, Dropout, Adam, Inference Mode Crash Tests. |
| **4. CNNs & Vision** | 🚧 In Progress | Convolution Mechanics, ResNet, Transfer Learning. |


#### 🧪 Latest Engineering Experiment
**The "Dying ReLU" Phenomenon**
I intentionally initialized a Neural Network with negative weights to demonstrate the "Dying ReLU" problem.
* **Outcome:** Gradients dropped to `0.0` immediately. The model became brain-dead.
* **Solution:** Applied **Kaiming (He) Initialization** to mathematically preserve variance across layers.


**Status: Currently mastering Convolutional Neural Networks (CNNs).**


### 🔹 [LeetCode DSA](./LeetCode_DSA)
Daily problem-solving to sharpen algorithmic thinking. Focus on Trees, Graphs, and Dynamic Programming.

---

## 🛠️ Tech Stack
* **Languages:** Python, SQL
* **Libraries:** PyTorch, NumPy, Pandas, Scikit-Learn
* **Ops:** Docker, FastAPI, Git, CI/CD

---
*Author: Shridhar Bhandar*

---
## 🧵 Contact
For discussions, improvements or collaborations, reach me via [LinkedIn](https://www.linkedin.com/in/shridhar-bhandar/).
