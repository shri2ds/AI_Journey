# Classical Machine Learning & Algorithms

This directory contains two types of work:
1.  **First Principles Implementation:** Coding algorithms using only `NumPy` to understand the underlying math.
2.  **MLOps Implementation:** Taking a trained model and wrapping it in a production-grade microservice.

## 🧠 Algorithms from Scratch
I re-implemented core algorithms to master the optimization logic.

### 1. Linear Regression
* **Logic:** Implemented Batch Gradient Descent.
* **Key Math:**
    $$\theta = \theta - \alpha \cdot \frac{1}{m} \sum (h_\theta(x^{(i)}) - y^{(i)})x^{(i)}$$
* **Result:** Verified convergence by plotting the Loss Curve.
    ![Loss Curve](./Algorithm_from_Scratch/loss_curve.png)

### 2. Logistic Regression
* **Logic:** Binary classification using the Sigmoid Activation function.
* **Loss Function:** Binary Cross Entropy (Log Loss).

## 🚀 Production Project: [Titanic API](./Titanic_API)
A containerized API that serves real-time predictions.
* **Model:** XGBoost Classifier.
* **Serving:** FastAPI (Asynchronous).
* **Containerization:** Docker (Multi-stage build).
* **Live Demo:** [https://titanic-api-g2d8.onrender.com/]
