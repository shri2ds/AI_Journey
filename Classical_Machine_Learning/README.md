# Classical Machine Learning & Algorithms

This directory contains two types of work:
1.  **First Principles Implementation:** Coding algorithms using only `NumPy` to understand the underlying math.
2.  **MLOps Implementation:** Taking a trained model and wrapping it in a production-grade microservice.

## 🧠 Algorithms from Scratch
I re-implemented core algorithms to master the optimization logic.

### 1. Linear Regression
#### Model Performance: Convergence Check
We used Gradient Descent to optimize the parameters. The graph below shows the cost function ($J(\theta)$) decreasing over iterations, confirming that the learning rate was tuned correctly.

* **Logic:** Implemented Batch Gradient Descent.
* **Key Math:**
    $$\theta = \theta - \alpha \cdot \frac{1}{m} \sum (h_\theta(x^{(i)}) - y^{(i)})x^{(i)}$$

* **Result:** Verified convergence by plotting the Loss Curve.
  
    ![Loss Curve](./Algorithm_from_Scratch/loss_curve.png)

*Observation: The loss drops exponentially and stabilizes around iteration 200, indicating convergence.*

### 2. Logistic Regression

### Mathematical Foundation: Binary Cross Entropy (Log Loss)
For binary classification, we model the target $y$ as a Bernoulli distribution. Instead of Mean Squared Error (MSE), which results in a non-convex loss surface for sigmoid activation, we use **Log Loss**.

**The Cost Function:**
$$J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1 - \hat{y}^{(i)})]$$

**Why this works:**
1.  **Convexity:** It guarantees a global minimum, allowing Gradient Descent to converge.
2.  **Maximum Likelihood:** Minimizing this loss is mathematically equivalent to maximizing the likelihood of the observed data.
3.  **Penalty:** It heavily penalizes "confident but wrong" predictions (e.g., predicting 0.99 probability for a negative class).

## 🚀 Production Project: [Titanic API](./Titanic_API)
A containerized API that serves real-time predictions.
* **Model:** XGBoost Classifier.
* **Serving:** FastAPI (Asynchronous).
* **Containerization:** Docker (Multi-stage build).
* **Live Demo:** [https://titanic-api-g2d8.onrender.com/]
