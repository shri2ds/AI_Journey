import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionScratch:
    def __init__(self):
        self.theta = None

    def fit_normal_eq(self, X, y):
        # Step 1: Add a column of ones to X for the intercept (bias term)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Step 2: Implement the formula: theta = (X^T * X)^-1 * X^T * y
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def fit_gd(self, X, y, learning_rate=0.01, n_iterations=1000):
        m = len(y)
        X_b = np.c_[np.ones((m, 1)), X]  # Add bias term
        self.theta = np.random.randn(X_b.shape[1], 1) # Random initialization
        self.cost_history = []

        for iteration in range(n_iterations):
            gradients = (1/m) * X_b.T.dot(X_b.dot(self.theta) - y) # The Vectorized Math
            self.theta = self.theta - learning_rate * gradients
            
            # Log Cost 
            cost = (1/(2*m)) * np.sum(np.square(X_b.dot(self.theta) - y))
            self.cost_history.append(cost)

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)
    
    def plot_cost_history(self):
        """Plot the cost function over iterations to diagnose learning rate."""
        if not hasattr(self, 'cost_history') or len(self.cost_history) == 0:
            print("No cost history to plot. Train the model first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history, linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cost J(θ)', fontsize=12)
        plt.title('Cost Function vs Iterations', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add diagnostic text
        final_cost = self.cost_history[-1]
        initial_cost = self.cost_history[0]
        plt.text(0.02, 0.98, f'Initial Cost: {initial_cost:.4f}\nFinal Cost: {final_cost:.4f}',
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig("loss_curve.png") 
        plt.show()


# Test it
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

model = LinearRegressionScratch()
# Commented this for batch gradient descent algorithm
# model.fit_normal_eq(X, y) 
model.fit_gd(X, y, learning_rate=0.01, n_iterations=1000)
print(f"\nDataset: X is {X.shape}, y is {y.shape}")
print(f"Predicted Theta: {model.theta.T}")
print(f"Expected: [[4], [3]] (approximately)")
print(f"Final Cost: {model.cost_history[-1]:.4f}")

# Plot cost history
model.plot_cost_history()
