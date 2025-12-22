import numpy as np

class LogisticRegressionScratch:
    def __init__(self):
        self.theta = None
        self.cost_history = []

    def _sigmoid(self, z):
        """
        Sigmoid activation function sig(z) = 1/(1 + e^(-z))
        """
        np.clip(z, -250, 250)
        return 1/(1 + np.exp(-z))

    def fit(self, X, y, epochs=1000, lr=0.01):
        """
        Train Logistic Regression using Gradient descent

        Args:
            X : Feature vector matrix (m x n)
            y : Target vector (m x 1) with binary labels - 0 or 1
            epochs : Number of iterations
            lr : learning rate
        """
        m = len(y)
        X_b = np.c_[np.ones((m, 1)), 1] #Added bias
        self.theta = np.random.randn(X_b.shape[1], 1) #Random initialisation

        for epoch in epochs:
            #Forward Pass
            z = np.dot(X_b, self.theta) 
            predictions = self._sigmoid(z) #Appyling sigmoid activation

            gradients = (1/m) * np.dot(X_b.T, (predictions - y)) #Calculating gradient

            self.theta -= lr * gradients #Updating the params

            epsilon = 1e-15
            # Compute Cost = -1/m * Σ[y*log(h) + (1-y)*log(1-h)] using Binary Cross Entropy 
            cost = (1/m) * np.sum(
                y * np.log(predictions + epsilon) +
                (1-y) * np.log(1 - predictions + epsilon)
            )
            self.cost_history.append(cost)

    def predict_proba(self, X):
        """
        Predict Probability for each sample

        Returns:
            Probability 
        """
        X_b = np.c_[np.ones((X.shape[0], 1))]
        z = np.dot(X_b, self.theta)
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        Predict Binary class labels

        Args:
            X : Feature vector matrix
            threshold : Decision Boundary

        Returns:
            Binary Prediction : 1 if prediction > threshold else 0
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
            


          
