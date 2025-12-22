"""Simple two-layer neural network training on MNIST using NumPy."""

import numpy as np
from tensorflow.keras.datasets import mnist

def load_data():
    """Load, reshape, and normalize the MNIST train/test splits.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test) with samples flattened and transposed.
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Flatten each 28x28 image to a vector and transpose to (features, samples)
    X_train = X_train.reshape(X_train.shape[0], -1).T
    X_test = X_test.reshape(X_test.shape[0], -1).T

    # Scale pixel intensities into [0, 1]
    X_train = X_train / 255
    X_test = X_test / 255

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data()
print(f"Shape check: X_train is {X_train.shape}") # Should be (784, 60000)

def init_params():
    """Randomly initialize weights and biases for the two-layer network."""

    # Subtract 0.5 to center weights around 0
    W1 = np.random.rand(10, 784) - 0.5  # first layer weights (10 neurons)
    b1 = np.random.rand(10, 1) - 0.5

    W2 = np.random.rand(10, 10) - 0.5  # second layer weights (10 output classes)
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2

def ReLU(Z):
    """Apply element-wise ReLU activation."""
    return np.maximum(0, Z)

def softmax(Z):
    """Compute softmax activation along each column (sample)."""
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    """Perform forward propagation through the two-layer network."""

    # Layer 1 -> ReLU
    Z1 = np.dot(W1, X) + b1
    A1 = ReLU(Z1)

    # Layer 2 -> Softmax
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def one_hot(Y):
    """Convert integer class labels to one-hot encoded columns."""
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def deriv_ReLU(Z):
    """Compute derivative of ReLU (1 where input positive)."""
    return Z > 0

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    """Compute gradients of loss w.r.t. all parameters."""

    m = Y.size
    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    """Apply gradient descent update for all parameters."""

    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    """Return predicted class indices for each sample."""

    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    """Compute accuracy given predicted labels and ground-truth."""

    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    """Run gradient descent for a fixed number of iterations."""

    W1, b1, W2, b2 = init_params()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print("Accuracy: ", get_accuracy(predictions, Y))

    return W1, b1, W2, b2


print("Starting Training...")
W1, b1, W2, b2 = gradient_descent(X_train, y_train, 0.10, 500)
