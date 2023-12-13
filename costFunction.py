import numpy as np
from sigmoid import *


def cost_function(theta, X, y):
    m = y.size

    # Initialize cost and gradient to be zero
    cost = 0
    gradient = np.zeros(theta.shape)

    # Here the hypothesis is the sigmoid of the dot product of the feature vector (X) and theta (parameters)
    hypothesis = sigmoid(np.dot(X, theta))

    # Compute the cost using the logistic regression cost function
    # Average log-loss over all training examples
    # Penalizes wrong choices with higher penalty for wronger
    # Cost is averaged over all training examples
    cost = np.sum(-y * np.log(hypothesis) - (1-y) * np.log(1-hypothesis)) / m

    # Compute the gradient
    # Partial derivative of the cost function with respect to theta
    # Result is averaged over all the training samples
    gradient = np.dot(X.T, (hypothesis - y)) / m

    # ===========================================================

    return cost, gradient
