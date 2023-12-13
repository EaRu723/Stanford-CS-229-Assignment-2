import numpy as np
from sigmoid import *


def predict(theta, X):
    # Get the number of examples
    m = X.shape[0]

    # RInitialize prediction vector
    p = np.zeros(m)

    # Compute the predictions using the sigmoid
    p = sigmoid(np.dot(X, theta))

    # Convert to oistivie or negative class prediction
    positive = np.where(p >= 0.5)
    negative = np.where(p < 0.5)

    p[positive] = 1
    p[negative] = 0

    return p
