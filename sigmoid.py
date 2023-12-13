import numpy as np


def sigmoid(z):
    g = np.zeros(z.size)

    # Sigmoid function works for matrix, vector, or scalar
    g = 1 / (1 + np.exp(-z))


    return g
