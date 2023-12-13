import numpy as np


def map_feature(x1, x2):
    # max degree for polynomial terms - chosen arbitrarily
    degree = 6

    # Reshape x1 and x2 to be column vectors
    x1 = x1.reshape((x1.size, 1))
    x2 = x2.reshape((x2.size, 1))
    # Initialize the results matrix with a column of 1s for the intercept term
    result = np.ones(x1[:, 0].shape)

    # Generate polynomial and interaction features
    # For each pair of degrees i(for x1) and j(for x2) where i + j <= degree
    # Create a new feature my multiplying x1^i and x2^j
    # np.c puts these new features in a result matrix
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            result = np.c_[result, (x1**(i-j)) * (x2**j)]

    return result
