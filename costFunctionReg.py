import numpy as np
from sigmoid import *


def cost_function_reg(theta, X, y, lmd):
    m = y.size

    # Initialize cost and gradient vector
    cost = 0
    gradient = np.zeros(theta.shape)

    # Use sigmoid function to compute hypothesis (predictions)
    hypothesis = sigmoid(np.dot(X, theta))

    # Remove the intercept term from regularization
    reg_theta = theta[1:]
    
    # Compute the regularized cost
    # first portion is the same as standard log regression
    # second portion is regularization term
    cost = ((np.sum(-y * np.log(hypothesis) - (1 - y) * np.log(1-hypothesis))) / m ) + (lmd / (2*m)) * np.sum(reg_theta * reg_theta)

    # Compute the gradient for each theta parameter
    normal_grad = (np.dot(X.T, hypothesis - y)/m).flatten()

    # Set the gradient term for the intercept term
    gradient[0] = normal_grad[0]
    # Set the gradient for other parameters with regularization
    gradient[1:] = normal_grad[1:] + reg_theta * (lmd / m)

    return cost, gradient
