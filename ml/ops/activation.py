import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


def rectified_linear(z):
    # see https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    return np.maximum(z, 0)
