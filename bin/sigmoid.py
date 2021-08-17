import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_der(x):
    return 1 / (1 + np.exp(-x)) * (1 - (1 / (1 + np.exp(-x))))