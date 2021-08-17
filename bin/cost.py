import numpy as np


def mean_squared_error(x, y):
    return (1 / len(x)) * np.sum((np.array(x) - np.array(y))**2)

def mean_squared_error_der(x, y):
    return 2 * (np.array(x) - np.array(y))