import numpy as np


def change_mayor(a, time=True):
    axis1 = 0
    axis2 = 2
    if not time:
        axis2 = 1
    return np.swapaxes(a, axis1, axis2)
