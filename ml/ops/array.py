import numpy as np


def change_mayor(a):
    axis1 = 0
    axis2 = 1
    return np.swapaxes(a, axis1, axis2)


def _broadcast_attempt(a, b):
    try:
        np.broadcast(a, b)
        return True
    except ValueError:
        return False


def broadcast_attempt(a, b, axis=-1):
    if a.ndim != b.ndim:
        raise ValueError("Arrays are not broadcastable due to"
                         " different number of dimensions")
    if _broadcast_attempt(a, b):
        return b
    elif _broadcast_attempt(a, b.T):
        return b.T
    elif axis > -1:
        if a.shape[axis] == b.shape[axis]:
            return b
        index = b.shape.index(a.shape[axis])
        b = np.swapaxes(b, ((index + 1) % b.ndim), index)
    return b
