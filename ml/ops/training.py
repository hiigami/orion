import numpy as np


class Adam(object):
    # https://arxiv.org/pdf/1412.6980.pdf
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=np.spacing(1e-8)):
        self._alpha = alpha
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
