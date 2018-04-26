import numpy as np


class Activation(object):
    __slots__ = ()

    @staticmethod
    def get_derivative(fn):
        dfn = getattr(fn.__self__,
                      "{0}_derivative".format(fn.__name__),
                      None)
        if dfn is None:
            dfn = getattr(fn.__self__, "derivative", None)
        return dfn


class Sigmoid(object):
    # See https://en.wikipedia.org/wiki/Sigmoid_function
    __slots__ = ()

    @classmethod
    def logistic(cls, z):
        return 1.0 / (1.0 + (np.exp(-z, dtype=np.dtype('Float64'))
                             .astype(z.dtype)))

    @classmethod
    def logistic_derivative(cls, z):
        _sigmoid = cls.logistic(z)
        return _sigmoid * (1 - _sigmoid)


class Hyperbolic(object):
    # See https://en.wikipedia.org/wiki/Hyperbolic_function
    __slots__ = ()

    @classmethod
    def tanh(cls, z):
        return np.tanh(z.astype(np.float64))

    @classmethod
    def tanh_derivative(cls, z):
        return 1 - np.square(cls.tanh(z))


class ReLU(object):
    # See https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    __slots__ = ()

    @classmethod
    def noisy(cls, z):
        return np.maximum(z, 0)

    @classmethod
    def noisy_derivative(cls, z):
        return np.where(z > 0, 1, 0)


class Softmax(object):
    # See https://en.wikipedia.org/wiki/Softmax_function
    __slots__ = ()

    @classmethod
    def probability(cls, z):
        z = z.astype(np.dtype('Float64'))
        logits_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return logits_exp / np.sum(logits_exp, axis=1, keepdims=True)

    @classmethod
    def derivative(cls, z):
        z = z.astype(np.dtype('Float64'))
        logits_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        div = np.sum(logits_exp, axis=1, keepdims=True)
        logits_exp_1 = logits_exp[:, 0]
        logits_exp_2 = logits_exp[:, 1:]
        logits_exp_2 = np.sum(logits_exp_2, axis=1, keepdims=True)
        return (logits_exp_1 * logits_exp_2) / np.square(div)
