import decimal

import numpy as np


class Matrix(object):

    __slots__ = (
        "_value",
        "_name",
        "_dtype",
        "_shape",
        "_index"
    )

    def __init__(self, dtype, shape, name, init_fn=None):
        self._name = name
        self._dtype = dtype
        self._shape = self._get_shape(shape)
        self._value = None
        if init_fn is not None:
            self._value = init_fn(self._shape, dtype=self._dtype)
        self._index = 0

    def __add__(self, other):
        if isinstance(other, Matrix):
            return self._value + other.value
        return self._value + other

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return np.dot(self._value, other.value)
        b = self._value.flatten()
        if (self._index >= len(b)):
            self._index = 0
        a = b[self._index] * other
        self._index += 1
        return a

    def __rmul__(self, other):
        if isinstance(other, Matrix):
            return self._value * other.value
        return self.__mul__(other)

    def __array_ufunc__(self, ufunc, method, *inputs, **other):
        _inputs = []
        for x in inputs:
            if isinstance(x, Matrix):
                _inputs.append(x.value)
            else:
                _inputs.append(x)
        return ufunc(*_inputs, **other)

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4})".format(self.__class__.__name__,
                                                np.dtype(self._dtype).name,
                                                str(self._shape),
                                                self._name,
                                                self._value)

    def __getitem__(self, index):
        return self._value[index]

    def _get_shape(self, shape):
        if not isinstance(shape, (tuple, list,)):
            raise TypeError("Wrong type for shape, use Tuple or List.")
        if not all(isinstance(x, (int, np.number, type(None),)) for x in shape):
            raise TypeError(
                "Wrong type for element in shape, use int or None.")
        return tuple([x if x is not None else 1 for x in shape])

    def __call__(self, value):
        if isinstance(value, np.ndarray):
            dims = len(self._shape) - len(value.shape)
            self._value = value
            if self._dtype != value.dtype:
                self._value = value.astype(self._dtype)
            if dims > 0:
                for _ in range(dims):
                    _axis = len(self._value.shape) - 1
                    self._value = np.expand_dims(self._value, axis=_axis)
            elif dims < 0:
                raise ValueError("Invalid number of dims: {0}, waitting for {1} dims."
                                 .format(value.ndim, len(value.shape)))
        elif isinstance(value, (int, float, decimal.Decimal, np.number,)):
            self._value = np.full(self._shape, value, self._dtype)
        self._shape = self._value.shape

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self._value.shape)

    @property
    def size(self):
        if self._value is None:
            return 0
        return self._value.size

    @property
    def value(self):
        return self._value

    @property
    def T(self):
        self._value = self._value.T
        self._shape = self._value.shape
        return self

    def flatten(self, order='c'):
        return self._value.flatten(order)

    def argmax(self, **other):
        return np.argmax(self._value, **other)
