import copy

import numpy as np

from ml.network.matrix import Matrix
from ml.network.state import State
from ml.ops import activation, array, loss, training


class Layer(object):

    def __init__(self,
                 num_units,
                 activation=None,
                 name=None):
        self._num_units = num_units
        self._activation = activation
        self._name = name
        self.built = False

    @property
    def size(self):
        return self._num_units

    def build(self, shape):
        init_fn = np.zeros
        self._weights = Matrix(np.dtype('Float32'),
                               (shape[-1], self._num_units,),
                               "WEIGHTS",
                               init_fn=init_fn)
        self._bias = Matrix(np.dtype('Float32'),
                            (self._num_units,),
                            "BIAS",
                            init_fn=init_fn)
        self.built = True

    def apply_layer(self, inputs):
        # region logits
        output = np.add(
            np.dot(inputs,
                   self._weights),
            self._bias
        )
        # endregion
        if self._activation is not None:
            return self._activation(output)
        return output


class LSTMCell(object):
    __slots__ = (
        "_num_units",
        "_forget_bias",
        "_activation",
        "_name",
        "_weights_i",
        "_weights_f",
        "_weights_c",
        "_weights_o",
        "_bias_i",
        "_bias_f",
        "_bias_c",
        "_bias_o",
        "built"
    )

    def __init__(self,
                 num_units,
                 forget_bias=1.0,
                 activation=None,  # activation.tanh or put validation in apply_layer
                 name=None):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        self._name = name
        self.built = False

    def __deep_copy__(self, memo):
        item = LSTMCell(copy.deepcopy(self._num_units, memo))
        for k in self.__slots__:
            if k != "_num_units":
                val = getattr(self, k)
                setattr(item, k, copy.copy(val))
        return item

    @property
    def size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        h = np.zeros((batch_size, self.size), dtype=dtype)
        c = np.zeros((batch_size, self.size), dtype=dtype)
        return State(c, h)

    def build(self, shape):
        init_fn = np.zeros
        weights_shape = (shape[0] + self._num_units, self._num_units)
        bias_shape = (self._num_units,)
        self._weights_i = Matrix(
            np.dtype('Float32'), weights_shape, "WEIGHTS_INPUT", init_fn=init_fn)
        self._weights_f = Matrix(
            np.dtype('Float32'), weights_shape, "WEIGHTS_FORGET", init_fn=init_fn)
        self._weights_c = Matrix(
            np.dtype('Float32'), weights_shape, "WEIGHTS_CANDIDATE", init_fn=init_fn)
        self._weights_o = Matrix(
            np.dtype('Float32'), weights_shape, "WEIGHTS_OUTPUT", init_fn=init_fn)
        self._bias_i = Matrix(np.dtype('Float32'),
                              bias_shape, "BIAS_INPUT", init_fn=init_fn)
        self._bias_f = Matrix(np.dtype('Float32'), bias_shape,
                              "BIAS_FORGET", init_fn=init_fn)
        self._bias_c = Matrix(np.dtype('Float32'), bias_shape,
                              "BIAS_CANDIDATE", init_fn=init_fn)
        self._bias_o = Matrix(np.dtype('Float32'), bias_shape,
                              "BIAS_OUTPUT", init_fn=init_fn)
        self.built = True

    def apply_layer(self, inputs, state):
        # TODO (hiigami) Implement variants???
        # http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        _inputs_plus_h = np.concatenate((state.h, inputs.T), axis=1)
        inputs_plus_h = Matrix(
            dtype=inputs.dtype, shape=_inputs_plus_h.shape, name="inputs_plus_h")
        inputs_plus_h(_inputs_plus_h)

        forget_gate = activation.sigmoid(
            (np.dot(inputs_plus_h, self._weights_i) +
             self._forget_bias).astype(self._weights_i.dtype)
        )
        input_gate = activation.sigmoid(
            np.add(
                np.dot(inputs_plus_h,
                       self._weights_i),
                self._bias_i
            )
        )
        candidate_gate = self._activation(
            np.add(
                np.dot(inputs_plus_h,
                       self._weights_c),
                self._bias_c
            )
        )
        output_gate = activation.sigmoid(
            np.add(
                np.dot(inputs_plus_h,
                       self._weights_o),
                self._bias_o
            )
        )
        c = forget_gate * np.add(state.c, input_gate) * candidate_gate
        h = np.multiply(self._activation(c), output_gate)
        return h, State(c, h)

    def __call__(self, inputs, state):
        if not self.built:
            self.build(inputs.shape)
        return self.apply_layer(inputs, state)


class DropOutCell(object):
    # https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
    __slots__ = (
        "_cell",
        "_input_prob",
        "_output_prob",
        "_state_prob"
    )

    def __init__(self,
                 cell,
                 input_prob=1.0,
                 output_prob=1.0,
                 state_prob=1.0):
        self._cell = cell
        self._input_prob = input_prob
        self._output_prob = output_prob
        self._state_prob = state_prob

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    @property
    def size(self):
        return self._cell.size

    @staticmethod
    def should_dropout(p):
        return (not isinstance(p, float)) or p < 1.0

    def _dropout(self, x, keep_prob):
        if not (0 < x <= 1.0):
            raise ValueError(
                "The probability needs be a value between 0 and 1")
        if keep_prob == 1.0:
            return x
        return np.divide(x, keep_prob) * np.random.uniform(size=x.shape)

    def __call__(self, inputs, state):
        if not self._cell.built:
            self._cell.build(inputs.shape)
        if self.should_dropout(self._input_prob):
            inputs = self._dropout(inputs, self._input_prob)
        output, new_state = self._cell.apply_layer(inputs, state)
        if self.should_dropout(self._state_prob):
            new_state.h = self._dropout(new_state.h, self._state_prob)
        if self.should_dropout(self._output_prob):
            output = self._dropout(output, self._output_prob)
        return output, new_state

    def __deep_copy__(self, memo):
        item = DropOutCell(copy.deepcopy(self._cell, memo))
        for k in self.__slots__:
            if k != "_cell":
                val = getattr(self, k)
                setattr(item, k, copy.copy(val))
        return item


class MultiRNNCell(object):
    def __init__(self, layers, cell):
        self._layers = layers
        self._cells = [copy.deepcopy(cell)
                       for _ in range(layers)]

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        try:
            cell = self._cells[self._index]
        except IndexError:
            raise StopIteration
        self._index += 1
        return cell

    def __len__(self):
        return len(self._cells)

    def zero_state(self, batch_size, dtype):
        return np.array([cell.zero_state(batch_size, dtype)
                         for cell in self._cells])
