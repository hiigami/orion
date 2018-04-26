import copy

import numpy as np

from ml.network.matrix import InitFn, Matrix
from ml.network.state import State
from ml.ops import array, loss, training
from ml.ops.activation import Activation, Sigmoid

# TODO (hiigami) change weight_init_fn to truncate normal
WEIGHT_INIT_FN = InitFn(np.random.uniform, low=0.0, high=1.0, size=None)


class Layer(object):
    __slots__ = (
        "_num_units",
        "_activation",
        "_weights_init_fn",
        "_name",
        "_weights",
        "_bias",
        "built"
    )

    def __init__(self,
                 num_units,
                 activation=None,
                 weights_init_fn=np.ones,
                 name=None):
        self._num_units = num_units
        self._activation = activation
        self._weights_init_fn = weights_init_fn
        self._name = name
        self.built = False

    def _get_gradient_fn(self, activation):
        return Activation.get_derivative(activation)

    def _apply_gradient(self, error, gradient_output, output_min_1):
        o = array.broadcast_attempt(gradient_output, output_min_1)
        return np.dot(error * gradient_output, o)

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs.shape)
        return self.apply_layer(inputs)

    @property
    def size(self):
        return self._num_units

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, activation):
        self._activation = activation

    def update_layer(self, w, b):
        _w = self._weights - w[:1, :].T  # .astype(self._weights.dtype)
        self._weights(_w)
        _b = self._bias - b[:1, :].T  # .astype(self._weights.dtype)
        self._bias(_b)

    def build(self, shape):
        self._weights = Matrix(np.dtype('Float32'),
                               (shape[-1], self._num_units,),
                               "WEIGHTS",
                               init_fn=self._weights_init_fn)
        self._bias = Matrix(np.dtype('Float32'),
                            (self._num_units, 1,),
                            "BIAS",
                            init_fn=np.zeros)
        self.built = True

    def apply_layer(self, inputs):
        # region logits
        output = np.dot(inputs, self._weights) + self._bias
        bias_broadcastable = array.broadcast_attempt(output, self._bias)
        output = output + bias_broadcastable
        # endregion
        if self._activation is not None:
            return self._activation(output.astype(inputs.dtype))
        return output

    def back(self, error, output, output_min_1):
        # region logits
        _output = np.dot(output, self._weights)
        bias_broadcastable = array.broadcast_attempt(_output, self._bias)
        _output = _output + bias_broadcastable
        # endregion
        if self._activation is not None:
            gradient_fn = self._get_gradient_fn(self._activation)
            _output = gradient_fn(_output)
        d_b = error * _output
        d_w = d_b[:, :self._weights.shape[0]] * output
        return self._apply_gradient(error, _output, output_min_1), d_w, d_b


class LSTMCell(Layer):
    # See http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    __slots__ = (
        "_num_units",
        "_forget_bias",
        "_activation",
        "_weights_init_fn",
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
                 activation=None,
                 weights_init_fn=WEIGHT_INIT_FN,
                 name=None):
        super().__init__(num_units, activation, weights_init_fn, name)
        self._forget_bias = forget_bias

    def __deep_copy__(self, memo):
        item = LSTMCell(copy.deepcopy(self._num_units, memo))
        for k in self.__slots__:
            if k != "_num_units":
                val = getattr(self, k)
                setattr(item, k, copy.copy(val))
        return item

    def _apply_gradient(self, error, gradient_output, output_min_1):
        return error * gradient_output * output_min_1

    def __call__(self, inputs, state):
        if not self.built:
            self.build(inputs.shape)
        return self.apply_layer(inputs, state)

    def zero_state(self, batch_size, dtype):
        h = np.zeros((batch_size, self.size), dtype=dtype)
        c = np.zeros((batch_size, self.size), dtype=dtype)
        _c = Matrix(h.dtype, h.shape, "c")
        _c(c)
        _h = Matrix(h.dtype, h.shape, "h")
        _h(h)
        return State(_c, _h)

    def build(self, shape):
        forget_bias_init_fn = InitFn(np.full, fill_value=self._forget_bias)
        bias_init_fn = self._weights_init_fn  # np.zeros
        weights_shape = (shape[1], self._num_units,)
        #(shape[1] + self._num_units, self._num_units)
        bias_shape = (shape[1], 1)  # (self._num_units, 1,)
        self._weights_i = Matrix(np.dtype('Float32'),
                                 weights_shape,
                                 "WEIGHTS_INPUT",
                                 init_fn=self._weights_init_fn)
        self._weights_f = Matrix(np.dtype('Float32'),
                                 weights_shape,
                                 "WEIGHTS_FORGET",
                                 init_fn=self._weights_init_fn)
        self._weights_c = Matrix(np.dtype('Float32'),
                                 weights_shape,
                                 "WEIGHTS_CANDIDATE",
                                 init_fn=self._weights_init_fn)
        self._weights_o = Matrix(np.dtype('Float32'),
                                 weights_shape,
                                 "WEIGHTS_OUTPUT",
                                 init_fn=self._weights_init_fn)
        self._bias_i = Matrix(np.dtype('Float32'),
                              bias_shape, "BIAS_INPUT", init_fn=bias_init_fn)
        self._bias_f = Matrix(np.dtype('Float32'), bias_shape,
                              "BIAS_FORGET", init_fn=forget_bias_init_fn)
        self._bias_c = Matrix(np.dtype('Float32'), bias_shape,
                              "BIAS_CANDIDATE", init_fn=bias_init_fn)
        self._bias_o = Matrix(np.dtype('Float32'), bias_shape,
                              "BIAS_OUTPUT", init_fn=bias_init_fn)
        self.built = True

    def _gate(self, w, i, s_h, b):
        h_w = np.multiply(w, s_h)
        i_w = np.multiply(w.T, i)
        return (np.add(np.add(h_w, i_w.T), b)).astype(w)

    def apply_layer(self, inputs, state):
        # TODO (hiigami) Implement variants???
        _sigmoid = Sigmoid.logistic
        _activation = self._activation

        if not isinstance(inputs, Matrix):
            _inputs = Matrix(inputs.dtype, inputs.shape, "")
            _inputs(inputs)

        forget_gate = _sigmoid(self._gate(
            self._weights_f, _inputs, state.h, self._bias_f))
        input_gate = _sigmoid(self._gate(
            self._weights_i, _inputs, state.h, self._bias_i))
        candidate_gate = _activation(self._gate(
            self._weights_c, _inputs, state.h, self._bias_c))
        output_gate = _sigmoid(self._gate(
            self._weights_o, _inputs, state.h, self._bias_o))

        c = (forget_gate * state.c) + (input_gate * candidate_gate)
        h = np.multiply(_activation(c), output_gate)
        return h, State(c, h)

    def back(self, error, output, output_min_1, state):

        d_sigmoid = self._get_gradient_fn(Sigmoid.logistic)
        d_activation = self._get_gradient_fn(self._activation)
        d_w = []
        d_b = []

        if not isinstance(output, Matrix):
            _inputs = Matrix(output.dtype, output.shape, "")
            _inputs(output)

        forget_gate = d_sigmoid(self._gate(
            self._weights_f, _inputs, state.h, self._bias_f))
        input_gate = d_sigmoid(self._gate(
            self._weights_i, _inputs, state.h, self._bias_i))
        candidate_gate = d_activation(self._gate(
            self._weights_c, _inputs, state.h, self._bias_c))
        output_gate = d_sigmoid(self._gate(
            self._weights_o, _inputs, state.h, self._bias_o))

        c = (forget_gate * state.c) + (input_gate * candidate_gate)
        h = np.multiply(d_activation(c), output_gate)
        g_h = self._apply_gradient(error, h, output_min_1)

        d_b.append(error * forget_gate)
        d_b.append(error * input_gate)
        d_b.append(error * candidate_gate)
        d_b.append(error * output_gate)

        d_w.append(d_b[0] * output.T)
        d_w.append(d_b[1] * output.T)
        d_w.append(d_b[2] * output.T)
        d_w.append(d_b[3] * output.T)

        return g_h, State(c, g_h), np.array(d_w), np.array(d_b)

    def update_layer(self, w, b):
        self._weights_f(self._weights_f - w[0])
        self._weights_i(self._weights_i - w[1])
        self._weights_c(self._weights_c - w[2])
        self._weights_o(self._weights_o - w[3])

        self._bias_f(self._bias_f - b[0])
        self._bias_i(self._bias_i - b[1])
        self._bias_c(self._bias_c - b[2])
        self._bias_o(self._bias_o - b[3])


class DropOutCell(object):
    # See https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
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

    @property
    def activation(self):
        return self._cell.activation

    @staticmethod
    def should_dropout(p):
        return (not isinstance(p, float)) or p < 1.0

    def _dropout(self, x, keep_prob):
        if not (0 < keep_prob <= 1.0):
            raise ValueError(
                "The probability needs be a value between 0 and 1")
        if keep_prob == 1.0:
            return x
        return (np.divide(x, keep_prob) *
                (np.random.uniform(size=x.shape) < keep_prob))

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

    def back(self, error, output, output_min_1, state):
        if state is None:
            self._cell.back(error, output, output_min_1)
        return self._cell.back(error, output, output_min_1, state)

    def update_layer(self, w, b):
        self._cell.update_layer(w, b)

    def update_props(self, input_prob, output_prob, state_prob):
        self._input_prob = input_prob
        self._output_prob = output_prob
        self._state_prob = state_prob


class MultiRNNCell(object):
    def __init__(self, layers, cell=None):
        self._cells = []
        if cell is not None:
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

    def __reversed__(self):
        return reversed(self._cells)

    def zero_state(self, batch_size, dtype):
        return np.array([cell.zero_state(batch_size, dtype)
                         for cell in self._cells])
