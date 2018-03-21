import numpy as np

from ml.network.cell import DropOutCell, Layer, LSTMCell, MultiRNNCell
from ml.network.embedding import Embedding
from ml.network.matrix import Matrix
from ml.ops import activation, array


class LSTM(object):

    def __init__(self,
                 input_prob=1.0,
                 output_prob=1.0,
                 state_prob=1.0,
                 do_dropout=True):
        super().__init__()
        self._input_prob = self._validate_prob(input_prob)
        self._output_prob = self._validate_prob(output_prob)
        self._state_prob = self._validate_prob(state_prob)
        self._do_dropout = do_dropout
        self._layer = None

    def _validate_prob(self, x):
        if not (0 < x <= 1.0):
            raise ValueError(
                "The probability needs be a value between 0 and 1")
        return x

    def inputs(self, inputs, labels):
        _inputs = Matrix(np.int32, (None, None), "inputs")
        _labels = Matrix(np.int32, (None, None), "labels")
        _inputs(inputs)
        _labels(labels)
        return _inputs, _labels

    def embedding(self, shape, inputs):
        params = np.random.uniform(-1, 1, shape)
        embedding = Embedding()
        return embedding.look_up(params, inputs)

    def layers(self, lstm_size, lstm_layers, batch_size, activation=activation.tanh):
        _cell = LSTMCell(lstm_size, activation=activation)
        d_cell = None
        if self._do_dropout:
            d_cell = DropOutCell(_cell, self._input_prob,
                                 self._output_prob, self._state_prob)
        else:
            d_cell = _cell
        self._multi_cell = MultiRNNCell(lstm_layers, d_cell)
        return self._multi_cell.zero_state(batch_size, np.dtype('Float32'))

    def fully_connected(self,
                        inputs,
                        output_units,
                        activation=activation.rectified_linear,
                        trainable=True):
        # TODO (hiigami) trainable implementation
        if self._layer is None:
            self._layer = Layer(output_units, activation=None)
        if not self._layer.built:
            self._layer.build(inputs.shape)
        outputs = self._layer.apply_layer(inputs)
        if activation is not None:
            outputs = activation(outputs)
        return outputs

    @staticmethod
    def create_network(cells, inputs, initial_state):
        # batch-major -> time-major
        _inputs = array.change_mayor(inputs)

        steps = _inputs.shape[0]
        # batch_size = _inputs.shape[1] # ???

        outputs = []
        states = []
        state = initial_state[0]  # ???

        current_step = 0
        while current_step < steps:
            output = _inputs[current_step].T
            if current_step >= steps:
                output = np.zeros(
                    (_inputs.shape[1], _inputs.shape[2]), dtype=_inputs.dtype)
                new_state = state
            else:
                for cell in cells:
                    output, new_state = cell(output.T, state)
                    state = new_state
            outputs.append(output)
            states.append(new_state)
            current_step = current_step + 1
        final_output = np.array(outputs, dtype=_inputs.dtype)
        # time-major -> batch-major
        final_output = array.change_mayor(final_output, time=False)
        return final_output, states

    def forward(self, embed, state):
        return self.create_network(self._multi_cell, embed, state)
