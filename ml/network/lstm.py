import numpy as np

from ml.network.cell import DropOutCell, Layer, LSTMCell, MultiRNNCell
from ml.network.embedding import Embedding
from ml.network.matrix import Matrix
from ml.ops import array
from ml.ops.activation import Hyperbolic


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

    def training(self, value):
        input_prob = 1.0
        output_prob = 1.0
        state_prob = 1.0
        if not value:
            input_prob = self._input_prob
            output_prob = self._output_prob
            state_prob = self._state_prob

        for cell in self._multi_cell:
            if isinstance(cell, DropOutCell):
                cell.update_props(input_prob, output_prob, state_prob)

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

    def layers(self, lstm_size, lstm_layers, batch_size, activation=Hyperbolic.tanh):
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
                        activation=Hyperbolic.tanh):
        if self._layer is None:
            self._layer = Layer(output_units, activation=activation)
        if not self._layer.built:
            self._layer.build(inputs.shape)
        self._layer.activation=None
        outputs = self._layer.apply_layer(inputs)
        self._layer.activation=activation
        return outputs

    @staticmethod
    def create_network(cells, inputs, initial_state):
        # batch-major -> time-major
        _inputs = array.change_mayor(inputs)
        steps = _inputs.shape[0]
        # batch_size = _inputs.shape[1] # ???

        outputs = []
        states = []
        state = initial_state

        current_step = 0
        while current_step < steps:
            output = _inputs[current_step]
            if current_step >= steps:
                output = np.zeros((_inputs.shape[1], _inputs.shape[2]),
                                  dtype=_inputs.dtype)
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
        final_output = array.change_mayor(final_output)
        return final_output, states

    def forward(self, embed, state):
        return self.create_network(self._multi_cell, embed, state)

    def back(self, optimizer, loss, outputs, predictions, states, embed):
        _outputs = array.change_mayor(outputs)
        step = _outputs.shape[0] - 1

        m, v, o, s = optimizer.minimize(loss,
                                        predictions,
                                        outputs[:, -1],
                                        self._layer)
        #m=0.0
        #v=0.0
        while step >= 0:
            output = _outputs[step]
            state = states[step]
            if step > 0:
                _outputs_min_1 = _outputs[step - 1]
            else:
                _inputs = array.change_mayor(embed)
                _outputs_min_1 = _inputs[0]
            for cell in reversed(self._multi_cell):
                m, v, output, state = optimizer.minimize(loss,
                                                         output.T,
                                                         _outputs_min_1,
                                                         cell,
                                                         state=state,
                                                         m=m,
                                                         v=v)
            step = step - 1
