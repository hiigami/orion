import numpy as np
from ml.ops import array
from ml.ops.activation import Activation


class Adam(object):
    # https://arxiv.org/pdf/1412.6980.pdf
    def __init__(self,
                 alpha=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=np.spacing(1e-8)):
        self._alpha = alpha
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

    def _update_moment(self, decay_rate, moment, gradient):
        w = decay_rate * moment + (1 - decay_rate)
        if gradient.ndim == 3:
            return np.array([w * g for g in gradient])
        return w * gradient

    def _correct_moment(self, decay_rate, moment):
        return moment / (1 - decay_rate)

    def _update_parameter(self, cell, unit_m_w,  unit_v_w, unit_m_b, unit_v_b):
        l_w = self._alpha * unit_m_w / \
            (np.sqrt(unit_v_w.astype(np.float64)) + self._epsilon)
        l_b = self._alpha * unit_m_b / \
            (np.sqrt(unit_v_b.astype(np.float64)) + self._epsilon)
        cell.update_layer(l_w, l_b)

    def minimize(self,
                 error,
                 output,
                 output_1,
                 cell,
                 state=None,
                 m=0.0,
                 v=0.0):
        _shape = output.shape
        if state is not None:
            _shape = state.h.shape

        d_w_prev = np.zeros(_shape, dtype=output.dtype)
        d_w = np.full(_shape, 1.0, dtype=output.dtype)

        o, s, g = None, None, None
        index = 0
        while not np.allclose(d_w, d_w_prev, equal_nan=True):
            d_w_prev = d_w
            if state is None:
                g = cell.back(error, output, output_1)
            else:
                g = cell.back(error, output, output_1, state)
            if len(g) == 4:
                o, s, d_w, d_b = g
            else:
                o, d_w, d_b = g

            if d_w_prev.ndim != d_w.ndim:
                d_w_prev = np.zeros(d_w.shape, dtype=d_w.dtype)
            else:
                d_w_prev = d_w
            m_w = self._update_moment(self._beta1, m, d_w)
            m_b = self._update_moment(self._beta1, m, d_b)

            v_w = self._update_moment(self._beta2, v, np.power(d_w, 2))
            v_b = self._update_moment(self._beta2, v, np.power(d_b, 2))

            unit_m_w = self._correct_moment(self._beta1, m_w)
            unit_m_b = self._correct_moment(self._beta1, m_b)

            unit_v_w = self._correct_moment(self._beta1, v_w)
            unit_v_b = self._correct_moment(self._beta1, v_b)

            self._update_parameter(cell,
                                   unit_m_w,
                                   unit_v_w,
                                   unit_m_b,
                                   unit_v_b)
            if index > 10:  # ???
                break
            index = index + 1
        return m, v, o, s
