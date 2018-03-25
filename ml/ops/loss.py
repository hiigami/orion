import numpy as np

from ml.ops.activation import Softmax


def cross_entropy(p, q, estimate=False):
    # https://en.wikipedia.org/wiki/Cross_entropy
    event_prob = np.log(q)
    p_log_q = np.sum((p * event_prob), 1)
    if estimate:
        p_log_q = p_log_q / p.size
    return -1.0 * p_log_q


def softmax_cross_entropy(labels, logits):
    return cross_entropy(labels, Softmax.probability(logits), True)
