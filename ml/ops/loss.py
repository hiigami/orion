import numpy as np


def softmax(z):
    # https://en.wikipedia.org/wiki/Softmax_function
    logits_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return logits_exp / np.sum(logits_exp, axis=1, keepdims=True)


def cross_entropy(p, q, estimate=False):
    # https://en.wikipedia.org/wiki/Cross_entropy
    event_prob = np.log(q)
    p_log_q = np.sum((p * event_prob), 1)
    if estimate:
        p_log_q = p_log_q / p.size
    return -1.0 * p_log_q


def softmax_cross_entropy(labels, logits, estimate=True):
    return cross_entropy(labels, softmax(logits), estimate)
