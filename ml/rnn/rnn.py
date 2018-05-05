import sys
from datetime import datetime

import numpy as np


def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)


def softmax_dev(z):
    z = z.astype(np.dtype('Float64'))
    logits_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    div = np.sum(logits_exp, axis=1, keepdims=True)
    logits_exp_1 = logits_exp[:, 0]
    logits_exp_2 = logits_exp[:, 1:]
    logits_exp_2 = np.sum(logits_exp_2, axis=1, keepdims=True)
    return (logits_exp_1 * logits_exp_2) / np.square(div)


def rss(o, y):
    er = np.power(y - o, 2)
    return np.sum(er, axis=1)


def mse(o, y):
    # https://en.wikipedia.org/wiki/Mean_squared_error#Regression
    N = np.asarray([len(y_i) for y_i in y])
    return rss(o, y) / N


class RNN:
    def __init__(self, batch_size, input_dim):

        self.input_dim = input_dim

        self.b = np.random.uniform(0, 1.0, (1, input_dim,))
        self.w = np.random.uniform(0, 1.0, (batch_size, input_dim,))

        self.b2 = np.random.uniform(0, 1.0, (1, input_dim,))
        self.w2 = np.random.uniform(0, 1.0, (batch_size, input_dim,))
    
    def calculate_loss(self, x, y):
        return np.square(x - y).mean()

    def zero_state(self, batch_size):
        return np.zeros((batch_size, self.input_dim,), dtype=np.dtype('Float32'))

    def forward(self, inputs, h_min_1):
        aa = np.add(
            np.multiply(self.w, inputs),
            np.multiply(self.w, h_min_1)
        )
        bb = np.add(
            aa,
            self.b
        )
        output = np.tanh(bb)
        return output, output

    def forward_propagation(self, inputs, initial_state):
        return self.forward(inputs, initial_state)

    def bptt(self, output, h, error):
        aa = np.add(
            np.multiply(self.w, output),
            np.multiply(self.w, h)
        )
        bb = np.add(
            aa,
            self.b
        )
        _output = np.tanh(bb)
        d_b = error * _output
        d_w = d_b * output
        return (_output, d_w, d_b)

    def sgd_step(self, x, state, cost, learning_rate):
        output = softmax_dev(
            np.add(np.multiply(self.w2, x), self.b2))
        d_b2 = cost * output[:,:self.b2.shape[1]]
        d_w2 = d_b2[:,:self.w2.shape[1]] * x
        self.w2 -= learning_rate * d_w2
        self.b2 -= learning_rate * d_b2[:1, :]

        o, d_w, d_b = self.bptt(output[:,:self.b2.shape[1]], state, cost)
        self.w -= learning_rate * d_w
        self.b -= learning_rate * d_b[:1, :]

    def validation(self, labels, predictions):
        correct_prediction = (np.argmax(predictions, 1) == np.argmax(
            labels, 1)).astype(np.dtype('Float32'))
        return np.mean(correct_prediction)

    def validate(self, num_batches, batch_size, inputs, labels):
        state = self.zero_state(batch_size)
        accuracy = []
        total = len(inputs) - 1
        for index in range(num_batches):
            ii = total - index
            inputs_batch = inputs[ii]
            labels_batch = labels[ii]
            outputs, final_states = self.forward_propagation(inputs_batch, state)
            final_output = softmax(
                    np.add(np.multiply(self.w2, outputs), self.b2))
            accuracy.append(self.validation(labels_batch, final_output))
        return np.mean(accuracy)

    def train_with_sgd(self,
                       inputs,
                       labels,
                       num_batches,
                       num_batches_validate,
                       batch_size,
                       learning_rate=0.005,
                       nepoch=100):
        steps_count = 0
        for epoch in range(nepoch):
            state = self.zero_state(batch_size)
            for index in range(num_batches):
                inputs_batch = inputs[index]
                labels_batch = labels[index]
                outputs, final_states = self.forward_propagation(inputs_batch, state)
                final_output = softmax(
                    np.add(np.multiply(self.w2, outputs), self.b2))
                cost = self.calculate_loss(labels_batch, final_output)

                self.sgd_step(final_output, final_states, cost, learning_rate)
                steps_count = steps_count + 1
                if steps_count % 5 == 0:
                    print("steps_count: ", steps_count)
                    print("cost: ", cost)
                    if steps_count % 25 == 0:
                        accuracy = self.validate(num_batches_validate,
                                                 batch_size,
                                                 inputs,
                                                 labels)
                        print("--------------------")
                        print("accuracy: ", accuracy)
                        print("--------------------")
        # ---- Example    -----
        # Apply new model and get prediction
        o, s = self.forward_propagation(inputs[1], self.zero_state(batch_size))
        print([o[i, x] for i, x in enumerate(np.argmax(o, 1).astype(np.int32))])
        # ---- Example End -----
