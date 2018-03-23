from datetime import datetime
import numpy as np
import sys
from pre import *

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def rss(o, y):
    er = np.power(y - o, 2)
    return np.sum(er, axis=1)

def mse(o, y):
    # https://en.wikipedia.org/wiki/Mean_squared_error#Regression
    N = np.asarray([len(y_i) for y_i in y])
    return rss(o, y) / N


class RNN:
    def __init__(self, input_dim, hidden_dim=100, bptt_truncate=4):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        self.U = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (hidden_dim, input_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (input_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T, self.hidden_dim))
        # s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.input_dim))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:,t] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def _old_calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L
 
    def _old_calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self._calculate_total_loss(x,y)/N

    def calculate_loss(self, x, y):
        # For each sentence...
        # We just have one
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            diag = np.arange(len(y[i]))
            o_predictions = o[diag, diag]
            # Add to the loss based on how off we were
            # L += -1 * np.sum(np.log(o_predictions))
            er = np.power(y[i] - o_predictions, 2) # ToDo
            # np.sum(er)

            # https://en.wikipedia.org/wiki/Mean_squared_error#Regression
            N = len(y[i])
            return np.sum(er) / N

        # return L

        # N = np.sum((len(y_i) for y_i in y))
        # return self._calculate_total_loss(x,y)/N
        # return mse()

    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        diag = np.arange(len(y))
        delta_o[diag, diag] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)

            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step-1])              
                dLdU[:,x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]

    # Performs one step of SGD.
    def sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
     
    # Outer SGD Loop
    # - model: The RNN model instance
    # - X_train: The training data set
    # - y_train: The training data labels
    # - learning_rate: Initial learning rate for SGD
    # - nepoch: Number of times to iterate through the complete dataset
    # - evaluate_loss_after: Evaluate the loss after this many epochs
    def train_with_sgd(self, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
                # Adjust the learning rate if loss increases
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5 
                    print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                self.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1
        return losses

