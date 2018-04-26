import pickle
import time
import copy
import numpy as np

from ml.data.core import Data2, get_batches
from ml.network.lstm import LSTM
from ml.ops.activation import Softmax
from ml.ops import loss, training
from utils.addfunc import add_class_method
from utils.logger import getLogger

logger = getLogger(__name__)
SAVE = False

embed_size = 256
lstm_size = 256
lstm_layers = 2
batch_size = 1000
learning_rate = 0.001
keep_prob = 0.2
epochs = 10


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


if SAVE:
    with open('./reviews.txt', 'r') as f:
        reviews = f.read()

    a = Data2()
    a.preprocessing(reviews)
    reviews = None

    with open('./labels.txt', 'r') as f:
        labels_org = f.read()

    review_lens = a.label_encoding(labels_org, steps=lstm_size)
    _max_review_lens = max(review_lens)
    logger.info("Zero-length reviews: %s", review_lens[0])
    logger.info("Maximum review length: %s", _max_review_lens)
    labels_org = None
    a.separate_data(seq_len=_max_review_lens)

    xt, yt = a.training_data
    n_words = a.n_words
    x_val, y_val = a.validation_data
    save_object(xt, 't1.pkl')
    save_object(yt, 't2.pkl')
    save_object(n_words, 't3.pkl')
    save_object(x_val, 't4.pkl')
    save_object(y_val, 't5.pkl')
else:
    xt, yt, x_val, y_val = None, None, None, None
    with open('t1.pkl', 'rb') as input:
        xt = pickle.load(input)
    with open('t2.pkl', 'rb') as input:
        yt = pickle.load(input)
    with open('t3.pkl', 'rb') as input:
        n_words = pickle.load(input)
    with open('t4.pkl', 'rb') as input:
        x_val = pickle.load(input)
    with open('t5.pkl', 'rb') as input:
        y_val = pickle.load(input)


@add_class_method(LSTM)
def predictions(self, outputs):
    _prediction = self.fully_connected(outputs[:, -1],
                                       1,
                                       activation=Softmax.probability)
    # TODO (hiigami) add missing histogram value
    return _prediction.astype(np.dtype('Float32'))


@add_class_method(LSTM)
def cost(self, labels, predictions):
    _cost = np.mean(loss.softmax_cross_entropy(labels, predictions))
    # TODO (hiigami) add missing histogram value
    return _cost


@add_class_method(LSTM)
def optimizer(self, learning_rate):
    return training.Adam(learning_rate)


@add_class_method(LSTM)
def validation(self, labels, predictions):
    correct_prediction = (np.argmax(predictions, 1).astype(
        np.int32) == np.argmax(labels, 1)).astype(np.dtype('Float32'))
    return np.mean(correct_prediction)


lstm = LSTM(
    input_prob=keep_prob,
    output_prob=keep_prob,
    state_prob=keep_prob
)


def validate(state):
    accuracy = []
    lstm.training(True)
    for xv, yv in get_batches(x_val, y_val, batch_size=batch_size):
        inputs, labels = lstm.inputs(xv, yv)
        embed = lstm.embedding((n_words, embed_size), inputs)
        outputs, final_states = lstm.forward(embed, state)
        predictions = lstm.predictions(outputs)  # pylint: disable=no-member
        predictions = Softmax.probability(predictions)
        accuracy.append(lstm.validation(labels.T, predictions))  # pylint: disable=no-member
    lstm.training(False)
    return np.mean(accuracy)


def train():
    initial_state = lstm.layers(lstm_size, lstm_layers, batch_size)
    total_batch_step = len(xt)
    start_time = time.time()
    index = 0
    for epoch in range(epochs):
        state = initial_state[-1]
        for (x, y) in get_batches(xt, yt, batch_size=batch_size):
            inputs, labels = lstm.inputs(x, y)
            embed = lstm.embedding((n_words, embed_size), inputs)
            outputs, final_states = lstm.forward(embed, state)
            predictions = lstm.predictions(outputs)  # pylint: disable=no-member
            cost = lstm.cost(labels, predictions)  # pylint: disable=no-member
            optimizer = lstm.optimizer(learning_rate)  # pylint: disable=no-member
            predictions = Softmax.probability(predictions)
            lstm.back(optimizer, cost, outputs, predictions, final_states, embed)
            # TODO (hiigami) add partial save method.
            index = index + 1
            if index % 5 == 0:
                elapsed_time = round(time.time() - start_time)
                minutes = elapsed_time // 60
                seconds = elapsed_time % 60
                logger.info("\nEpoch: %s/%s batch_step: %s/%s\nloss: %s"
                            "\nTime %s:%s",
                            epoch, epochs, index, total_batch_step, cost,
                            minutes, seconds)
                if index % 25 == 0:
                    accuracy = validate(initial_state[-1])
                    logger.info("accuracy: %s", accuracy)
    # TODO (hiigami) add model save method.


def test():
    # TODO (hiigami) complete method.
    pass


train()
