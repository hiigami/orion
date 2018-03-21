import collections
import multiprocessing
from collections import Counter, deque
from string import punctuation

import numpy as np

from utils.logger import getLogger

logger = getLogger(__name__)


def get_batches(x, y, batch_size=100):
    n_batches = len(x) // batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for i in range(0, len(x), batch_size):
        yield x[i:i+batch_size], y[i:i+batch_size]


class Data(object):
    def __init__(self):
        self._labels = None
        self.train_data_x = None
        self.train_data_y = None
        self.val_x = None
        self.val_y = None
        self.test_x = None
        self.test_y = None
        self._int_map = deque()

    def label_encoding(self, labels, steps=200):
        self._labels = np.array(
            [1 if l == "positive" else 0 for l in labels.split()])

        self._int_map = [r[0:steps] for r in self._int_map if len(r) > 0]

        review_lens = Counter([len(x) for x in self._int_map])

        logger.info("Zero-length reviews: %s", review_lens[0])
        logger.info("Maximum review length: %s", max(review_lens))

    def separate_data(self, seq_len=200, split_frac_training=0.8, split_frac_test_val=0.5):

        features = np.zeros((len(self._int_map), seq_len), dtype=np.int)
        for i, row in enumerate(self._int_map):
            features[i, -len(row):] = np.array(row)[:seq_len]

        split_index = int(split_frac_training * len(features))

        self._train_x, self._val_x = features[:
                                              split_index], features[split_index:]
        self._train_y, self._val_y = self._labels[:
                                                  split_index], self._labels[split_index:]

        split_index = int(split_frac_test_val * len(self._val_x))

        self._val_x, self._test_x = self._val_x[:
                                                split_index], self._val_x[split_index:]
        self._val_y, self._test_y = self._val_y[:
                                                split_index], self._val_y[split_index:]

        logger.info("--- Feature Shapes ---")
        logger.info("Train set: {}".format(self._train_x.shape))
        logger.info("Validation set: {}".format(self._val_x.shape))
        logger.info("Test set: {}".format(self._test_x.shape))
        logger.info("label set: {}".format(self._train_y.shape))
        logger.info("Validation label set: {}".format(self._val_y.shape))
        logger.info("Test label set: {}".format(self._test_y.shape))

    @property
    def training_data(self):
        return self._train_x, self._train_y

    @property
    def test_data(self):
        return self._test_x, self._test_y

    @property
    def validation_data(self):
        return self._val_x, self._val_y


class Data2(Data):
    def __init__(self):
        super().__init__()
        self.n_words = 0
        self.words_to_int = {}

    def no_punctuation(self, text):
        # TODO (hiigami) add emojis interpretion.
        return ''.join([c for c in text if c not in punctuation])

    def _map_to_integers(self, sentences, words):
        counts = Counter(words)
        vocab = sorted(counts, key=counts.get, reverse=True)
        self.words_to_int = {word: i for i, word in enumerate(vocab, 1)}

        for each in sentences:
            self._int_map.append([self.words_to_int[word]
                                  for word in each.split()])

        self.n_words = len(self.words_to_int) + 1  # Add 1 for 0 added to vocab

    def preprocessing(self, data):
        all_text = ''.join(list(map(self.no_punctuation, data)))
        data = all_text.split('\n')

        all_text = ' '.join(data)
        words = all_text.split()
        self._map_to_integers(data, words)
