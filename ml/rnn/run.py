import numpy as np
from pre import *
from rnn import RNN

np.random.seed(10)

model = RNN(len(X_train[0])) # TODO (jordycuan)
losses = model.train_with_sgd(X_train, y_train, nepoch=10, evaluate_loss_after=2)

def predict_prices(model):
    pass

