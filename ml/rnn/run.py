import numpy as np
from pre import batch_size, X_train, y_train
from rnn import RNN

np.random.seed(10)

model = RNN(batch_size)
o, s = model.forward_propagation(X_train[10])

losses = model.train_with_sgd(X_train, y_train, nepoch=10, evaluate_loss_after=2)

def predict_prices(model):
    pass


