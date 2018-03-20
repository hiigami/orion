import numpy as np
from pre import *
from nn import RNNNumpy

np.random.seed(10)

o, s = model.forward_propagation(X_train[1])

predictions = model.predict(X_train[1])

model = RNNNumpy(len(X_train)) # TODO (jordycuan)
losses = model.train_with_sgd(X_train, y_train, nepoch=10, evaluate_loss_after=2)

def predict_prices(model):
    pass

