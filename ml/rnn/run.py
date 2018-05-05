import numpy as np

from pre import X_train, y_train
from rnn import RNN

np.random.seed(10)

# 19 sets de 10 días con 4 features
new_x = X_train[:190, :].reshape((19, 10, 4))
new_y = y_train[:190, :].reshape((19, 10, 4))

train_batch = int(19 * 0.8)
validate_batch = int(19 * 0.5)

model = RNN(10, 4)

model.train_with_sgd(new_x,
                     new_y,
                     train_batch,
                     validate_batch,
                     10,
                     nepoch=10)
