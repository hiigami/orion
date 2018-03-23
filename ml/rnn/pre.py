# https://github.com/kimanalytics/Recurrent-Neural-Network-to-Predict-Stock-Prices/blob/master/Recurrent%20Neural%20Network%20to%20Predict%20Tesla%20Stock%20Prices.ipynb

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reading CSV file into training set
tesla_data = pd.read_csv('ml/rnn/TSLA.csv')

# Getting relevant feature
tesla_data = tesla_data.iloc[:,1:2]

tesla_data = tesla_data.values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(tesla_data[:-200])
test_set = tesla_data[-200:] # Not necesary to scale

# TEST
training_set = np.asarray([0.1, 0.5, 0.9, 0.9])

# Getting the inputs and the ouputs
X_train = training_set[:len(training_set) - 1]
y_train = training_set[1:len(training_set)]

X_train = X_train.reshape((1, X_train.shape[0]))
y_train = y_train.reshape((1, y_train.shape[0]))

# We feed the rnn with the last "known" 100 days
test_begin = tesla_data[-300:-200]
