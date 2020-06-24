# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:07:30 2020

@author: unibl
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Part 1 Data Preprocessing
# Importing Training Data-Sets
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values # Only Open stock price (one column)
# Feature Scaling 
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0 ,1))
training_set_scaled = sc.fit_transform(training_set)

#Creating the data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range (60, 1258):
    X_train.append(training_set_scaled[i -60:i, 0]) # 60 previous stock prices
    y_train.append(training_set_scaled[i, 0]) # next stock price 
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
#Initilizing RNN
regressor = Sequential()

# Adding LSTM layer and some dropout regularization
regressor.add(LSTM(units = 52, return_sequences = True, input_shape = ( X_train.shape[1], 1))) # units = neurons 
regressor.add(Dropout(0.20)) # 20% of neurons get igonored at random during each iteration

# Adding additional hidden layers 
regressor.add(LSTM(units = 52, return_sequences = True)) 
regressor.add(Dropout(0.20))
regressor.add(LSTM(units = 52, return_sequences = True)) 
regressor.add(Dropout(0.20))
regressor.add(LSTM(units = 52, return_sequences = True)) 
regressor.add(Dropout(0.20))
# Adding 3rd LSTM layer with Dropout
regressor.add(LSTM(units = 52))
regressor.add(Dropout(0.20))

# Adding output layer
regressor.add(Dense(units = 1))
# compiling RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting RNN to training set 
regressor.fit(X_train, y_train, epochs = 50, batch_size = 32) 


dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#Getting predicted Stock Prize 
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) # for hirizonal concatination axis = 1 for vertical axis = 0
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 180:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range (60, 80):
    X_test.append(inputs[i -60:i, 0]) # 60 previous stock prices to predict the next stock price 
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = RNN_model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visulizing the results 
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
relative_error = rmse/840

# Savong the model
from keras.models import load_model
regressor.save('RNN_trained_v2_52neurons20DAdam4hl.h5')
RNN_model = load_model('RNN_trained_v2_52neurons20DAdam4hl.h5')

#-----------------------------------------------------------------------------------------------------
# Tuning of RNN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential 
from keras.layers import Dense
def build_RNN(optimizer):
    RNN_model_ = Sequential()
    RNN_model_.add(LSTM(units=53, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    RNN_model_.add(Dropout(0.2))
    RNN_model_.add(LSTM(units = 53, return_sequences = True))
    RNN_model_.add(Dropout(0.2))
    RNN_model_.add(LSTM(units = 53, return_sequences = True))
    RNN_model_.add(Dropout(0.2))
    RNN_model_.add(LSTM(units = 53))
    RNN_model_.add(Dropout(0.2))
    RNN_model_.add(Dense(units = 1))
    RNN_model_.compile(optimizer = optimizer,loss = 'mean_squared_error')
    return RNN_model_
classifier = KerasClassifier(build_fn = build_RNN)
parameters = {'batch_size':[32,28,30], 'epochs': [100, 150], 'optimizer':['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'mean_squared_error', cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

