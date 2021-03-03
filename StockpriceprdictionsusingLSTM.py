# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:25:09 2021

@author: Akash

Project: Stock price predictions using LSTM
"""

# Import required libraries
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# read the csv file
df = pd.read_csv('C:/Users/user/Desktop/Akash Desktop/documents/Company/Intertrust/NSE-Tata-Global-Beverages-Limited.csv')
                 
# Remove the date from dataframe
df.drop(columns = ['Date'], axis = 1, inplace = True)

# check for null values
df.isna().sum() # no NULL values

## Lets do some plot to find pattern
# plot for "Open", "High", "Low", "Close"
plt.figure()
plt.plot(df["Open"])
plt.plot(df["High"])
plt.plot(df["Low"])
plt.plot(df["Last"])
plt.plot(df["Close"])
plt.title('NSE-Tata-Global-Beverages-Limited Stock Price History Data')
plt.ylabel('Price')
plt.xlabel('Days')
plt.legend(['Open','High','Low','Last','Close'])
plt.show()
# The figure shows the prices does not vary much and rather coincide

# Plot for "Total Trade Quantity"
plt.figure()
plt.plot(df["Total Trade Quantity"])
plt.title('Tata Total Quantity Traded History Data ')
plt.ylabel('Total Trade Quantity')
plt.xlabel('Days')
plt.show()
# We could see the when the stock price plumetted the trade quantity is high

# Plot for "Turnover (Lacs)"
plt.figure()
plt.plot(df["Turnover (Lacs)"])
plt.title('Tata Turnover (Lacs) History Data ')
plt.ylabel('Turnover (Lacs)')
plt.xlabel('Days')
plt.show()
# We could see the turn over is highest when the prices is high and vice-versa

# Normalising the data
scaler = MinMaxScaler(feature_range = (0, 1))
scaledData = scaler.fit_transform(df)


#divide dataset into train and test
train_fraction = 0.9
train_size = int(len(scaledData)*train_fraction)
test_size = len(scaledData) - train_size
train, test = scaledData[:train_size,], scaledData[train_size:len(scaledData)]

# create a function to create dataset for training and testing
def create_dataset(dataset, WS = 1, FS = 1):
    length = len(dataset)
    data_x, data_y = [], []
    for i in range(length-WS-FS+1):
        a = scaledData[i:(i+WS),]
        data_x.append(a)
        b = scaledData[(i+WS):(i+WS+FS), 4]
        data_y.append(b)
        
    return (np.array(data_x), np.array(data_y))
        
# The first column is the Open value and the fifth Column is the "Close" value of the stock which needs to forecasted
# We will us past some days data to predict the next 5 days close value
WS = 60 # window size or time steps
FS = 5
train_X, train_Y = create_dataset(train, WS = 60, FS = 5) # train set
test_X, test_Y = create_dataset(test, WS = 60, FS = 5) # test set

#shape of data
train_X.shape, train_Y.shape # (1047,60,7) & (1047,5)\
# train_X contains 1047 input sequences where each input sequence has 60 timesteps and feature length = 7

# now LSTM accept input = [batch_size, feature dimension, sequence_length]
# reshaping the train_X and test_X
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[2], WS ))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[2], WS ))

## Building the model
model = Sequential() # initiate

# Add a LSTM with number of neurons = 50,
model.add(LSTM(units = 50, return_sequences = True, input_shape = (train_X.shape[1], WS)))
model.add(Dropout(0.2))

# model.add(LSTM(units = 50, return_sequences = True))
# model.add(Dropout(0.2))

# model.add(LSTM(units = 50, return_sequences = True))
# model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(units = 5))

model.compile(optimizer = 'adam', loss = 'mean_squared_error') # MSE is loss function for regression prob

# model summary
model.summary()

# train the model
model.fit(train_X, train_Y, epochs = 150, batch_size = 32, verbose = 1)

# Prediction
pred = model.predict(test_X)

# Inverse transforming a single column will not work and we need to find the y_min and y_max
ymin = min(df["Close"])
ymax = max(df["Close"])

# define a function for inverse transform
def inverse_trans(data, ymax, ymin):
    newdata = ymin + data*(ymax - ymin)
    return(newdata)

test_pred = inverse_trans(pred, ymax, ymin)
test_act = inverse_trans(test_Y, ymax, ymin)   
    
# calculate the RMSE
RMSEscore = math.sqrt(mean_squared_error(test_act, test_pred)) 
print('The RMSE score is % .3f' % (RMSEscore))

# plot the predicted vs real stock price for the test dataset
plt.figure()
plt.plot(test_act[:,0], color = 'black', label = 'Actual TATA Stock Closing Price')
plt.plot(test_pred[:,0], color = 'green', label = 'Predicted TATA Stock Closing Price')
plt.title('TATA Stock Closing Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Closing Price')
plt.legend()
plt.show()


