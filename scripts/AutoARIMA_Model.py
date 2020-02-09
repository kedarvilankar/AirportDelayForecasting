# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:58:59 2020

@author: Kedar
"""
import pandas as pd
import pmdarima as pm
#from pmdarima.model_selection import train_test_split
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
#from pyramid.arima import auto_arima

#load the data
dataFileName = "..\data\processed\JFK.csv"
data = pd.read_csv(dataFileName)
data["DATETIME"]=pd.to_datetime(data['DATETIME']) 
data.set_index("DATETIME",inplace=True)
data.sort_index()
data.drop(data.columns.difference(['PERCENT_DELAYED_DEP','AVG_DEP_DELAY']), 1,
          inplace=True)
#divide into train and validation set
train = data[:int(0.8*(len(data)))]
valid = data[int(0.8*(len(data))):]

#preprocessing (since arima takes univariate series as input)
train.drop('PERCENT_DELAYED_DEP',axis=1,inplace=True)
valid.drop('PERCENT_DELAYED_DEP',axis=1,inplace=True)

#plotting the data
# train['PERCENT_DELAYED_DEP'].plot()
# valid['PERCENT_DELAYED_DEP'].plot()

#building the model

# Fit your model
model = pm.auto_arima(train, error_action='ignore', trace=True,
                      suppress_warnings=True, maxiter=10,
                      seasonal=True, m=20)
print("Done Fitting")

#load if model isalready  pickeled
# model = pickle.load( open( "ARIMABaselineModel.sav", "rb" ) )

# make your forecasts
forecasts = model.predict(valid.shape[0])  # predict N steps into the future
print("Done Forecasting")


# plot prediction
train_len = len(train)
num_samples = 150
x = np.arange(len(data))
plt.plot(x[train_len-num_samples:train_len], train[-num_samples:], c='blue')
plt.plot(x[train_len:train_len+num_samples], forecasts[0:num_samples], c='green')
plt.plot(x[train_len:train_len+num_samples], valid[0:num_samples], c='red')
plt.show()


#compute performnace
rmse= np.sqrt(mean_squared_error(valid,forecasts))
print(rmse)



test_y = valid.to_numpy()


rmse = np.sqrt(np.mean((forecasts-test_y)**2, axis=0))
print('Model Performance')
print('RMSE: {:0.4f}.'.format(np.mean(rmse)))

forecasts = forecasts +1
test_y = test_y +1
rel = np.mean(np.abs((test_y - forecasts) / test_y),axis=0) * 100
print('Model Performance')
print('Rel: {:0.4f}.'.format(np.mean(rel)))

