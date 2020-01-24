# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 22:49:03 2020

@author: Kedar

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from MergeDelayAndWeatherData import get_data_for_learning


lag_order = 24 # past number of observation to look for forecasts
delay_columns = ['PERCENT_DELAYED_DEP','AVG_DEP_DELAY',
                                   'NUM_SCHEDULED_DEP','NUM_SCEDULED_ARR',
                                   'PERCENT_DELAYED_ARR','AVG_ARR_DELAY']
weather_columns = ["PRECIP_PROBABILITY","TEMPERATURE","WIND_SPEED", "WIND_GUST"]
all_columns = delay_columns + weather_columns

#load the data
data = get_data_for_learning(delay_columns,weather_columns)


# data.drop(data.columns.difference(column),1,inplace=True)
# data = data[column]

#creating the train and validation set
train = data[:int(0.8*(len(data)))]
valid = data[int(0.8*(len(data))):]

#fit the model
model = VAR(endog=train)
model_fit = model.fit(lag_order)

# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(valid))
prediction2= np.empty(valid.shape)

# prediction from only last lag_order observations
for itr in range(0,len(valid),lag_order):
    if itr == 0:
        past_df = train.iloc[-lag_order:].to_numpy()
    else:
        past_df = train.iloc[itr-lag_order:itr].to_numpy()
    gt_df = valid.iloc[itr:itr+lag_order]
    prediction2[itr:itr+lag_order,:] = model_fit.forecast(past_df,
                                                          steps=len(gt_df))
    
prediction2[prediction2<0]=0    

#converting predictions to dataframe
store_pred_df = pd.DataFrame(data=prediction, index=valid.index,
                             columns=all_columns)  #
#store_pred_df.to_csv("VAR_prediction.csv",index=True)
#valid.to_csv("GT_delays.csv",index=True)
#check rmse
for (col,i) in zip(all_columns,range(0,len(all_columns))):
    print('rmse value for', col, 'is : ',
          np.sqrt(mean_squared_error(valid[col], prediction[:,i])))
    
    
# plot 
train_len = len(train)
num_samples = 300
x = np.arange(len(data))
plt.plot(x[train_len-num_samples:train_len], 
         train['PERCENT_DELAYED_DEP'][-num_samples:], c='blue')
plt.plot(x[train_len:train_len+num_samples], prediction2[0:num_samples,0], c='green')
plt.plot(x[train_len:train_len+num_samples],
         valid['PERCENT_DELAYED_DEP'][0:num_samples], c='red')
plt.show()


plt.plot(x[train_len-num_samples:train_len],
         train['AVG_DEP_DELAY'][-num_samples:], c='blue')
plt.plot(x[train_len:train_len+num_samples],  
         prediction2[0:num_samples,1], c='green')
plt.plot(x[train_len:train_len+num_samples], 
         valid['AVG_DEP_DELAY'][0:num_samples], c='red')
plt.show()