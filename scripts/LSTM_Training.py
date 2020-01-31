# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:25:06 2020

@author: Kedar
"""

import numpy as np
import pandas as pd
import time
import pickle 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, Activation 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)  
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

def get_jfk_data(num_past_hours, num_future_hours, categorical_feature_JFK,
                 past_numerical_features_JFK, future_to_predict_JFK):
    
    JFK_data_file = "..\data\processed\JFK_for_regressor.csv"
    
    past_column_names_JFK = []
    for column_name in past_numerical_features_JFK:
            for hour_itr in range(0,num_past_hours):
                past_column_names_JFK.append(column_name+"_"+str(-(hour_itr+1)))
                
    future_column_names_JFK = []  
    for column_name in future_to_predict_JFK:
        for hour_itr in range(0,num_future_hours):
                future_column_names_JFK.append(column_name+"_"+str((hour_itr+1)))
        
                
    past_column_names_JFK = past_numerical_features_JFK + \
        past_column_names_JFK
        
    all_columns_JFK = past_column_names_JFK + categorical_feature_JFK +\
        future_column_names_JFK
    
    JFK_df = pd.read_csv(JFK_data_file)
    JFK_df["DATETIME"]=pd.to_datetime(JFK_df['DATETIME']) 
    JFK_df.set_index("DATETIME",inplace=True)
    JFK_df.sort_index()
    JFK_df.drop(JFK_df.index[:48], inplace=True)
    JFK_df.drop(JFK_df.tail(48).index,inplace=True) # drop last n rows
    
    JFK_df.drop(JFK_df.columns.difference(all_columns_JFK),1,inplace=True)
    JFK_df = JFK_df[all_columns_JFK]
    
    # convert categorical variables to categories type
    for column_name in  categorical_feature_JFK:
        JFK_df[column_name]=JFK_df[column_name].astype("category")
        
    return future_column_names_JFK, past_column_names_JFK,JFK_df
    

def get_weather_data(numeric_weather_column,categorical_weather_column):
    weather_data_file = "..\data\processed\JFK_weather.csv"
    weather_df = pd.read_csv(weather_data_file)
    weather_df["DATETIME"]=pd.to_datetime(weather_df['DATETIME'])
    weather_df.set_index("DATETIME",inplace=True)
    weather_df.sort_index()
    
    weather_df.drop(weather_df.index[:48], inplace=True)
    weather_df.drop(weather_df.tail(48).index,inplace=True) # drop last n rows
    
    all_columns_weather = numeric_weather_column + categorical_weather_column
    
    weather_df.drop(weather_df.columns.difference(all_columns_weather),1,
                    inplace=True)
    weather_df = weather_df[all_columns_weather]
    
    # convert categorical variables to categories type
    for column_name in  categorical_weather_column:
        weather_df[column_name]=weather_df[column_name].astype("category")
        
    # fill nans with previous value
    weather_df = weather_df.fillna(method='ffill')
    return weather_df

def get_other_airport_data(num_past_hours,past_numerical_features_other_airport,
                           airport_code):
    data_file = "..\data\processed\\" + airport_code +"_for_regressor.csv"
    
    past_column_names_other = []
    for column_name in past_numerical_features_other_airport:
            for hour_itr in range(0,num_past_hours):
                past_column_names_other.append(column_name+"_"+str(-(hour_itr+1)))
    
    past_column_names_other = past_numerical_features_other_airport + \
        past_column_names_other
        
    other_df = pd.read_csv(data_file)
    other_df["DATETIME"]=pd.to_datetime(other_df['DATETIME']) 
    other_df.set_index("DATETIME",inplace=True)
    other_df.sort_index()
    other_df.drop(other_df.index[:48], inplace=True)
    other_df.drop(other_df.tail(48).index,inplace=True) # drop last n rows
    
    other_df.drop(other_df.columns.difference(past_column_names_other),1,inplace=True)
    other_df = other_df[past_column_names_other]
    
    return past_column_names_other,other_df

num_past_hours = 23
num_future_hours = 24
train_ratio = 0.6
val_ratio = 0.8

categorical_feature_JFK = ['DAY_OF_WEEK', 'HOUR']
past_numerical_features_JFK = ['NUM_SCHEDULED_DEP','PERCENT_DELAYED_DEP',
                               'AVG_DEP_DELAY','NUM_SCEDULED_ARR',
                               'PERCENT_DELAYED_ARR', 'AVG_ARR_DELAY']
future_to_predict_JFK  = ['PERCENT_DELAYED_DEP']  #PERCENT_DELAYED_DEP

other_airports = ['BOS', 'ATL']
past_numerical_features_other_airport = ['PERCENT_DELAYED_DEP','AVG_DEP_DELAY']

future_column_names_JFK, past_column_names_JFK, JFK_df = get_jfk_data(num_past_hours,num_future_hours,
                                              categorical_feature_JFK,
                                              past_numerical_features_JFK,
                                              future_to_predict_JFK)

## Weather data
numeric_weather_column = ["PRECIP_PROBABILITY","TEMPERATURE","WIND_SPEED", "WIND_GUST"]
categorical_weather_column = ["ICON"]
weather_df = get_weather_data(numeric_weather_column,categorical_weather_column)
weather_df = weather_df.join(JFK_df[categorical_feature_JFK])
num_pipeline = Pipeline([
    ("std_scaler", MinMaxScaler()),
      ])
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, numeric_weather_column),
    ("cat", OneHotEncoder(), categorical_weather_column+categorical_feature_JFK),
    ])

weather_df = full_pipeline.fit_transform(weather_df)   
weather_df_train = weather_df[:int(train_ratio*(len(JFK_df))),:]
weather_df_val = weather_df[int(train_ratio*(len(JFK_df))):int(val_ratio*(len(JFK_df))),:]  
weather_df_test = weather_df[int(val_ratio*(len(JFK_df))):,:] 


# normalize the JFK dataset
scaler_JFK_dep = MinMaxScaler(feature_range=(0, 1))
JFK_df_normalized = scaler_JFK_dep.fit_transform(JFK_df[past_column_names_JFK])



# shape training data for LSTM shape =  [samples, time steps, features]
jfk_df_for_LSTM = np.empty(shape = [len(JFK_df), num_past_hours+1, 
                                    len(past_numerical_features_JFK)],
                           dtype = float)

for itr in range(len(JFK_df)):
    for feature_num in range(len(past_numerical_features_JFK)):
        jfk_df_for_LSTM[itr,:(num_past_hours+1),feature_num] = \
            JFK_df_normalized[itr,(feature_num*(num_past_hours+1)):((feature_num*(num_past_hours+1))+24)]
            

 
jfk_df_for_LSTM_y = np.empty(shape = [len(JFK_df), num_future_hours],
                           dtype = float)    
for itr in range(len(JFK_df)):
    for feature_num in range(len(future_to_predict_JFK)):          
        jfk_df_for_LSTM_y[itr,:] = JFK_df[future_column_names_JFK].iloc[itr]
            
other_df_for_LSTM = np.empty(shape = [len(JFK_df), num_past_hours+1, 
                                        len(past_numerical_features_other_airport)*len(other_airports)],
                               dtype = float)
# normalize the other airport  dataset
for airport_code,airport_itr in zip(other_airports,range(len(other_airports))):
    past_column_names_other,other_df = \
    get_other_airport_data(num_past_hours,
                           past_numerical_features_other_airport,
                           airport_code)
    scaler_other_airport = MinMaxScaler(feature_range=(0, 1))
    other_df = scaler_other_airport.fit_transform(other_df[past_column_names_other])
    
    for itr in range(len(other_df)):
        for feature_num in range(len(past_numerical_features_other_airport)):
            other_df_for_LSTM[itr,:(num_past_hours+1),
                              (airport_itr*len(past_numerical_features_other_airport))+feature_num] = \
                other_df[itr,(feature_num*(num_past_hours+1)):((feature_num*(num_past_hours+1))+24)]

jfk_df_for_LSTM = np.concatenate((jfk_df_for_LSTM, other_df_for_LSTM), axis=2)

# normalize the  train y dataset
scaler_y = MinMaxScaler(feature_range=(0, 1))
jfk_df_for_LSTM_y = scaler_y.fit_transform(jfk_df_for_LSTM_y)


# concat other airport info to jfk_df_for_LSTM

train_X_set = jfk_df_for_LSTM[:int(train_ratio*(len(JFK_df))),:,:]
val_X_set = jfk_df_for_LSTM[int(train_ratio*(len(JFK_df))):int(val_ratio*(len(JFK_df))),:,:]
test_X_set = jfk_df_for_LSTM[int(val_ratio*(len(JFK_df))):,:,:]

train_y = jfk_df_for_LSTM_y[0:int(train_ratio*(len(JFK_df))),:]
val_y = jfk_df_for_LSTM_y[int(train_ratio*(len(JFK_df))):int(val_ratio*(len(JFK_df))),:]
test_y = jfk_df_for_LSTM_y[int(val_ratio*(len(JFK_df))):,:]


# num_pipeline = Pipeline([
#     ("std_scaler", StandardScaler()),
#     ("pca", PCA(n_components=10)),
#      ])
# full_pipeline = ColumnTransformer([
#     ("num", num_pipeline, past_numerical_features_JFK+numeric_weather_column),
#     ("cat", OneHotEncoder(), categorical_feature_JFK+categorical_weather_column),
#     ])

# train_x_prepared = full_pipeline.fit_transform(train_x)
# test_x_prepared = full_pipeline.transform(test_x)


# create and fit the LSTM network
print("Bulding model")
model1 = Sequential()
model1.add(LSTM(10, input_shape=(train_X_set.shape[1], train_X_set.shape[2])))
model1.add(Dense(num_future_hours*2))
model1.add(Activation('relu'))
model1.add(Dense(num_future_hours))
# model1.compile(loss='mean_squared_error', optimizer='adam')

model2 = Sequential()
model2.add(Dense(40, input_dim=weather_df_train.shape[1]))
model2.add(Activation('relu'))
model2.add(Dense(10))


model_concat = concatenate([model1.output, model2.output], axis=-1)
model_concat = Dense(num_future_hours)(model_concat)
model = Model(inputs=[model1.input, model2.input], outputs=model_concat)

model.compile(loss='mean_squared_error', optimizer='adam')

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(
                 filepath='..\trained_models\best_model_'+future_to_predict_JFK[0]+'.h5',
                             monitor='val_loss', save_best_only=True
                             )
             ]

print("Training model")
start = time.time()
history = model.fit([train_X_set, weather_df_train], train_y, 
                    validation_data=([val_X_set, weather_df_val], val_y), 
                    epochs=500, callbacks=callbacks, 
                    batch_size=300, verbose=2)
end = time.time()
print("Done Training")
print(end - start)


# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='Val')
plt.legend()
plt.show()


# make predictions
testPredict = model.predict([test_X_set, weather_df_test])
#denormalize prediction and GT
testPredict = scaler_y.inverse_transform(testPredict)
testPredict[testPredict<0] = 0
test_y = scaler_y.inverse_transform(test_y)

rmse = np.sqrt(np.mean((testPredict-test_y)**2, axis=0))
print('Model Performance')
print('Percent Delay Dep RMSE: {:0.4f}.'.format(np.mean(rmse)))


# save results as csv
testPredict_df = pd.DataFrame(testPredict)
testPredict_df['DATETIME'] = JFK_df.index[int(val_ratio*(len(JFK_df))):]
testPredict_df.to_csv('LSTM_op_' + future_to_predict_JFK[0] + '.csv')

test_y_df = pd.DataFrame(test_y)
test_y_df['DATETIME'] = JFK_df.index[int(val_ratio*(len(JFK_df))):]
test_y_df.to_csv('LSTM_y_' + future_to_predict_JFK[0] + '.csv')



