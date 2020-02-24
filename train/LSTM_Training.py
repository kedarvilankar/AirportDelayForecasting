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

import LSTM_Utils
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, concatenate, Activation 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def build_model(lstm_inp_shape,cat_inp_shape):
    """
    
    Parameters
    ----------
    lstm_inp_shape : LSTM input shape 
    cat_inp_shape : input shape of categorical data

    Returns
    -------
    model : LSTM model architecture to train
    callbacks : callbacks to evaluate after every epocj

    """
    # create LSTM network
    print("Bulding model")
    model1 = Sequential()
    model1.add(LSTM(256, input_shape=(lstm_inp_shape[1], lstm_inp_shape[2]), return_sequences=True))
    model1.add(LSTM(128,return_sequences=True, dropout=0.2))
    model1.add(LSTM(64,return_sequences=False, dropout=0.2))
  
    #model1.add(LSTM(10, input_shape=(lstm_inp_shape[1], lstm_inp_shape[2])))
    model1.add(Dense(num_future_hours*4))
    model1.add(Activation('relu'))
    model1.add(Dense(num_future_hours))
    
    model2 = Sequential()
    model2.add(Dense(128, input_dim=cat_inp_shape[1]))
    model2.add(Activation('relu'))
    model2.add(Dense(64))
    model2.add(Activation('relu'))
    
    
    model_concat = concatenate([model1.output, model2.output], axis=-1)
    model_concat = Dense(num_future_hours)(model_concat)
    model = Model(inputs=[model1.input, model2.input], outputs=model_concat)
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    callbacks = [
                  EarlyStopping(monitor='val_loss', patience=20),
                  ModelCheckpoint(
                      filepath='..\\trained_models\\best_model_'+future_to_predict_JFK[0]+'.h5',
                                  monitor='val_loss', save_best_only=True
                                  ),
                  ReduceLROnPlateau(
                      monitor='val_loss', factor=0.1, 
                                    patience=5, verbose=0, mode='auto', 
                                    min_delta=0.0001, cooldown=0, min_lr=0
                                    )
    
                  ]
    return model,callbacks

num_future_hours = 24
num_past_hours = num_future_hours-1
train_ratio = 0.6
val_ratio = 0.8

categorical_feature_JFK = ['DAY_OF_WEEK', 'HOUR','ICON']
past_numerical_features_JFK = ['NUM_SCHEDULED_DEP','PERCENT_DELAYED_DEP',
                               'AVG_DEP_DELAY','NUM_SCEDULED_ARR',
                               'PERCENT_DELAYED_ARR', 'AVG_ARR_DELAY']
future_to_predict_JFK  = ['PERCENT_DELAYED_DEP']  #PERCENT_DELAYED_DEP
#future_numerical_features_JFK = []
future_numerical_features_JFK = ["PRECIP_INTENSITY","PRECIP_PROBABILITY",
                                                     "TEMPERATURE","WIND_SPEED",
                                                     "WIND_GUST"]
other_airports = ['BOS', 'ATL']
past_numerical_features_other_airport = ['PERCENT_DELAYED_DEP','AVG_DEP_DELAY']

print("Reading JFK Data")
# load JFK data with extended column names
[future_column_names_to_predict_JFK, future_extended_column_names_JFK, 
 past_extended_column_names_JFK, JFK_df] = LSTM_Utils.get_jfk_data(
     num_past_hours,num_future_hours, categorical_feature_JFK, 
     past_numerical_features_JFK, future_to_predict_JFK,
     future_numerical_features_JFK
     )

# load other airport data
print("Reading Other airport Data")
past_extended_column_names_all_other = []
all_other_aiport_df = pd.DataFrame()
for airport_code,airport_itr in zip(other_airports,range(len(other_airports))):
    past_extended_column_names_other,other_df = \
    LSTM_Utils.get_other_airport_data(num_past_hours,
                            past_numerical_features_other_airport,
                            airport_code)
    past_extended_column_names_all_other = \
        past_extended_column_names_all_other + \
            [s + '_' + airport_code for s in past_extended_column_names_other]
    all_other_aiport_df = pd.concat([all_other_aiport_df, other_df], axis=1)

all_other_aiport_df.columns = past_extended_column_names_all_other

# concat all numerical columns of JFK and other airports
all_numerical_df_X = pd.concat(
    [JFK_df[past_extended_column_names_JFK+future_extended_column_names_JFK],
     all_other_aiport_df], axis=1
    )
all_categorical_df_X = JFK_df[categorical_feature_JFK]
future_to_predict_df_Y =  JFK_df[future_column_names_to_predict_JFK]

# split into train test val
print("Split data into train, val, and test")
all_numerical_df_X_train, all_numerical_df_X_val, all_numerical_df_X_test = \
    LSTM_Utils.split_into_train_val_test(all_numerical_df_X, train_ratio, val_ratio)
[future_to_predict_df_Y_train, future_to_predict_df_Y_val, 
 future_to_predict_df_Y_test] = \
    LSTM_Utils.split_into_train_val_test(future_to_predict_df_Y, train_ratio, val_ratio)

# scale, OneHotEncode and normalize
print("Scaling and Normalizing data")
scaler_X = MinMaxScaler(feature_range=(0, 1))
all_numerical_df_X_train = scaler_X.fit_transform(all_numerical_df_X_train)
all_numerical_df_X_val = scaler_X.transform(all_numerical_df_X_val)
all_numerical_df_X_test = scaler_X.transform(all_numerical_df_X_test)

scaler_Y = MinMaxScaler(feature_range=(0, 1))
future_to_predict_df_Y_train = scaler_Y.fit_transform(future_to_predict_df_Y_train)
future_to_predict_df_Y_val = scaler_Y.transform(future_to_predict_df_Y_val)
future_to_predict_df_Y_test = scaler_Y.transform(future_to_predict_df_Y_test)

cat_encoder = OneHotEncoder()
all_categorical_df_X2 = cat_encoder.fit_transform(all_categorical_df_X)
[all_categorical_df_X_train, all_categorical_df_X_val, 
all_categorical_df_X_test] = \
    LSTM_Utils.split_into_train_val_test(all_categorical_df_X2, train_ratio, val_ratio)

all_categorical_df_X_train = all_categorical_df_X_train.todense()
all_categorical_df_X_val = all_categorical_df_X_val.todense()
all_categorical_df_X_test = all_categorical_df_X_test.todense()


# format input as required for LSTM model shape =  [samples, time steps, features]
print("Formating data for LSTM")
LSTM_inp_X_train = all_numerical_df_X_train.reshape(
    (-1,num_future_hours,int(all_numerical_df_X_train.shape[1]/num_future_hours)),
    order='F'
    )
LSTM_inp_X_val = all_numerical_df_X_val.reshape(
    (-1,num_future_hours,int(all_numerical_df_X_val.shape[1]/num_future_hours)),
    order='F'
    )
LSTM_inp_X_test = all_numerical_df_X_test.reshape(
    (-1,num_future_hours,int(all_numerical_df_X_test.shape[1]/num_future_hours)),
    order='F'
    )


# build model and call backs
model, callbacks = build_model(LSTM_inp_X_train.shape, all_categorical_df_X2.shape)


print("Training model")
start = time.time()
history = model.fit(
    [LSTM_inp_X_train, all_categorical_df_X_train], future_to_predict_df_Y_train,
    validation_data=([LSTM_inp_X_val, all_categorical_df_X_val], future_to_predict_df_Y_val),
    epochs=500, callbacks=callbacks, batch_size=300, verbose=2
    )
end = time.time()
print("Done Training")
print(end - start)


# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='Val')
plt.legend()
plt.show()


# make predictions
testPredict = model.predict([LSTM_inp_X_test, all_categorical_df_X_test])
#denormalize prediction and GT
testPredict = scaler_Y.inverse_transform(testPredict)
testPredict[testPredict<0] = 0
test_y = scaler_Y.inverse_transform(future_to_predict_df_Y_test)

rmse = np.sqrt(np.mean((testPredict-test_y)**2, axis=0))
print('Model Performance')
print('Percent Delay Dep RMSE: {:0.4f}.'.format(np.mean(rmse)))

testPredict = testPredict +1
test_y = test_y +1
rel = np.mean(np.abs((test_y - testPredict) / test_y),axis=0) * 100
print('Model Performance')
print('Percent: {:0.4f}.'.format(np.mean(rel)))