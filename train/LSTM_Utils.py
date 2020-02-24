# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:07:30 2020

@author: Kedar

All the helper function to build the LSTM model to predeict delays
"""
import numpy as np
import pandas as pd


def prepend(List, str):
    """
    Prepend str to List

    """
    # Using format() 
    str += '{0}'
    List = ((map(str.format, List))) 
    return List

def get_extended_past_columns(past_columns, num_past_hours):
    """
    get extended unique names for past column names
    """
    past_extended_column_names = []
    hour_list = list(range(-1,-(num_past_hours+1),-1))
    for column_name in past_columns:
        past_extended_column_names.append(column_name)     
        past_extended_column_names.extend(prepend(hour_list, column_name + '_'))
        
    return past_extended_column_names

def get_extended_future_columns(future_columns, num_future_hours): 
    """
    get extended unique names for future column names
    """
    future_extended_column_names = [] 
    hour_list = list(range(1,(num_future_hours+1)))       
    for column_name in future_columns:
        future_extended_column_names.extend(prepend(hour_list, column_name + '_'))

    return future_extended_column_names

def get_jfk_data(num_past_hours, num_future_hours, categorical_feature_JFK,
                 past_numerical_features_JFK, future_to_predict_JFK,
                 future_numerical_features_JFK):
    """
    Read JFK data preprocessed for the regression model also read only the 
    columns which are passed to the function
    """
    
    JFK_data_file = "..\data\processed\JFK_for_regressor.csv"

    past_extended_column_names_JFK = get_extended_past_columns(
        past_numerical_features_JFK, num_past_hours
        )
                
    future_column_names_to_predict_JFK = get_extended_future_columns(
        future_to_predict_JFK, num_future_hours
        )
          
    future_extended_column_names_JFK = [] 
    if len(future_numerical_features_JFK) !=0:
        future_extended_column_names_JFK = get_extended_future_columns(
            future_numerical_features_JFK, num_future_hours
            )

        
    all_columns_JFK = past_extended_column_names_JFK + categorical_feature_JFK +\
        future_extended_column_names_JFK + future_column_names_to_predict_JFK
    
    JFK_df = pd.read_csv(JFK_data_file)
    JFK_df = JFK_df.fillna(0)

    # convert categorical variables to categories type
    for column_name in  categorical_feature_JFK:
        JFK_df[column_name]=JFK_df[column_name].astype("category")

    JFK_df["DATETIME"]=pd.to_datetime(JFK_df['DATETIME']) 
    JFK_df.set_index("DATETIME",inplace=True)
    JFK_df.sort_index()
    JFK_df.drop(JFK_df.index[:48], inplace=True)
    JFK_df.drop(JFK_df.tail(48).index,inplace=True) # drop last n rows
    print("Sorted time index and droped top and boittom")

    JFK_df = JFK_df[all_columns_JFK]
    print("Done extarcting required cols")
    
        
    return future_column_names_to_predict_JFK, \
        future_extended_column_names_JFK, past_extended_column_names_JFK,JFK_df
    

def get_other_airport_data(num_past_hours,past_numerical_features_other_airport,
                           airport_code):
    """
    get data of other airports from the processed data and extract only the column names 
    passed to the function
    """
    data_file = "..\data\processed\\" + airport_code +"_for_regressor.csv"
            
    past_extended_column_names_other = get_extended_past_columns(
        past_numerical_features_other_airport, num_past_hours
        )
        
    other_df = pd.read_csv(data_file)
    other_df["DATETIME"]=pd.to_datetime(other_df['DATETIME']) 
    other_df.set_index("DATETIME",inplace=True)
    other_df.sort_index()
    other_df.drop(other_df.index[:48], inplace=True)
    other_df.drop(other_df.tail(48).index,inplace=True) # drop last n rows
    
    other_df.drop(other_df.columns.difference(past_extended_column_names_other),
                  1,inplace=True)
    other_df = other_df[past_extended_column_names_other]
    
    return past_extended_column_names_other,other_df



def split_into_train_val_test(tmp_df, train_ratio, val_ratio):
    """
    split dataframe into train/val/test using the passed ratio
    """
    df_len = tmp_df.shape[0]
    train = tmp_df[:int(train_ratio*df_len)]
    val = tmp_df[int(train_ratio*df_len):int(val_ratio*df_len)]
    test = tmp_df[int(val_ratio*df_len):]
    return train, val, test
