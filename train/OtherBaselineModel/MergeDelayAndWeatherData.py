# -*- coding(): utf-8 -*-
"""
Created on Thu Jan 23 12:10:05 2020

@author: Kedar
"""

import numpy as np
import pandas as pd

def get_data_for_learning_with_weather(delay_columns,weather_columns, 
                                       airport_code):
   
    #load the data
    dataFileName = "..\data\processed\\" + airport_code  + ".csv"
    weather_data_file = "..\data\processed\\" + airport_code +"_weather.csv"
    
    data = pd.read_csv(dataFileName)
    data["DATETIME"]=pd.to_datetime(data['DATETIME']) 
    data.set_index("DATETIME",inplace=True)
    data.sort_index()
    
    weather_data = pd.read_csv(weather_data_file)
    weather_data["DATETIME"]=pd.to_datetime(weather_data['DATETIME'])
    weather_data.set_index("DATETIME",inplace=True)
    weather_data.sort_index()
    
    
    data.drop(data.columns.difference(delay_columns),1,inplace=True)
    data = data[delay_columns]
    weather_data.drop(weather_data.columns.difference(weather_columns),1,inplace=True)
    weather_data = weather_data[weather_columns]
    
    
    return(data.join(weather_data))
