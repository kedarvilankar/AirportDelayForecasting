# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 23:49:38 2020

@author: kedar
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:11:21 2020

@author: Kedar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MergeDelayAndWeatherData import get_data_for_learning_with_weather
import time



def get_past_future_feature_in_one_row(data_tmp,num_past_hours, 
                                       num_future_hours):
    column_names = data_tmp.columns
    
    
    past_column_names = []
    future_column_names = []
    
    for column_name in column_names:
        for hour_itr in range(0,num_past_hours):
            past_column_names.append(column_name+"_"+str(-(hour_itr+1)))
            
        
        for hour_itr in range(0,num_future_hours):
            future_column_names.append(column_name+"_"+str((hour_itr+1)))
            
    past_hours_df = pd.DataFrame(index = data_tmp.index, 
                                    columns=past_column_names)
    
    future_hours_df = pd.DataFrame(index = data_tmp.index, 
                                    columns=future_column_names)
    
    for row_itr in range(0,len(data_tmp)):
        for column_name in column_names:
            if row_itr>=num_past_hours:
                for hour_itr in range(0,num_past_hours):
                    past_hours_df[column_name+"_"+str(-(hour_itr+1))][row_itr] = \
                        data_tmp[column_name][row_itr-(1+hour_itr)]
            else:
                pass
            
            if row_itr<len(data_tmp)-num_future_hours:
                for hour_itr in range(0,num_future_hours):
                    future_hours_df[column_name+"_"+str((hour_itr+1))][row_itr] = \
                        data_tmp[column_name][row_itr+(1+hour_itr)]
            else:
                pass
    
    return(past_hours_df.join(future_hours_df))

def get_past_feature_in_one_row(data_tmp,num_past_hours):
    column_names = data_tmp.columns
    
    
    past_column_names = []
    
    for column_name in column_names:
        for hour_itr in range(0,num_past_hours):
            past_column_names.append(column_name+"_"+str(-(hour_itr+1)))
            
    past_hours_df = pd.DataFrame(index = data_tmp.index, 
                                    columns=past_column_names)
    
    
    for row_itr in range(0,len(data_tmp)):
        for column_name in column_names:
            if row_itr>=num_past_hours:
                for hour_itr in range(0,num_past_hours):
                    past_hours_df[column_name+"_"+str(-(hour_itr+1))][row_itr] = \
                        data_tmp[column_name][row_itr-(1+hour_itr)]
            else:
                pass
                
    return(past_hours_df)

def get_future_feature_in_one_row(data_tmp, num_future_hours):
    column_names = data_tmp.columns
    
    future_column_names = []
    
    for column_name in column_names:
        for hour_itr in range(0,num_future_hours):
            future_column_names.append(column_name+"_"+str((hour_itr+1)))
    
    future_hours_df = pd.DataFrame(index = data_tmp.index, 
                                    columns=future_column_names)
    
    for row_itr in range(0,len(data_tmp)):
        for column_name in column_names:            
            if row_itr<len(data_tmp)-num_future_hours:
                for hour_itr in range(0,num_future_hours):
                    future_hours_df[column_name+"_"+str((hour_itr+1))][row_itr] = \
                        data_tmp[column_name][row_itr+(1+hour_itr)]
            else:
                pass
    
    return(future_hours_df)
        
airport_code = 'DEN'    
delay_columns = ['PERCENT_DELAYED_DEP','AVG_DEP_DELAY',
                                   'NUM_SCHEDULED_DEP','NUM_SCEDULED_ARR',
                                   'PERCENT_DELAYED_ARR','AVG_ARR_DELAY',
                                   "QUARTER", "MONTH", "DAY_OF_WEEK",
                                   "HOUR"]

save_data_filename = "..\data\processed\\" + airport_code  + "_for_regressor.csv"

#load the data
dataFileName = "..\data\processed\\" + airport_code  + ".csv"
    
data = pd.read_csv(dataFileName)
data["DATETIME"]=pd.to_datetime(data['DATETIME']) 
data.set_index("DATETIME",inplace=True)
data.sort_index()

data.drop(data.columns.difference(delay_columns),1,inplace=True)
data = data[delay_columns]

# data = data[0:500]

# add past and future 24 hr features to each row
start = time.time()
print("hello")

tmp_past_features = get_past_feature_in_one_row(data[["NUM_SCHEDULED_DEP",
                                                "PERCENT_DELAYED_DEP",
                                                "AVG_DEP_DELAY"]], 
                                          num_past_hours=23)
print("Done future features")
end = time.time()
print(end - start)

data = data.join(tmp_past_features)
data.to_csv(save_data_filename,index=True)
