# Forecast airport delays
The project forecasts hourly average airport delays for the next 24 hours, using the current and recent delays (past 24 hours) at an airport, 
the local weather forecast, and recent delays at other connecting airports in the region. Delays cost a lot to airline industry in terms of 
increased spending on fuel, crew, maintainenca, and compensation. A system that could forecast these delays could be beneficial for the 
airlines. Such a delay forecaster could petentially be also used by stores and restaurants at airport terminals to restock the invetories well 
ahead of the delays and increase their revenue. The current publicly available delay forecaster predicts delays only for next one or two hours, 
the proposed forecaster will predict hourly delays for the next 24 hours.


### The forecasted predicts two delay features.
1) Percent Delayed Flights: The percent number of scheduled flights delayed by more than 15 minutes
2) Average Departure Delay: The average delay per delayed flight in minutes.


### Directory Navigation
- **data**:  Folder used to dump raw and processed data 
- **notebooks**:  Folder containing Jupyter Notebooks to perform EDA on data
- **train**: Folder containing scripts to train LSTM model. 
	- **OtherBaselineModel**: Subfolder containing scripts to train baseline ARIMA and Vector Auto Regression models
	
### Model Training
The final model for the project is an LSTM model. The architecture and the training code of the model is in *scripts* folder and the file name is *LSTM_Training.py*.
The model trains using 60/20/20 Train/Val/Test and early stoping on validation set loss.


### Web App
URL: https://airport-delay-forecast.herokuapp.com/
Web App source code is maintained on Heroku Git