import streamlit as st
import yfinance as fn
from datetime import date

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import requests 
import json
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator


import torch
import torch.nn as nn

API_URL = "http://127.0.0.1:8000/stockprediction"

# API_URL = "http://127.0.0.1:8000"


## FRONTEND

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Lemon Exchange")

stocks = ("AAPL", "GOOG", "MSFT", "GME", "TSLA")

selected_stocks = st.selectbox("Select dataset for prediction", stocks)
#This code is like a drop box which will allow us to select from different options


n_years = st.slider("Years of Prediction:", 1,4) 
# This is the code for the slider widget which you can manipulate with respect to your comfort
# Yearss of selection from 1 to 4

period = n_years * 365 
# This variable period stands for no of days

@st.cache_data
# Very important code as it stores the data for eg of apple once we run the load_data function
# And if we re-run this function then it won't reinstall the data again.

# Function to load data
def load_data(ticker):
    data = fn.download(ticker, START, TODAY)
    data.reset_index(inplace = True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data...done!")


st.subheader('Raw Data')
st.write(data.tail())


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter( x = data['Date'], y = data['Open'], name = 'stock_open')) 
    # Use the add_trace function to add additional traces to the plotly object. 
    # Specify the data for each trace using the x and y arguments, and specify the type of plot using the type argument
    fig.add_trace(go.Scatter( x = data['Date'], y = data['Close'], name = 'stock_close')) 
    fig.layout.update(title_text="Time Series Analysis", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plot_raw_data()


## Now we will start making changes in the main file to prevent confusions


data['Date'] = pd.to_datetime(data['Date'])
data = data.set_axis(data['Date'], copy = False)
data.drop(columns=['Open', 'High', 'Low', 'Volume'])


close_data = data['Close'].values
close_data = close_data.reshape((-1,1))

split_percent = 0.80
split = int(split_percent*len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

date_train = data['Date'][:split]
date_test = data['Date'][split:]


look_back = 15

train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)     
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)


from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(
    LSTM(10,
        activation='relu',
        input_shape=(look_back,1))
)
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

num_epochs = 25
model.fit_generator(train_generator, epochs=num_epochs, verbose=1)

prediction = model.predict_generator(test_generator)

close_train = close_train.reshape((-1))
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))

trace1 = go.Scatter(
    x = date_train,
    y = close_train,
    mode = 'lines',
    name = 'Data'
)
trace2 = go.Scatter(
    x = date_test,
    y = prediction,
    mode = 'lines',
    name = 'Prediction'
)
trace3 = go.Scatter(
    x = date_test,
    y = close_test,
    mode='lines',
    name = 'Ground Truth'
)
st.subheader("Predicting Existing Data")
layout = go.Layout(
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Close"},
    xaxis_rangeslider_visible = True
)
fig3 = go.Figure(data=[trace1, trace2, trace3], layout=layout)
st.plotly_chart(fig3)
# fig1.show()

close_data = close_data.reshape((-1))

def predict(num_prediction, model):
    prediction_list = close_data[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
        
    return prediction_list
    
def predict_dates(num_prediction):
    last_date = data['Date'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

num_prediction = 30
forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)

prediction = model.predict_generator(test_generator)

close_train = close_train.reshape((-1))
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))

trace1 = go.Scatter(
    x = date_train,
    y = close_train,
    mode = 'lines',
    name = 'Data'
)
trace2 = go.Scatter(
    x = date_test,
    y = prediction,
    mode = 'lines',
    name = 'Prediction'
)
trace3 = go.Scatter(
    x = date_test,
    y = close_test,
    mode='lines',
    name = 'Ground Truth'
)
trace4 = go.Scatter(
    x = forecast_dates,
    y = forecast,
    mode='lines',
    name = 'forecast'
)
st.subheader("Forecast")
layout = go.Layout(
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Close"},
    xaxis_rangeslider_visible = True
)
fig4 = go.Figure(data=[trace1, trace2, trace3,trace4], layout=layout)
st.plotly_chart(fig4)
# fig.show()


# Forecasting
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={'Date':'ds', 'Close':'y'})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast Data")
st.write(forecast.tail())


st.write("Forecast Data")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)




# if st.button("Predict"):
#     payload = {"stock_name": selected_stocks}

#     try:
#         response = requests.post(API_URL, json=payload)
#         response.raise_for_status()

#         predictions = response.json()
#         predicted_prices = predictions["prediction"]

#         actual_prices = data['Close'].tolist()
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=data.index, y=actual_prices, name='Actual'))
#         fig.add_trace(go.Scatter(x=data.index[-len(predicted_prices):], y=predicted_prices, name='Predicted'))
#         # fig.add_trace(go.Scatter(x=data['Date'], y=actual_prices, name='Actual'))
#         # fig.add_trace(go.Scatter(x=data['Date'][-len(predicted_prices):], y=predicted_prices, name='Predicted'))
#         fig.update_layout(title=f"{selected_stocks} Stock Price", xaxis_rangeslider_visible = True)
#         st.plotly_chart(fig)

#     except requests.exceptions.RequestException as e:
#         st.error(f"Error occurred while making the request: {e}")






