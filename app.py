import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go

start = '2010-01-01'
end = '2023-12-31'


st.title('Stock Prediction')
user_input = st.text_input('Enter Stock Ticker','AAPL')
# Fetch historical stock data using yfinance
df = yf.download(user_input, start=start, end=end)

st.subheader('Data From 2010 - 2023')
st.write(df.describe())

#visualization
st.subheader('Closing Price Chart vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price Chart vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price Chart vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.8)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.8):int(len(df))])
# print(data_training.shape)
# print(data_testing.shape)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

#Load My Model:
model = load_model('keras_model.h5')

#testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test , y_test = np.array(x_test) , np.array(y_test)
y_predicated = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicated = y_predicated * scale_factor
y_test = y_test * scale_factor

st.subheader('Predictions VS Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label= 'Original Price')
plt.plot(y_predicated,'r',label='Predicated Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

#CandelStick Chart
st.subheader('CandleStick Chart')
fig = go.Figure(data = [go.Candlestick(x=df.index,open=df['Open'],high=df['High'],low=df['Low'],close=df['Close'],increasing_line_color='green',decreasing_line_color='red',line_width=1,opacity=0.7)])
st.plotly_chart(fig)