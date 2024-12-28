import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

start = '2010-01-01'
end = '2023-12-31'

# Fetch historical stock data using yfinance
df = yf.download('AAPL', start=start, end=end)

print(df.head())
print(df.tail())
plt.plot(df.Close)
print(df)

ma100 = df.Close.rolling(100).mean()

plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100, color='red')
plt.title("Moving Average Graph")

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.8)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.8):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i][0])

x_train , y_train = np.array(x_train) , np.array(y_train)

from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(units=50,activation='relu',return_sequences=True,input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60,activation='relu',return_sequences=True,))
model.add(Dropout(0.3))

model.add(LSTM(units=80,activation='relu',return_sequences=True,))
model.add(Dropout(0.4))

model.add(LSTM(units=120,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1))

print(model.summary())

model.compile(optimizer='adam' , loss='mean_squared_error')
#Fit the model to the data
model.fit(x_train , y_train , epochs=5)
model.save('keras_model.h5')

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)
print(input_data.shape)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test , y_test = np.array(x_test) , np.array(y_test)
print(x_test.shape)
print(y_test.shape)

y_predicated = model.predict(x_test)
print(y_predicated.shape)

print(scaler.scale_)
scale_factor = 1/scaler.scale_[0]
y_predicated = y_predicated * scale_factor
y_test = y_test * scale_factor

plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label= 'Original Price')
plt.plot(y_predicated,'r',label='Predicated Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()