import os
from math import sqrt
import pandas as pd #type: ignore
import numpy as np #type: ignore
from numpy import concatenate #type: ignore
import matplotlib.pyplot as plt #type: ignore
import tensorflow as tf #type: ignore
from pandas import DataFrame #type: ignore
from pandas import concat #type: ignore
from sklearn.preprocessing import LabelEncoder #type: ignore
from sklearn.preprocessing import MinMaxScaler #type: ignore
from sklearn.metrics import mean_squared_error #type: ignore
from keras.models import Sequential #type: ignore
from keras.layers import Dense #type: ignore
from keras.layers import LSTM #type: ignore
from keras import activations as activ # type: ignore

pd.options.mode.chained_assignment = None  # default='warn'

dir_name = os.path.dirname(__file__)
relative_path = "data/train.csv"
file_path = os.path.join(dir_name, relative_path)

mhp_df = pd.read_csv(
    file_path,
    sep=",",
    header=0,
    low_memory=False,
    infer_datetime_format=True,
    parse_dates={"datetime": [0]},
    index_col=["datetime"],
)
mhp_df.head()


print(mhp_df.head(5))

# values = mhp_df.values
# # specify columns to plot
# groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# i = 1
# # plot each column
# plt.figure()
# plt.subplots_adjust(left= .125, right= .9, bottom= .1, top= 1.9, wspace=.2, hspace=.2)
# plt.rcParams['font.size'] = 8
# for group in groups:
# 	plt.subplot(len(groups), 1, i)
# 	plt.plot(values[:, group])
# 	plt.title(mhp_df.columns[group], y=.05, loc='right')
# 	i += 1
# plt.show()


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# Load in the data set
values = mhp_df.values
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# specify the number of lag days
n_days = 365 * 3
n_features = 11
# frame as supervised learning
reframed = series_to_supervised(scaled, n_days, 1)
print(reframed.shape)


# split into train and test sets
values = reframed.values
n_train_days = 15000
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
n_obs = n_days * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
test_X = test_X.reshape((test_X.shape[0], n_days, n_features))
# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# design network
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1, activation=activ.relu))
model.add(Dense(64, activation=activ.relu))
model.add(Dense(128, activation=activ.relu))
model.add(Dense(64, activation=activ.relu))
model.add(Dense(1, activation=activ.relu))

model.compile(loss='huber', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=25, batch_size=128, validation_data=(test_X, test_y), verbose=1, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_days*n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -10:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -10:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

#save model


model.save(os.path.join(dir_name, 'models/seq_nn'))