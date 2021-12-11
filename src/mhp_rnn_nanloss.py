import os

import pandas as pd #type: ignore
import numpy as np #type: ignore
from sklearn.preprocessing import MinMaxScaler #type: ignore
import matplotlib.pyplot as plt #type: ignore
import tensorflow as tf #type: ignore
from keras import activations as activ # type: ignore
from keras.layers import Dense #type: ignore
import os

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

train_df, test_df = mhp_df[1:15000], mhp_df[15000:]

train = train_df
scalers = {}
for i in train_df.columns:
    scaler = MinMaxScaler(feature_range=(0, 100))
    s_s = scaler.fit_transform(train[i].values.reshape(-1, 1))
    s_s = np.reshape(s_s, len(s_s))
    scalers["scaler_" + i] = scaler
    train[i] = s_s
test = test_df
for i in train_df.columns:
    scaler = scalers["scaler_" + i]
    s_s = scaler.transform(test[i].values.reshape(-1, 1))
    s_s = np.reshape(s_s, len(s_s))
    scalers["scaler_" + i] = scaler
    test[i] = s_s


def split_series(series, n_past, n_future):
    #
    # n_past ==> no of past observations
    #
    # n_future ==> no of future observations
    #
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = series[window_start:past_end, :], series[past_end:future_end, :]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)


n_past = 1200
n_future = 364
n_features = 11


X_train, y_train = split_series(train.values, n_past, n_future)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
X_test, y_test = split_series(test.values, n_past, n_future)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))


# E2D2
# n_features ==> no of features at each timestep in the data.
#
encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)
encoder_states1 = encoder_outputs1[1:]
encoder_l2 = tf.keras.layers.LSTM(100, return_state=True)
encoder_outputs2 = encoder_l2(encoder_outputs1[0])
encoder_states2 = encoder_outputs2[1:]
#
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])
#
decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(
    decoder_inputs, initial_state=encoder_states1
)

decoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True)(
    decoder_l1, initial_state=encoder_states2
)
decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(
    decoder_l2
)
#
model_e2d2 = tf.keras.models.Model(encoder_inputs, decoder_outputs2)
#
model_e2d2.summary()


plt.plot(train)
plt.show()


reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.01 ** x)

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=3, verbose=1,
    mode='auto', baseline=None, restore_best_weights=True
)

model_e2d2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-15, clipnorm=1, clipvalue=1),
    loss=tf.keras.losses.Huber(),
)
history_e2d2 = model_e2d2.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test),
    batch_size=32,
    verbose=1,
    callbacks=[callback]
)


pred_e2d2 = model_e2d2.predict(X_test)
pred1_e2d2 = model_e2d2.predict(y_test)


for index, i in enumerate(train_df.columns):
    scaler = scalers["scaler_" + i]
    pred1_e2d2[:, :, index] = scaler.inverse_transform(pred1_e2d2[:, :, index])
    pred_e2d2[:, :, index] = scaler.inverse_transform(pred_e2d2[:, :, index])
    y_train[:, :, index] = scaler.inverse_transform(y_train[:, :, index])
    y_test[:, :, index] = scaler.inverse_transform(y_test[:, :, index])

from sklearn.metrics import mean_absolute_error #type: ignore

for index, i in enumerate(train_df.columns):
    print(i)
    for j in range(1, 6):
        print("Day ", j, ":")
        print(
            "MAE-E2D2 : ",
            mean_absolute_error(y_test[:, j - 1, index], pred1_e2d2[:, j - 1, index]),
        )
    print()
    print()
