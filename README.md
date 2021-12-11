# HousingMarket_TimeSeries
A time series neural network that forecasts the median US house price
### Sam Morse
### Professor Roy Turner
### University of Maine COS70
---
Neural networks:

`mhp_rnn_nanloss.py`
- training code for RNN that had consistent Nan loss

`mhp_nn_multi.py`
- training code for a series of neural networks as well as data visualization

`mhp.py`
- training code for working sequential neural network

---

**Loading Networks**

complete networks can be found in:

`src/models`

method

`new_model = tf.keras.models.load_model('filepath')`

using `mhp_data_set.csv`, training was done on first 15000 time slices

```
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
values = mhp_df.values
test = values[n_train_days:, :]
```
