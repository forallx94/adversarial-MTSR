import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph.
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.model_selection import KFold # use for cross validation
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline # pipeline making
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score

import tensorflow.keras.backend as K
import os

## for Deep-learing:
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import itertools

from tensorflow.keras.layers import Conv1D, BatchNormalization,\
Dropout, Dense, InputLayer, Flatten, MaxPool1D, Activation, GlobalAveragePooling1D

## Data can be downloaded from: http://archive.ics.uci.edu/ml/machine-learning-databases/00235/
## Just open the zip file and grab the file 'household_power_consumption.txt' put it in the directory
## that you would like to run the code.

model_path = '../../Trained models/Power consumption dataset/Power_regression_CNN.h5'

df = pd.read_csv('../../Dataset/household_power_consumption.txt', sep=';',
                 parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True,
                 low_memory=False, na_values=['nan','?'], index_col='dt')

# # load 
# Xtest1 = pd.read_csv('../../Output/Power_GRU_BIM_attack.csv', sep=',', header=None)

## finding all columns that have nan:

droping_list_all=[]
for j in range(0,7):
    if not df.iloc[:, j].notnull().all():
        droping_list_all.append(j)
        #print(df.iloc[:,j].unique())
droping_list_all

# filling nan with mean in any columns

for j in range(0,7):
        df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())

# another sanity check to make sure that there are not more any nan
df.isnull().sum()

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	dff = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(dff.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(dff.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

## resampling of data over days
df_resample = df.resample('h').mean()
df_resample.shape

## * Note: I scale all features in range of [0,1].

## If you would like to train based on the resampled data (over hour), then used below
values = df_resample.values


## full data without resampling
#values = df.values

# integer encode direction
# ensure all data is float
#values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# drop columns we don't want to predict
reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values

n_train_time = 365*72
train = values[:n_train_time, :]
test = values[n_train_time:, :]
##test = values[n_train_time:n_test_time, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

#df = pd.DataFrame(test_X)


## save to xlsx file

#filepath1 = '../../Output/Power_train_dataset.csv'


#df.to_csv(filepath1, index=False)
# test_X=Xtest1.to_numpy()

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

cnn = Sequential()
cnn.add(InputLayer(input_shape=(train_X.shape[1], train_X.shape[2])))
#cnn.add(BatchNormalization(axis=-1))  #Scaling the data

cnn.add(Conv1D(filters=60,
               kernel_size=1,
               padding="valid",
               activation="relu")
       )

cnn.add(Conv1D(filters=60,
               kernel_size=1,
               padding="valid",
               activation="relu")
       )

cnn.add(Conv1D(filters=60,
               kernel_size=1,
               padding="valid",
               activation="relu")
       )

cnn.add(Flatten())
cnn.add(Dense(50, activation='relu'))
cnn.add(Dense(units=1))
cnn.add(Activation("relu"))


def compute_gradient(model_fn, loss_fn, x, y, targeted):
    """
    Computes the gradient of the loss with respect to the input tensor.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param loss_fn: loss function that takes (labels, logits) as arguments and returns loss.
    :param x: input tensor
    :param y: Tensor with true labels. If targeted is true, then provide the target label.
    :param targeted:  bool. Is the attack targeted or untargeted? Untargeted, the default, will
                      try to make the label incorrect. Targeted will instead try to move in the
                      direction of being more like y.
    :return: A tensor containing the gradient of the loss with respect to the input tensor.
    """

    with tf.GradientTape() as g:
        g.watch(x)
        # Compute loss
        loss = loss_fn(y, model_fn(x))
        if (
            targeted
        ):  # attack is targeted, minimize loss of target label rather than maximize loss of correct label
            loss = -loss

    # Define gradient of loss wrt input
    grad = g.gradient(loss, x)
    return grad


def fgsm(X, Y, model,epsilon,targeted= False):
    ten_X = tf.convert_to_tensor(X)
    grad = compute_gradient(model,rmse,ten_X,Y,targeted)
    dir=np.sign(grad)
    return X + epsilon * dir, Y


if os.path.isfile(model_path):
    cnn.load_weights(model_path)

    # make adversarial example
    adv_X, _ = fgsm(X =test_X, Y=test_y, model=cnn , epsilon=0.2)

    # make a prediction
    adv_yhat = cnn.predict(adv_X,verbose=1, batch_size=200)
    adv_X = adv_X.reshape((adv_X.shape[0], 7))

    # make a prediction
    yhat = cnn.predict(test_X,verbose=1, batch_size=200)
    test_X = test_X.reshape((test_X.shape[0], 7))

    # invert scaling for adv forecast
    inv_adv_yhat = np.concatenate((adv_yhat, test_X[:, -6:]), axis=1)
    inv_adv_yhat = scaler.inverse_transform(inv_adv_yhat)
    inv_adv_yhat = inv_adv_yhat[:,0]

    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    # calculate adv RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_adv_yhat))
    print('Test adv RMSE: %.3f' % rmse)

    ## time steps, every step is one hour (you can easily convert the time step to the actual time index)
    ## for a demonstration purpose, I only compare the predictions in 200 hours.

    fig_verify = plt.figure(figsize=(100, 50))
    aa=[x for x in range(200)]
    plt.plot(aa, inv_y[:200], marker='.', label="actual")
    plt.plot(aa, inv_yhat[:200], 'r', label="prediction")
    plt.plot(aa, inv_adv_yhat[:200], 'g', label="adv prediction")
    plt.ylabel('Global_active_power', size=15)
    plt.xlabel('Time step', size=15)
    plt.legend(fontsize=15)
    plt.show()