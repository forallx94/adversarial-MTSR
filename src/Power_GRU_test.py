import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph.
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from utils import power_data, fgsm, compute_gradient, rmse

## for Deep-learing:
import tensorflow.keras.backend as K
# from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.models import load_model


## Data can be downloaded from: http://archive.ics.uci.edu/ml/machine-learning-databases/00235/
## Just open the zip file and grab the file 'household_power_consumption.txt' put it in the directory
## that you would like to run the code.

model_path = '../Trained models/Power_regression_GRU.h5'

train_X, train_y, test_X, test_y , scaler = power_data()

# if best iteration's model was saved then load and use it
if os.path.isfile(model_path):
	model = load_model(model_path, custom_objects={'rmse': rmse})

	# make adversarial example
	adv_X, _ = fgsm(X =test_X, Y=test_y, model=model ,loss_fn = rmse , epsilon=0.2)

	# make a adv prediction
	adv_yhat = model.predict(adv_X)
	adv_X = adv_X.reshape((adv_X.shape[0], 7))
	# make a prediction
	yhat = model.predict(test_X)
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