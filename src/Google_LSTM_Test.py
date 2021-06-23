import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
#matplotlib inline

from utils import google_data , rmse, fgsm

from tensorflow.keras.models import load_model


# define path to save model
model_path = '../Trained models/Google_regression_LSTM.h5'

train_X, train_y, test_X, test_y , scaler = google_data()

# if best iteration's model was saved then load and use it
if os.path.isfile(model_path):
    model = load_model(model_path, custom_objects={'rmse': rmse})

    # make adversarial example
    adv_X, _ = fgsm(X =test_X, Y=test_y, model= model ,loss_fn = rmse , epsilon=0.2)

    # make a adv prediction
    adv_yhat =  model.predict(adv_X,verbose=1, batch_size=200)
    # make a prediction
    yhat =  model.predict(test_X,verbose=1, batch_size=200)

    test_X = test_X[:,-1,:1]
    # invert scaling for adv forecast
    inv_adv_yhat = np.concatenate((adv_yhat, test_X), axis=1)
    inv_adv_yhat = scaler[1].inverse_transform(inv_adv_yhat)
    inv_adv_yhat = inv_adv_yhat[:,0]

    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X), axis=1)
    inv_yhat = scaler[1].inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X), axis=1)
    inv_y = scaler[1].inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    # calculate adv RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_adv_yhat))
    print('Test adv RMSE: %.3f' % rmse)


    fig_verify = plt.figure(figsize=(100, 50))
    plt.plot(inv_y[:200], marker='.', label="actual")
    plt.plot(inv_yhat[:200], 'r', label="prediction")
    plt.plot(inv_adv_yhat[:200], 'g', label="adv prediction")
    plt.title("Opening price of stocks sold")
    plt.xlabel("Time (latest-> oldest)")
    plt.ylabel("Stock Opening Price")
    plt.legend(fontsize=15)
    plt.show()