import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
import tensorflow.keras.backend as K

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def power_data(df):
    ## finding all columns that have nan:
    droping_list_all=[]
    for j in range(0,7):
        if not df.iloc[:, j].notnull().all():
            droping_list_all.append(j)
            #print(df.iloc[:,j].unique())

    # filling nan with mean in any columns
    for j in range(0,7):
            df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())

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

    # split into train and test sets
    values = reframed.values

    n_train_time = 365*72
    train = values[:n_train_time, :]
    test = values[n_train_time:, :]
    ##test = values[n_train_time:n_test_time, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]


    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    # We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].

    return train_X, train_y, test_X, test_y , scaler


def compute_gradient(model_fn, loss_fn, x, y, targeted):
    """
    cleverhans : https://github.com/cleverhans-lab/cleverhans/blob/1115738a3f31368d73898c5bdd561a85a7c1c741/cleverhans/tf2/utils.py#L171

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


def fgsm(X, Y, model, loss_fn , epsilon, targeted= False):
    ten_X = tf.convert_to_tensor(X)
    grad = compute_gradient(model,loss_fn,ten_X,Y,targeted)
    dir = np.sign(grad)
    return X + epsilon * dir, Y


def bim(X, Y, model, loss_fn, epsilon, alpha, I, targeted= False):
    Xp= np.zeros_like(X)
    for t in range(I):
        ten_X = tf.convert_to_tensor(X)
        grad = compute_gradient(model,loss_fn,ten_X,Y,targeted)
        dir = np.sign(grad)
        Xp = Xp + alpha * dir
        Xp = np.where(Xp > X+epsilon, X+epsilon, Xp)
        Xp = np.where(Xp < X-epsilon, X-epsilon, Xp)
    return Xp, Y