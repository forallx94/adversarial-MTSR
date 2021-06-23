from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, Dense, Flatten, InputLayer, Activation, Dropout

# train_X.shape  ['-',1,7]

def setup_cnn_model(train_X,kernel_size):
    cnn = Sequential()
    cnn.add(InputLayer(input_shape=(train_X.shape[1], train_X.shape[2])))

    cnn.add(Conv1D(filters=60,
                kernel_size=kernel_size,
                padding="valid",
                activation="relu")
        )

    cnn.add(Conv1D(filters=60,
                kernel_size=kernel_size,
                padding="valid",
                activation="relu")
        )

    cnn.add(Conv1D(filters=60,
                kernel_size=kernel_size,
                padding="valid",
                activation="relu")
        )

    cnn.add(Flatten())
    cnn.add(Dense(50, activation='relu'))
    cnn.add(Dense(units=1))
    cnn.add(Activation("relu"))
    return cnn


def setup_gru_model(train_X):
	model = Sequential()
	model.add(GRU(
			input_shape=(train_X.shape[1], train_X.shape[2]),
			units=100,
			return_sequences=True))
	model.add(Dropout(0.2))
	model.add(GRU(
			units=100,
			return_sequences=True))
	model.add(Dropout(0.2))
	model.add(GRU(
			units=100,
			return_sequences=False))
	model.add(Dropout(0.2))
	model.add(Dense(units=1))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=[rmse])
	return model


def setup_lstm_model(train_X):
	model = Sequential()
	model.add(LSTM(
			input_shape=(train_X.shape[1], train_X.shape[2]),
			units=100,
			return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(
			units=100,
			return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(
			units=100,
			return_sequences=False))
	model.add(Dropout(0.2))
	model.add(Dense(units=1))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=[rmse])
	return model