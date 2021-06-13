import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import Sequential, Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from keras import backend as K
from keras.utils import plot_model

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split 

import random
import numpy as np
import datetime
import pandas as pd

import os
import glob

# Hyperparameters for model
g_batch_size = 20
g_epochs = 150

# for layers
g_drop = 0.1
g_units = 100

# for lstm
g_window = 15
g_layers = 6


def normalize_data(lst):
	return lst / np.max(lst)


def accumulate_data(lst):
	accumulate_list = []
	for val in lst:
		if len(accumulate_list) > 0:
			# radiation = accumulate_list[-1] * 0.001
			accumulate_list.append(val + accumulate_list[-1])
		else:
			accumulate_list.append(val)

	return accumulate_list


def load_data_set(path):
	""" load data from csv """
	# load data
	df = pd.read_csv(path)
	current_accum_list = accumulate_data(df["current"])
	df["accum"] = current_accum_list
	X = np.array(df[["current", "accum"]])
	y = np.array(df[["T"]])

	# Scale data
	s_x = MinMaxScaler()
	Xs = s_x.fit_transform(X)

	s_y = MinMaxScaler()
	ys = s_y.fit_transform(y)

	# split time step
	X_lstm = []
	y_lstm = []
	for i in range(g_window, len(df)):
		X_lstm.append(Xs[i - g_window:i])
		y_lstm.append(ys[i])

	return X_lstm, y_lstm, df, X, y


def load_data_by_label(path, label):
	df = pd.read_csv(path)
	return np.array(df[label])


def build_model(Xtrain):
	""" build model @ model.compile """
	model = Sequential()

	if g_layers == 1:
		model.add(LSTM(units=g_units,
						input_shape=(Xtrain.shape[1],Xtrain.shape[2])
						)
					)
		model.add(Dropout(rate=g_drop))
	else:
		# First layer specifies input_shape and returns sequences
		model.add(LSTM(units=g_units, 
						return_sequences=True, 
						input_shape=(Xtrain.shape[1],Xtrain.shape[2])
						)
					)
		model.add(Dropout(rate=g_drop))
		# Middle layers return sequences
		for i in range(g_layers-2):
			model.add(LSTM(units=g_units,return_sequences=True))
			model.add(Dense(units=g_units))
			model.add(Dropout(rate=g_drop))
		# Last layer doesn't return anything
		model.add(LSTM(units=g_units))
		model.add(Dropout(rate=g_drop))

		model.add(Dense(1))

	print(model.summary())

	optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
	model.compile(loss='mean_squared_error',
					optimizer=optimizer,
					metrics=['accuracy'])
	return model


def plot_hist(history, path, file_name):
	# Plot training & validation accuracy values
	plt.plot(history.history['loss'], label="loss")
	plt.plot(history.history['val_loss'], label="val_loss")
	plt.title('Model History')
	plt.ylabel('loss')
	plt.xlabel('Epoch')
	plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0))

	plt.savefig(path + file_name, bbox_inches='tight')
	plt.clf()
	plt.close()


def fit_model(model, model_path, x_train, t_train, x_val, t_val, batch_size=10, epochs=100, log_dir_base="/logs"):
	""" fit model & save """
	# callbacks
	log_dir = log_dir_base + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	tb = TensorBoard(log_dir=log_dir, histogram_freq=1)
	es = EarlyStopping(monitor="loss", patience=100, verbose=1)

	# to fit
	hist = model.fit(x = x_train,
						y = t_train,
						batch_size=batch_size,
						epochs=epochs,
						verbose=1,
						validation_data=(x_val, t_val),
						callbacks=[es, tb])

	# save
	hist_file_name = "/hist_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
	plot_hist(hist, model_path, hist_file_name)
	model.save(model_path)


def pred_model(model_path, valid_data_path):
	""" prediction model & fig save """
	model = keras.models.load_model(model_path)

	X_lstm, y_lstm, df, X, y = load_data_set(data_path)

	s_x = MinMaxScaler()
	Xs = s_x.fit_transform(X)

	s_y = MinMaxScaler()
	ys = s_y.fit_transform(y)

	Xtest = np.array(X_lstm)
	ytest = np.array(y_lstm)

	# Predict using LSTM
	print("==========================================")
	yp_s = model.predict(x=np.array(X_lstm))
	print("predict done.")
	print("==========================================")

	# Unscale data
	yp = s_y.inverse_transform(yp_s)

	time_list = df["time"]
	current_list = df["current"]

	# plot
	fig, ax1 = plt.subplots( )
	ax2 = ax1.twinx()

	ax1.plot(time_list, current_list, "k--", label="current")
	ax1.set_xlabel("Time[s]")
	ax1.set_ylabel("Current[A]")

	ax2.plot(yp, "g-", label="pred[degC]")
	ax2.plot(time_list, y, "r-", label="real[degC]")
	ax2.set_ylabel("$(^oC)$")

	handler1, label1 = ax1.get_legend_handles_labels()
	handler2, label2 = ax2.get_legend_handles_labels()

	plt.legend(handler1+handler2, label1+label2, loc='upper left', bbox_to_anchor=(1.1, 1.0))
	plt.tight_layout()
	plt.grid()

	basename_without_ext = os.path.splitext(os.path.basename(valid_data_path))[0]
	plt.savefig(model_path + "/pred_" + basename_without_ext + ".png", bbox_inches='tight')
	plt.clf()
	plt.close()



def fit_main(data_path, model_path="model"):
	""" do fit sequence """
	# 1. preparation
	x, t, df, _, _ = load_data_set(data_path)
	x_train, x_val, t_train, t_val = \
		train_test_split(x, t, test_size=0.01, shuffle=False)
	x_train = np.array(x_train)
	x_val = np.array(x_val)
	t_train = np.array(t_train)
	t_val = np.array(t_val)
	print(x_train.shape, x_val.shape)

	# 2. build model
	try:
		model = keras.models.load_model(model_path)
	except OSError:
		# assignment not exist path
		os.makedirs(model_path, exist_ok=True)
		model = build_model(x_train)
		plot_model(model, to_file= model_path + '/model.png', show_shapes=True, show_layer_names=True)


	# 3. fit model
	fit_model(model, model_path, x_train, t_train, x_val, t_val,\
				batch_size=g_batch_size, epochs=g_epochs)

	# 4. predict model
	pred_model(model_path, data_path)


def pred_main(data_path, model_path="model"):
	""" do prediction sequence """
	pred_model(model_path, data_path)


if __name__=="__main__":
	# for reappearance result
	tf.random.set_seed(123)

	model_path = "./model_lstm"

	# one data
	data_path = "./data/MockDataDcr/40degC_1.71_75_A_150rpm_MC.csv"
	fit_main(data_path, model_path)

	data_path = "./data/MockDataDcr/40degC_2.28_100_A_150rpm_MC.csv"
	pred_main(data_path, model_path)

	# data_path_list = glob.glob("./data/MockData/*.csv")
	# for data_path in data_path_list:
	# 	fit_main(data_path, model_path)
