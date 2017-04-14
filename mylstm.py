from __future__ import division, print_function, absolute_import
import numpy as np
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import sklearn
from sklearn.model_selection import train_test_split
import mydata as data

def build_model(max_features, maxlen):
		net = tflearn.input_data(shape=[None,73])
		print('Here')
		
		net = tflearn.embedding(net, input_dim=16, output_dim=128)
		print('Here1')
		net = tflearn.lstm(net, 128, dropout=0.8)
		print('Here2')
		net = tflearn.fully_connected(net, 16, activation='softmax')
		print('Here3')
		net = tflearn.regression(net, optimizer='rmsprop', learning_rate=0.001,
		loss='categorical_crossentropy')
		print('Here4')
		# Training
		model = tflearn.DNN(net, tensorboard_verbose=0)
		print('Here5')
		return model


def run(max_epoch=25, nfolds=10, batch_size=16):
		"""Run train/test on logistic regression model"""
		indata = data.get_data()

		# Extract data and labels
		X = [x[1] for x in indata]
		labels = [x[0] for x in indata]

		# Generate a dictionary of valid characters
		valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}

		max_features = len(valid_chars) + 1
		maxlen = np.max([len(x) for x in X])
		X = X[:200]
		labels = labels[:200]

		# Convert characters to int and pad
		X = [[valid_chars[y] for y in x] for x in X]
		X = pad_sequences(X, maxlen=maxlen)

		# Convert labels to 0-1
		y = [0 if x == 'benign' else 1 for x in labels]

		X_train = X[:128]
		y_train = y[:128]
		X_test  = X[101:240]
		y_test  = y[101:240]
		print('X_train shape')
		print (X_train.shape)
		print ('Y_trina shape')
		print (len(y_train))



		print('Build model...')
		model = build_model(max_features, maxlen)

		print("Train...")
		model.fit(X_train, y_train, batch_size=batch_size, show_metric=True,n_epoch=3)
		print("Predicting")
		model.predict(X_test,y_test)
