## x -> xsinx


import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD


# Generate dummy data
def generate_data(nsamples, ndim):
	x_train = (np.random.random((nsamples, ndim)) - 0.5) * (5 * np.pi)
	y_train = x_train * np.sin(x_train)
	
	return x_train, y_train


# Create the model
def create_model(ndim, encoding_dim):
	# this is our input placeholder
	input_sig = Input(shape = (ndim, ))
	
	# hidden layers
	hidden_0 = Dense(encoding_dim, activation = 'sigmoid', kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros')(input_sig)
	
	# output layer
	output = Dense(ndim)(hidden_0)
	
	# model
	model = Model(input_sig, output)
	model.summary()
	
	return model


# Train the model, iterating on the data in batches of batch_size samples
def train_model(model, x_train, y_train, batch_size, epochs):
	# compile the model
	sgd = SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov = True)
	model.compile(loss = 'mean_squared_error', optimizer = sgd)
	
	# train the model
	model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, shuffle = True)
	
	return model


# Test the model
def predict_outputs(model, inputs):
	# dimensions
	nsamples, ndim = inputs.shape
	
	# initialize outputs
	outputs = np.zeros((nsamples, ndim), dtype = np.float32)
	
	# predict the outputs
	for i, input in enumerate(inputs):
		outputs[i] = model.predict(input)
	
	return outputs



	# parameters
nsamples = 10000
ndim = 1
encoding_dim = 50
batch_size = 100
epochs = 10000
	
	# generate dummy data
x_train, y_train = generate_data(nsamples, ndim)
	
	# model
model = create_model(ndim, encoding_dim)
	
	# train model
model = train_model(model, x_train, y_train, batch_size, epochs)
	
	# predict
y_predict = predict_outputs(model, x_train)
	
	# plot
plt.plot(x_train, y_train, '.g')
plt.plot(x_train, y_predict, '.r')
plt.savefig('predictions.png')



