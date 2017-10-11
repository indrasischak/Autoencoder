## autoencoder


import numpy as np

from keras.layers import Input, Dense
from keras import regularizers
from keras.models import Model
from keras import optimizers
from scipy.stats import logistic
import data_gather_sin
import data_gather_sin1
import matplotlib.pyplot as plt
import xlwt
#mnist2=data_gather2.ggg()
#mnist1=data_gather.ggg()


# Create the model
def create_model(ndim, encoding_dim):
        
	# this is our input placeholder
	input_img = Input(shape = (ndim, ))
	
	# "encoded" is the encoded representation of the input
	encoded = Dense(4, activation = 'selu')(input_img)
	#encoded1 = Dense(6, activation='tanh')(encoded)
	#encoded2 = Dense(6, activation='tanh')(encoded1)
	#encoded3 = Dense(encoding_dim, activation='tanh')(encoded2)
        #encoded = Dense(64, activation='relu')(encoded)
        #encoded = Dense(32, activation='relu')(encoded)

        #decoded = Dense(128, activation='relu')(decoded)
        #decoded = Dense(784, activation='sigmoid')(decoded)
	# "decoded" is the lossy reconstruction of the input
	decoded=Dense(4, activation = 'tanh')(encoded)
	#decoded=Dense(4, activation = 'selu')(decoded1)
	#decoded = Dense(ndim, activation = 'relu')(decoded1)
	
	# this model maps an input to its reconstruction
	autoencoder = Model(input_img, decoded)
	
	# encoder
	encoder= Model(input_img, encoded)
	
	#decoder=Model(encoded,decoded)
        #encoder = Model(input_img, encoded)
	#encoder = Model(input_img, [encoded,encoded1])

	

        #autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())
        #ae.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch,
       #show_accuracy=False, verbose=1, validation_data=[X_test, X_test])

	
	# compile the autoencoder
	optimizer=optimizers.Adam(lr=0.05, epsilon=1e-08, decay=0)
	autoencoder.compile(loss = 'mean_squared_error', optimizer = optimizer)
	
	return (autoencoder,encoder)


# Train the model, iterating on the data in batches of batch_size samples
def train_model(autoencoder, x_train, batch_size, epochs):
	autoencoder.fit(x_train, x_train, batch_size = batch_size, epochs = epochs, shuffle = True)
	
	return autoencoder


def main():
	# parameters
	nsamples = 30000
	ndim = 4
	encoding_dim = 1
	batch_size = 30000
	epochs = 500
	
	# generate dummy data
	x_train=data_gather_sin1.ggg()
	
	# model
	autoencoder, encoder= create_model(ndim, encoding_dim)
	
	# train model
	autoencoder = train_model(autoencoder, x_train, batch_size, epochs)
	
	# encoded signal
	x_test=data_gather_sin.ggg()
	encodedsig = autoencoder.predict(x_test)
	encodedsig1=x_train
	encodedsig2=encoder.predict(x_train)
	encodedsig3=autoencoder.predict(x_train)
	encodedsig4=x_test
	

	encodedsig5 = autoencoder.get_weights()[0]
	#encodedsig5=np.vstack([autoencoder.get_weights()[0],autoencoder.get_weights()[1]])
	encodedsig6 = autoencoder.get_weights()[1]
	#encodedsig7=decoder.get_weights()[0]
	#encodedsig8=decoder.get_weights()[1]
	return(encodedsig,encodedsig1,encodedsig2,encodedsig3,encodedsig4,encodedsig5,encodedsig6)

x,y,z,w,v,a,b=main()
err=np.abs(x[:,2]-v[:,2])
max_err=np.max(err)/np.abs(np.max(v[:,2]))
print(max_err)
