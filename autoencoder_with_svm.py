## autoencoder


import numpy as np

from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
import data_gather
import data_gather4
import matplotlib.pyplot as plt
import xlwt
#mnist2=data_gather2.ggg()
#mnist1=data_gather.ggg()


# Create the model
def create_model(ndim, encoding_dim):
	# this is our input placeholder
	input_img = Input(shape = (ndim, ))
	
	# "encoded" is the encoded representation of the input
	encoded = Dense(encoding_dim, activation = 'selu')(input_img)
	#encoded2=Dense(20, activation = 'tanh')(encoded1)
	#encoded3=Dense(200, activation = 'tanh')(encoded2)
	#encoded4=Dense(300, activation = 'tanh')(encoded3)
	#encoded5=Dense(400, activation = 'tanh')(encoded4)
	#encoded6=Dense(500, activation = 'tanh')(encoded5)
	#encoded1 = Dense(6, activation='tanh')(encoded)
	#encoded2 = Dense(6, activation='tanh')(encoded1)
	#encoded3 = Dense(encoding_dim, activation='tanh')(encoded2)
        #encoded = Dense(64, activation='relu')(encoded)
        #encoded = Dense(32, activation='relu')(encoded)

        #decoded = Dense(128, activation='relu')(decoded)
        #decoded = Dense(784, activation='sigmoid')(decoded)
	# "decoded" is the lossy reconstruction of the input
	#decoded1=Dense(500, activation = 'selu')(encoded6)
	#decoded2=Dense(400, activation = 'selu')(decoded1)
	#decoded3=Dense(300, activation = 'selu')(decoded2)
	#decoded4=Dense(200, activation = 'selu')(decoded3)
	#decoded5=Dense(100, activation = 'selu')(decoded4)
	#decoded6=Dense(50, activation = 'selu')(decoded5)
	decoded=Dense(ndim, activation = 'selu')(encoded)
	#decoded=Dense(4, activation = 'selu')(decoded1)
	#decoded = Dense(ndim, activation = 'relu')(decoded1)
	
	# this model maps an input to its reconstruction
	autoencoder = Model(input_img, decoded)
	
	# encoder
	encoder = Model(input_img, encoded)
	#encoder= Model(input_img, [encoded1,encoded2,encoded3,encoded4,encoded5,encoded6])
        
        #encoder = Model(input_img, [encoded,encoded1])
	
	# compile the autoencoder
	optimizer=optimizers.Adam(lr=0.0005, epsilon=1e-08, decay=0)
	autoencoder.compile(loss = 'mean_squared_error', optimizer = optimizer)
	
	return (autoencoder, encoder)


# Train the model, iterating on the data in batches of batch_size samples
def train_model(autoencoder, x_train, batch_size, epochs):
	autoencoder.fit(x_train, x_train, batch_size = batch_size, epochs = epochs, shuffle = True)
	
	return autoencoder


def main(encoding_dim):
	# parameters
	nsamples = 4321
	ndim = 52
	#encoding_dim = 200
	batch_size = 4321
	epochs = 1000
	
	# generate dummy data
	x_train=data_gather.ggg()
	
	# model
	autoencoder, encoder= create_model(ndim, encoding_dim)
	
	# train model
	autoencoder = train_model(autoencoder, x_train, batch_size, epochs)
	
	# encoded signal
	x_test=data_gather4.ggg()
	#encodedsig = encoder.predict(x_train)
	encodedsig = x_train
	#encodedsig1=encoder.predict(x_test)
	encodedsig1=x_test

	#encodedsig = autoencoder.predict(x_train)
	#encodedsig1=x_train
	return(encodedsig,encodedsig1)
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import NuSVC
import random
accur=np.zeros(20)
for iiii in range(1,2):
        x1,yy=main(52*iiii)

#for index in range(0,9):
 #       plt.plot(x[:,index])
  #      plt.plot(y[:,index])
   #     plt.show()


        X=np.concatenate((x1,yy))
        size=len(X);
        y=np.zeros(shape=[size,1])
        for index in range(0,size):
                if (4321-index)>=0:
                      y[index,0]=0
                else:
                        y[index,0]=1

        kf = KFold(n_splits=20)
        kf.get_n_splits(X)
        print(kf)

        KFold(n_splits=20, random_state=None, shuffle=False)
        ii=0
        dum=np.zeros(len(X))
        for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf=NuSVC()
                clf.fit(X_train,y_train)
                NuSVC(cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape=None, degree=1, gamma='auto', kernel='rbf',
                  max_iter=-1, nu=0.5, probability=False, random_state=None,
                  shrinking=True, tol=0.001, verbose=False)
                y_predict=clf.predict(X_test)
                for i in range(0,len(y_predict)-1):
                    if y_predict[i]==y_test[i]:
                        dum[ii]=1
                        ii=ii+1
                    else:
                        dum[ii]=0
                        ii=ii+1
        print("Accuracy is")
        print((np.sum(dum)/len(X))*100,"%")
        accur[iiii]=np.sum(dum)/len(X)
print(accur)
        
        
