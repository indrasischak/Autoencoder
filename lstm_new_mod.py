import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy import io
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('C:/Users/chak282/Downloads/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
aa=0
bb=0
for jj in range(0,20):
    # normalize the dataset
    if (jj-1)>0:
        dataset[len(dataset)-1]=bb
        dataset[len(dataset)-2]=aa 
        dataset=numpy.insert(dataset,len(dataset),0)
        dataset=numpy.insert(dataset,len(dataset),0)
        dataset=dataset.reshape(len(dataset),1)
    else:
         dataset[len(dataset)-1]=bb
         dataset[len(dataset)-2]=aa  
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
# split into train and test sets
    train_size = int(len(dataset)-10)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
    look_back = 3
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# create and fit the LSTM network
    batch_size = 1
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    model.add(Dense(1))
    optimizer=optimizers.Adam(lr=0.00005, epsilon=1e-08, decay=0)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    for i in range(10):
            model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
            model.reset_states()
# make predictions
    trainPredict = model.predict(trainX, batch_size=batch_size)
    model.reset_states()
    testPredict = model.predict(testX, batch_size=batch_size)
# invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    dataset = scaler.inverse_transform(dataset)
# calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    aa=testPredict[len(testPredict)-2,0]
    bb=testPredict[len(testPredict)-1,0]
    print(len(dataset))

data={'lstm':dataset}
io.savemat('lstm_data.mat',data)
