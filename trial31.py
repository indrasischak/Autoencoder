import tensorflow as tf
import numpy as np
import pandas as pd
import time
import pickle
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

def movingaverage (values, window):
    '''Simple Moving Average low pass filter
    '''
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

print(tf.__version__)

class MultilayerPerceptron:
    '''Class to define Multilayer Perceptron Neural Networks architectures such as 
       Autoencoder in Tensorflow'''
    def __init__(self, layersize, activation, learning_rate=0.0001, summaries_dir='C:/'):
        ''' Generate a multilayer perceptron network according to the specification and 
            initialize the computational graph.'''
        assert(len(layersize)-1==len(activation), 
		    'Activation function list must be one less than number of layers.')
        # Reset default graph
        ops.reset_default_graph()
        # Capture parameters
        self.learning_rate=learning_rate
        self.summaries_dir=summaries_dir
        # Define the computation graph for an Autoencoder
        with tf.name_scope('inputs'):
            self.X = tf.placeholder("float", [None, layersize[0]])
        inputs = self.X
        # Iterate through specification to generate the multilayer perceptron network
        for i in range(len(layersize)-1):
            with tf.name_scope('layer_'+str(i)):
                n_input = layersize[i]
                n_hidden_layer = layersize[i+1]
                # Init weights and biases
                weights = tf.Variable(tf.random_normal([n_input, n_hidden_layer])*0.001, 
				name='weights')
                biases  = tf.Variable(tf.random_normal([n_hidden_layer])*0.001, name='biases')
                # Create layer with weights, biases and given activation
                layer = tf.add(tf.matmul(inputs, weights), biases)
                tf.summary.histogram('pre-activation-'+activation[i].__name__, layer)
                layer = activation[i](layer)
                # Current outputs are the input for the next layer
                inputs = layer
                tf.summary.histogram('post-activation-'+activation[i].__name__, inputs)
        self.nn = layer
        # Define loss and optimizer
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.nn, self.X)))
            tf.summary.scalar("training_loss", self.loss)
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(
			learning_rate=self.learning_rate).minimize(self.loss)
        with tf.name_scope('anomalyscore'):
            self.anomalyscore = tf.reduce_mean(tf.abs(tf.subtract(self.nn, self.X)), 1)
        with tf.name_scope('crossentropy'):
            self.xentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			                                        logits=self.nn, labels=self.X))
            tf.summary.scalar("cross_entropy", self.xentropy)
        # Init session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        # Configure logs
        tf.gfile.MakeDirs(summaries_dir)
        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.summaries_dir, self.sess.graph)
        variables_names =[v.name for v in tf.trainable_variables()]
        values = self.sess.run(variables_names)
        for k,v in zip(variables_names, values):
            print(k, v)
            v1=v

    def train(self, data, nsteps=10000): 
        ''' Train the Neural Network using to the data.'''
        lossdev = []
        scoredev = []
        # Training cycle
        steps = np.linspace(0,len(data), num=nsteps, dtype=np.int)
        for it,(step1,step2) in enumerate(zip(steps[0:-1],steps[1:])):
            c = self.sess.run([self.optimizer, self.loss], 
				feed_dict={self.X: data[step1:step2,:]})  
            s = self.sess.run(self.nn, 
				feed_dict={self.X: data[step1:step1+1,:]})
            l,ts = self.sess.run([self.loss, self.merged], 
				feed_dict={self.X: data[step1:step1+1,:]})
            scoredev.append(s[0])
            lossdev.append(l)
            print('.', end='')
            self.summary_writer.add_summary(ts, it)
        print
        return lossdev

    def predict(self, data):
        '''Predict outcome for data'''
        return self.sess.run(self.nn, feed_dict={self.X: data})
        
    def score(self, data):
        '''Compute anomaly score based on reconstruction error.'''
        return self.sess.run(self.anomalyscore, feed_dict={self.X: data})

# Define the size of the neural network
n_input =3
n_hidden_layer = 4

N = 20
sigma = 0.1

import xlwt
import xlrd
import math
import csv
import random
import numpy as np
    

workbook = xlrd.open_workbook('tryy1.xlsx')
    #workbook = xlrd.open_workbook('myfile.xls')
sheet1 = workbook.sheet_by_name('tryy')
traindata = np.zeros(shape=[sheet1.nrows,n_input])
testdata_n = np.zeros(shape=[sheet1.nrows,n_input])
testdata_p = np.zeros(shape=[sheet1.nrows,n_input])
for index1 in range(0,sheet1.nrows):
    traindata[index1,2]=sheet1.cell_value(index1,2)
    traindata[index1,0]=sheet1.cell_value(index1,0)
    traindata[index1,1]=sheet1.cell_value(index1,1)
    #traindata[index1,3]=sheet1.cell_value(index1,21)
    #testdata_n[index1,1]=sheet1.cell_value(index1,4)
    #testdata_n[index1,0]=sheet1.cell_value(index1,3)
    #testdata_n[index1,2]=sheet1.cell_value(index1,5)
    #testdata_n[index1,3]=sheet1.cell_value(index1,20)
    #testdata_p[index1,1]=sheet1.cell_value(index1,4)
    #testdata_p[index1,0]=sheet1.cell_value(index1,3)
    #testdata_p[index1,2]=sheet1.cell_value(index1,6)
    #testdata_p[index1,3]=sheet1.cell_value(index1,20)

data=traindata
testdata_n=traindata
testdata_p=traindata
#data = np.random.normal(size=(N,n_input))
#mux = np.reshape(np.repeat(np.arange(n_input)+1,N),(N,n_input), order='F')
#sigx = np.reshape(np.repeat(sigma, n_input*N), (N,n_input))
#traindata = (data*sigx+mux)/float(n_input)
#traindata=np.transpose(np.sin(np.linspace(0,10,100)))

#data = np.random.normal(size=(N,n_input))
#sigx = np.reshape(np.repeat(sigma*2., n_input*N), (N,n_input))
#testdata_n = (data*sigx+mux)/float(n_input)
#testdata_n=np.transpose(np.sin(np.linspace(0,10,100)+0.5))


#data = np.random.poisson(size=(N,n_input))
#sigx = np.reshape(np.repeat(sigma, n_input*N), (N,n_input))
#testdata_p = (data*sigx+mux)/float(n_input)
#testdata_p=np.transpose(np.sin(np.linspace(0,10,100)))
#testdata_p=traindata

plt.figure(figsize=(15,6))
#plt.plot(traindata[:,:])
plt.grid(True)
plt.xlabel('Training data item id') ; plt.ylabel('Values approximate signal indices')
plt.title('First 200 rows of training data') ;
plt.show()

# Build autoencoder neural network graph
autoencoder = MultilayerPerceptron([n_input, n_hidden_layer, n_input], [tf.nn.tanh, tf.nn.sigmoid])

# Launch the graph
lossdev = autoencoder.train(traindata)
scoretrain = autoencoder.score(traindata)
scoretest_n = autoencoder.score(testdata_n)
scoretest_p = autoencoder.score(testdata_p)
gg=autoencoder.predict(testdata_n)
hh=autoencoder.predict(traindata)
hh1=autoencoder.predict(testdata_p)
plt.plot(lossdev, label='loss')
#plt.plot(scoretest_n, label='n')
#plt.plot(scoretest_p, label='p')
#plt.plot(scoredev, label='scoredev')
plt.grid(True)
plt.legend()
f, (axn, axp) = plt.subplots(1,2, sharey=True, figsize=(14,4))
axn.plot(movingaverage(scoretrain,10), label='train')
axn.plot(movingaverage(scoretest_n,10), label='test')
axp.plot(movingaverage(scoretrain,10), label='train')
axp.plot(movingaverage(scoretest_p,10), label='test', color='gold')
axn.grid(True) ; axn.legend() ; axp.grid(True) ; axp.legend()
axn.set_title('train vs. normal dist w. $2*\sigma$')
axp.set_title('train vs. poisson dist')


f, (axn, axp) = plt.subplots(1,2, sharey=True, figsize=(14,4))
axn.plot(sorted(movingaverage(scoretest_n,10)), label='test', color='orange', linewidth=4.)
axp.plot(sorted(movingaverage(scoretest_p,10)), label='test', color='gold', linewidth=4.)
axn.grid(True) ; axp.grid(True)
axn.set_title('normal dist w. $2*\sigma$') ; axp.set_title('poisson dist')
plt.show()


