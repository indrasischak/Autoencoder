import tensorflow as tf
import numpy as np
import math,random


n_steps=int(input("Input number of rows in training data:"))
n_inputs=int(input("Input number of inputs in training data:"))
n_neurons=30
n_layers=2
n_outputs=1

import xlwt
import xlrd
    

workbook = xlrd.open_workbook('Dymoladata_summer.xlsx')
    #workbook = xlrd.open_workbook('myfile.xls')
sheet1 = workbook.sheet_by_name('Dymoladata_summer')
y=np.zeros(shape=(sheet1.nrows,sheet1.ncols)) 
for index in range(0,sheet1.nrows):
    for index1 in range(0,sheet1.ncols):
        y[index,index1]=sheet1.cell_value(index,index1)


X=tf.placeholder(tf.float32,[None,n_inputs])
Y=tf.placeholder(tf.float32,[None,n_outputs])

#cell=tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,activation=tf.nn.relu)
layers=[tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons,activation=tf.nn.tanh) for layer in range(n_layers)]
multi_layer_cell=tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.MultiRNNCell(layers),output_size=n_outputs)


outputs,states=tf.nn.dynamic_rnn(multi_layer_cell,X,dtype=tf.float32)

learning_rate=0.001
loss=tf.reduce_mean(tf.square(outputs-Y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op=optimizer.minimize(loss)

init=tf.global_variables_initializer()

n_iterations=15000
batch_size=50
saver=tf.train.Saver()
from random import randint
with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        dum=[randint(0,sheet1.nrows-1) for p in range (0,batch_size-1)]
        X_batch=np.zeros(shape=(batch_size,n_inputs))
        y_batch=np.zeros(shape=(batch_size,1))
        for ii in range(0,n_inputs-1):
            for jj in range (0,batch_size-1):
                X_batch[jj,ii]=y[dum[jj],ii]
                y_batch[jj]=y[dum[jj],n_inputs]
        #y_batch=[row[n_inputs] for row in y]
        X_batch1 = np.array(X_batch).reshape(1,batch_size,n_inputs)
        y_batch1 = np.array(y_batch).reshape(1, batch_size,n_outputs)
        sess.run(training_op,feed_dict={X:X_batch1,Y:y_batch1})
        if iteration % 100==0:
             mse=loss.eval(feed_dict={X:X_batch1,Y:y_batch1})
             print(iteration,"\tMSE:",mse)
    save_path=saver.save(sess,"C:/Users/chak282/AppData/Local/Programs/Python/Python35/rnn_trial.ckpt")


    X_new1 = X_batch1
    y_pred=sess.run(outputs,feed_dict={X:X_new1})
    y_pred1=y_pred.reshape(n_steps,1)
   
