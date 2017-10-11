from sklearn.model_selection import KFold
import tensorflow as tf
from scipy import io
import numpy as np
    #import data_gather_sin1_auto
from random import randint
import xlwt
import xlrd
import math
import csv
import random
import numpy as np
num_derv=30
num_input=52
workbook = xlrd.open_workbook('derv_data1.xlsx')
    #workbook = xlrd.open_workbook('myfile.xls')
sheet1 = workbook.sheet_by_name('derv_data1')
total_train_row=4320
training_epochs=10000
batch=500
num_batch=np.int(total_train_row/batch)
traindata = np.zeros(shape=[total_train_row,num_input+num_derv])
#testdata_n = np.zeros(shape=[10000,num_input])
#testdata_p = np.zeros(shape=[sheet1.nrows,4])
for index1 in range(0,total_train_row):
    for index in range(0,num_input+num_derv):
        traindata[index1,index]=sheet1.cell_value(index1,index)
        #traindata[index1,5]=sheet1.cell_value(index1,5)
mnist=traindata
    #mnist1=traindata
    #np.random.shuffle(mnist1)
    #np.random.shuffle(mnist)
    #mnist=data_gather_sin1_auto.ggg()
def multilayer_perceptron(x, weights, biases,i):
    # Hidden layer with RELU activation
    
    layer_1 = tf.add(tf.matmul(x, weights['h'+str(i)]), biases['b'+str(i)])
    layer_1 = tf.nn.sigmoid(layer_1)
    # # Hidden layer with RELU activation
        #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        #layer_2 = tf.nn.tanh(layer_2)
        #layer_2=tf.nn.dropout(layer_2,0.5)

        #layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        #layer_3 = tf.nn.elu(layer_3)
        #layer_3=tf.nn.dropout(layer_3,0.5)

        #layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
        #layer_4 = tf.nn.elu(layer_4)
        #layer_4=tf.nn.dropout(layer_4,0.5)

        #layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        #layer_2 = tf.nn.tanh(layer_3)
    # Output layer with linear activation
    out_layer = tf.add(tf.matmul(layer_1, weights['out'+str(i)]), biases['out'+str(i)])
    
    return out_layer

# Parameters
#learning_rate = 0.01

x_data = tf.placeholder(tf.float32, [None, num_input])
y_data = tf.placeholder(tf.float32, [None, 1])
    
# Network Parameters
n_hidden_1 = 2*num_input  # 1st layer number of features
    #n_hidden_2 = 400  # 2nd layer number of features
n_input = num_input

for ijjjj in range(16,16+1):
    mnistx=mnist[:,num_derv:num_input+num_derv]
    mnistx=mnistx.reshape([total_train_row,num_input])
    mnisty=mnist[:,num_derv-ijjjj]
    mnisty=mnisty.reshape([total_train_row,1])
# Store layers weight & bias
    weights = {
    'h'+str(ijjjj): tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
    'out'+str(ijjjj): tf.Variable(tf.truncated_normal([n_hidden_1, 1], stddev=0.1))
    }

    biases = {
    'b'+str(ijjjj): tf.Variable(tf.constant(0.1, shape=[n_hidden_1])),
    'out'+str(ijjjj): tf.Variable(tf.constant(0.1, shape=[1]))
    }




# Construct model
   

    
    pred = multilayer_perceptron(x_data, weights, biases,ijjjj)

# Minimize the mean squared errors.
    #loss = tf.nn.l2_loss(y_data - pred)
    loss=tf.reduce_mean(tf.squared_difference(y_data, pred))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# train = optimizer.minimize(loss)
#loss=loss1[1]+loss1[2]+loss1[3]
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.001, global_step,
                                       100, 10^-6, staircase=True)
    train = tf.train.AdamOptimizer(learning_rate,0.9).minimize(loss)


    init = tf.initialize_all_variables()

# Launch the graph.
    sess = tf.Session()
    sess.run(init)
    #mnistx=mnist[:,0:5]
    #mnistx=np.delete(mnist,number,1)
    #mnistx=mnistx.reshape([total_train_row,24])
    #mnisty=mnist[:,number]
    #mnisty=mnisty.reshape([total_train_row,1])
    curLoss=5
    for step in range(training_epochs):
        if curLoss>=0.00005:
            kf = KFold(n_splits=10)
            kf.get_n_splits(mnistx)
           #print(kf)
            KFold(n_splits=10, random_state=None, shuffle=False)
            for train_index, test_index in kf.split(mnistx):
                rand_num=randint(1,num_batch)
        #print("TRAIN:", train_index, "TEST:", test_index)
               # X_train, X_test = mnistx[train_index],mnistx[test_index]
               # y_train, y_test = mnisty[train_index], mnisty[test_index]
                trainx=mnist[batch*(rand_num-1):batch*(rand_num),num_derv:num_input+num_derv]
                trainx=trainx.reshape([batch,num_input])
                trainy=mnist[batch*(rand_num-1):batch*(rand_num),num_derv-ijjjj]
            #trainy=mnist[:,1]
                trainy=trainy.reshape([batch,1])
                X_train, X_test = mnistx[train_index],mnistx[test_index]
                y_train, y_test = mnisty[train_index], mnisty[test_index]
                sess.run(train, feed_dict={x_data: X_train, y_data: y_train})
                curLoss1 = sess.run(loss, feed_dict={x_data: X_test, y_data: y_test})

            #trainx=mnist[batch*(rand_num-1):batch*(rand_num),:]
            #trainx=np.delete(trainx,2,1)
            #trainx=trainx.reshape([batch,num_input-1])
            #trainy=mnist[batch*(rand_num-1):batch*(rand_num),2]
            #trainy=mnist[:,2]
            #trainy=trainy.reshape([batch,1])
            #sess.run(train, feed_dict={x_data: trainx, y_data: trainy})
            #curLoss2 = sess.run(loss, feed_dict={x_data: trainx, y_data: trainy})

            #z= sess.run(pred, feed_dict={x_data: mnist})
                curLoss = curLoss1
                #print(train_index)
            print('Epoch',step)
            #print(curLoss)

        else:
            break
        print(curLoss)
        
    z= sess.run(pred, feed_dict={x_data: mnistx})
    w1=sess.run(weights['h'+str(ijjjj)])
    w2=sess.run(weights['out'+str(ijjjj)])
    b1=sess.run(biases['b'+str(ijjjj)])
    b2=sess.run(biases['out'+str(ijjjj)])
    error=z-mnisty
    data={'W1':w1,'W2':w2,'b1':b1,'b2':b2,'error':error}
    string='data_save'+str(ijjjj)
    io.savemat(string,data)
    #error1=z-mnisty
    
