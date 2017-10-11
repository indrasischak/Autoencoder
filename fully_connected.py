#!/usr/bin/env python
from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data_gather_sin
import data_gather_sin1
from random import randint
#import data_gather2
#mnist2=data_gather2.ggg()
#mnist1=data_gather_sin.ggg()
mnist=data_gather_sin1.ggg()

def get_target_result(x):
    return np.sin(x)


def multilayer_perceptron(x, weights, biases):
    """Create model."""
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.tanh(layer_2)

    # Output layer with linear activation
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    out_layer = tf.nn.tanh(out_layer)
    return out_layer

# Parameters
#learning_rate = 0.01
training_epochs = 500
batch_size = 30000
display_step = 500
total_train_row=2000
batch_num=np.int(total_train_row/batch_size)

# Network Parameters
n_hidden_1 = 10  # 1st layer number of features
n_hidden_2 = 10  # 2nd layer number of features
n_input = 1


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden_2, 1], stddev=0.1))
}

biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden_1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden_2])),
    'out': tf.Variable(tf.constant(0.1, shape=[1]))
}

x_data = tf.placeholder(tf.float32, [None, 1])
y_data = tf.placeholder(tf.float32, [None, 1])

# Construct model
pred = multilayer_perceptron(x_data, weights, biases)

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(pred - y_data))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# train = optimizer.minimize(loss)
train = tf.train.AdamOptimizer(0.01).minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)
mnistx=mnist[:,0]
mnistx=mnistx.reshape([total_train_row,1])
mnisty=mnist[:,1]
mnisty=mnisty.reshape([total_train_row,1])

for step in range(training_epochs):
    kf = KFold(n_splits=10)
    kf.get_n_splits(mnistx)
    print(kf)
    KFold(n_splits=10, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(mnistx):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = mnistx[train_index],mnistx[test_index]
        y_train, y_test = mnisty[train_index], mnisty[test_index]
        sess.run(train, feed_dict={x_data: X_train, y_data: y_train})

        z= sess.run(pred, feed_dict={x_data: X_test})
        curLoss = sess.run(loss, feed_dict={x_data: X_test, y_data: y_test})
        #print(curLoss)
z= sess.run(pred, feed_dict={x_data: mnistx})
plt.plot(z-mnisty)
plt.show()
