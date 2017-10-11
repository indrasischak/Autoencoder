from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import data_gather_sin
import data_gather_sin1
#import data_gather2
#mnist2=data_gather2.ggg()
mnist1=data_gather_sin.ggg()
mnist=data_gather_sin1.ggg()

# Parameters
learning_rate = 0.05
training_epochs = 5000
batch_size = 30000
display_step = 1
examples_to_show = 10
total_data_row=30000
total_batch=np.floor(total_data_row/batch_size)


# Network Parameters
n_hidden_1 = 4 # 1st layer num features
n_hidden_2 =4 # 2nd layer num features
n_input = 4 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b1': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.relu6(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    #layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
     #                              biases['encoder_b2']))
    return layer_1


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    #layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
     #                              biases['decoder_b2']))
    return layer_1

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost=tf.reduce_mean(tf.squared_difference(y_true, y_pred))
#cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
#step = tf.Variable(0, trainable=False)
#rate = tf.train.exponential_decay(0.0001, 10, 1, 0.000001)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    #total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(0,int(total_batch)):
            #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            #dum=np.random.random_integers(total_data_row-1,size=batch_size)
            #batch_xs=np.zeros(shape=[batch_size,n_input])
            #for j in range(0,batch_size-1):
             #   batch_xs[j,:]=mnist[dum[j],:]
            
            _, c = sess.run([optimizer, cost], feed_dict={X: mnist})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")
    xx1=sess.run(weights["encoder_h1"])
    xx2=sess.run(biases["encoder_b1"])
    #encode=np.tanh(
    # Applying encode and decode over test set
    
    z= sess.run(
        y_pred, feed_dict={X: mnist1})
    
    # Compare original images with their reconstructions
    
