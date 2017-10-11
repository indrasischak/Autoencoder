def ggg_auto(number):
    #from sklearn.model_selection import KFold
    import tensorflow as tf
    import numpy as np
    #import data_gather_sin1_auto
    from random import randint
    import xlwt
    import xlrd
    import math
    import csv
    import random
    import numpy as np
    workbook = xlrd.open_workbook('derv_data1.xlsx')
    #workbook = xlrd.open_workbook('myfile.xls')
    sheet1 = workbook.sheet_by_name('derv_data1')
    total_train_row=10000
    training_epochs=10000
    batch=500
    num_batch=np.int(total_train_row/batch)
    traindata = np.zeros(shape=[10000,53])
    testdata_n = np.zeros(shape=[10000,53])
    testdata_p = np.zeros(shape=[sheet1.nrows,4])
    for index1 in range(0,10000):
        for index in range(0,53):
            traindata[index1,index]=sheet1.cell_value(index1,index)
        
        #traindata[index1,5]=sheet1.cell_value(index1,5)
    mnist=traindata
    #mnist1=traindata
    #np.random.shuffle(mnist1)
    #np.random.shuffle(mnist)
    #mnist=data_gather_sin1_auto.ggg()
    def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
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
        out_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])
        #out_layer = tf.nn.elu(out_layer)
        #out_layer=tf.nn.dropout(out_layer,0.5)
        return out_layer

# Parameters
#learning_rate = 0.01


# Network Parameters
    n_hidden_1 = 100  # 1st layer number of features
    #n_hidden_2 = 400  # 2nd layer number of features
    n_input = 52


# Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
        #'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        #'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_2], stddev=0.1)),
        #'h4': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_2], stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([n_hidden_1, 1], stddev=0.1))
    }

    biases = {
        'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden_1])),
        #'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        #'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden_2])),
        #'b4': tf.Variable(tf.constant(0.1, shape=[n_hidden_2])),
        'out': tf.Variable(tf.constant(0.1, shape=[1]))
    }

    x_data = tf.placeholder(tf.float32, [None, 52])
    y_data = tf.placeholder(tf.float32, [None, 1])

# Construct model
    pred = multilayer_perceptron(x_data, weights, biases)

# Minimize the mean squared errors.
    #loss = tf.nn.l2_loss(y_data - pred)
    loss=tf.reduce_mean(tf.squared_difference(y_data, pred))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# train = optimizer.minimize(loss)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.001, global_step,
                                       100, 10^-6, staircase=True)
    train = tf.train.AdamOptimizer(learning_rate,0.9).minimize(loss)


    init = tf.initialize_all_variables()

# Launch the graph.
    sess = tf.Session()
    sess.run(init)
    #mnistx=mnist[:,0:5]
    mnistx=mnist[:,1:53]
    mnistx=mnistx.reshape([total_train_row,52])
    mnisty=mnist[:,0]
    mnisty=mnisty.reshape([total_train_row,1])
    curLoss=5
    for step in range(training_epochs):
        if curLoss>=0.000005:
            #kf = KFold(n_splits=100)
            #kf.get_n_splits(mnistx)
           #print(kf)
            #KFold(n_splits=100, random_state=None, shuffle=False)
            for train_index in range(0,20):
                rand_num=randint(1,num_batch)
        #print("TRAIN:", train_index, "TEST:", test_index)
               # X_train, X_test = mnistx[train_index],mnistx[test_index]
               # y_train, y_test = mnisty[train_index], mnisty[test_index]
                trainx=mnistx[batch*(rand_num-1):batch*(rand_num),:]
                trainy=mnisty[batch*(rand_num-1):batch*(rand_num),:]
                sess.run(train, feed_dict={x_data: trainx, y_data: trainy})

                z= sess.run(pred, feed_dict={x_data: mnistx})
                curLoss = sess.run(loss, feed_dict={x_data: mnistx, y_data: mnisty})
                #print(train_index)
            print('Epoch',step)
            print(curLoss)

        else:
            break
        #print(curLoss)
    z= sess.run(pred, feed_dict={x_data: mnistx})
    w1=sess.run(weights['h1'])
    w2=sess.run(weights['out'])
    b1=sess.run(biases['b1'])
    b2=sess.run(biases['out'])
    error1=z-mnisty
    
    return (error1,w1,w2,b1,b2)
