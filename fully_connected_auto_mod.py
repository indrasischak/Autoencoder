def ggg_auto(number):
    from sklearn.model_selection import KFold
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
    workbook = xlrd.open_workbook('tryy2.xlsx')
    #workbook = xlrd.open_workbook('myfile.xls')
    sheet1 = workbook.sheet_by_name('tryy2')
    total_train_row=10000
    training_epochs=1000
    traindata = np.zeros(shape=[10000,25])
    testdata_n = np.zeros(shape=[10000,25])
    testdata_p = np.zeros(shape=[sheet1.nrows,4])
    for index1 in range(0,10000):
        traindata[index1,0]=sheet1.cell_value(index1,0)
        traindata[index1,1]=sheet1.cell_value(index1,1)
        traindata[index1,2]=sheet1.cell_value(index1,2)
        traindata[index1,3]=sheet1.cell_value(index1,3)
        traindata[index1,4]=sheet1.cell_value(index1,4)
        traindata[index1,5]=sheet1.cell_value(index1,5)
        traindata[index1,6]=sheet1.cell_value(index1,6)
        traindata[index1,7]=sheet1.cell_value(index1,7)
        traindata[index1,8]=sheet1.cell_value(index1,8)
        traindata[index1,9]=sheet1.cell_value(index1,9)
        traindata[index1,10]=sheet1.cell_value(index1,10)
        traindata[index1,11]=sheet1.cell_value(index1,11)
        traindata[index1,12]=sheet1.cell_value(index1,12)
        traindata[index1,13]=sheet1.cell_value(index1,13)
        traindata[index1,14]=sheet1.cell_value(index1,14)
        traindata[index1,15]=sheet1.cell_value(index1,15)
        traindata[index1,16]=sheet1.cell_value(index1,16)
        traindata[index1,17]=sheet1.cell_value(index1,17)
        traindata[index1,18]=sheet1.cell_value(index1,18)
        traindata[index1,19]=sheet1.cell_value(index1,19)
        traindata[index1,20]=sheet1.cell_value(index1,20)
        traindata[index1,21]=sheet1.cell_value(index1,21)
        traindata[index1,22]=sheet1.cell_value(index1,22)
        traindata[index1,23]=sheet1.cell_value(index1,23)
        traindata[index1,24]=sheet1.cell_value(index1,24)
        #traindata[index1,5]=sheet1.cell_value(index1,5)
    mnist=traindata
    mnist1=traindata
    #np.random.shuffle(mnist)
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
    n_hidden_1 = 50  # 1st layer number of features
    #n_hidden_2 = 400  # 2nd layer number of features
    n_input = 24


# Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        #'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        #'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_2], stddev=0.1)),
        #'h4': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_2], stddev=0.1)),
        'out': tf.Variable(tf.random_normal([n_hidden_1, 1]))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        #'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        #'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden_2])),
        #'b4': tf.Variable(tf.constant(0.1, shape=[n_hidden_2])),
        'out': tf.Variable(tf.random_normal([1]))
    }

    x_data = tf.placeholder(tf.float32, [None, 24])
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
    mnistx=np.delete(mnist,number,1)
    mnistx=mnistx.reshape([total_train_row,24])
    mnisty=mnist[:,number]
    mnisty=mnisty.reshape([total_train_row,1])
    curLoss=5
    for step in range(training_epochs):
        if curLoss>=0.000005:
            
            kf = KFold(n_splits=100)
            kf.get_n_splits(mnistx)
           #print(kf)
            KFold(n_splits=100, random_state=None, shuffle=False)
            for train_index, test_index in kf.split(mnistx):
        #print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = mnistx[train_index],mnistx[test_index]
                y_train, y_test = mnisty[train_index], mnisty[test_index]
                sess.run(train, feed_dict={x_data: X_train, y_data: y_train})

                z= sess.run(pred, feed_dict={x_data: X_train})
                curLoss = sess.run(loss, feed_dict={x_data: X_train, y_data: y_train})
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
