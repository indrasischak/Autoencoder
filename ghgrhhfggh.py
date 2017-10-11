import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('rnn_trial.ckpt.meta')
    saver.restore(sess,"C:/Users/chak282/AppData/Local/Programs/Python/Python35/rnn_trial.ckpt")
    X_new1 = np.array([
        [1],
        [2],
                      ])
    y_pred=sess.run(outputs,feed_dict={X:X_new1})
    y_pred1=y_pred.reshape(n_steps,1)
