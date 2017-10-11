import tensorflow as tf

detection_graph = tf.Graph()
with tf.Session(graph=detection_graph) as sess:
    saver = tf.train.import_index('C:/Users/chak282/AppData/Local/Temp/tmp56fzzjxg/model.ckpt-5000.index')
    saver.restore(sess, 'C:/Users/chak282/AppData/Local/Temp/tmp56fzzjxg/model.ckpt-5000.data-00000-of-00001')




