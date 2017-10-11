from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
#import copy_data_excel

from six.moves import urllib

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)

LEARNING_RATE = 0.0001

#copy_data_excel.copy_data_excel()


def maybe_download(train_data, test_data, predict_data):

  if train_data:
    train_file_name = train_data
  else:
    train_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "file:///C:/Users/chak282/AppData/Local/Programs/Python/Python35/train_data.csv",
        train_file.name)
    train_file_name = train_file.name
    train_file.close()
    print("Training data is downloaded to %s" % train_file_name)

  if test_data:
    test_file_name = test_data
  else:
    test_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "file:///C:/Users/chak282/AppData/Local/Programs/Python/Python35/test_data.csv", test_file.name)
    test_file_name = test_file.name
    test_file.close()
    print("Test data is downloaded to %s" % test_file_name)

  if predict_data:
    predict_file_name = predict_data
  else:
    predict_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "file:///C:/Users/chak282/AppData/Local/Programs/Python/Python35/predict_data.csv",
        predict_file.name)
    predict_file_name = predict_file.name
    predict_file.close()
    print("Prediction data is downloaded to %s" % predict_file_name)

  return train_file_name, test_file_name, predict_file_name


def model_fn(features, targets, mode, params):

  # Connect the first hidden layer to input layer

  first_hidden_layer = tf.contrib.layers.fully_connected(inputs=features,num_outputs=60,activation_fn=tf.nn.tanh)

  # Connect the second hidden layer to first hidden layer
  second_hidden_layer = tf.contrib.layers.fully_connected(inputs=first_hidden_layer,num_outputs=60,activation_fn=tf.nn.tanh)

  #third_hidden_layer = tf.contrib.layers.fully_connected(inputs=second_hidden_layer,num_outputs=30,activation_fn=tf.nn.tanh)
  # Connect the output layer to second hidden layer

  #fourth_hidden_layer = tf.contrib.layers.fully_connected(inputs=third_hidden_layer,num_outputs=30,activation_fn=tf.nn.tanh)

  #output_layer = tf.contrib.layers.fully_connected(inputs=third_hidden_layer,num_outputs=1,activation_fn=tf.nn.tanh)
  output_layer = tf.contrib.layers.linear(second_hidden_layer, 1)

  #output_layer = tf.contrib.layers.fully_connected(inputs=second_hidden_layer,num_outputs=1,activation_fn=tf.softmax)
  
  # Reshape output layer to 1-dim Tensor to return predictions
  predictions = tf.reshape(output_layer, [-1])
  predictions_dict = {"disturbance": predictions}

  # Calculate loss using mean squared error
  loss = tf.losses.mean_squared_error(targets, predictions)

  # Calculate root mean squared error as additional eval metric
  eval_metric_ops = {
      "rmse": tf.metrics.root_mean_squared_error(
          tf.cast(targets, tf.float64), predictions)
  }

  step = tf.Variable(0, trainable=False)
  rate = tf.train.exponential_decay(0.001, step, 1, 0.00005)
  train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=params["learning_rate"],
      optimizer=tf.train.AdamOptimizer(learning_rate=rate))
  
  #train_op=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

  return model_fn_lib.ModelFnOps(
      mode=mode,
      predictions=predictions_dict,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  
  abalone_train, abalone_test, abalone_predict = maybe_download(
      FLAGS.train_data, FLAGS.test_data, FLAGS.predict_data)

 
  training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=abalone_train, target_dtype=np.float64, features_dtype=np.float64)


  test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=abalone_test, target_dtype=np.float64, features_dtype=np.float64)

  
  prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=abalone_predict, target_dtype=np.float64, features_dtype=np.float64)


  model_params = {"learning_rate": LEARNING_RATE}


  nn = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params)
  
  def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    return x, y
  

  nn.fit(input_fn=get_train_inputs, steps=50000)

 
  def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)
    return x, y
  
  ev = nn.evaluate(input_fn=get_test_inputs, steps=1)
  print("Loss: %s" % ev["loss"])
  print("Root Mean Squared Error: %s" % ev["rmse"])

  # Print out predictions
  predictions = nn.predict(x=prediction_set.data, as_iterable=True)
  for i, p in enumerate(predictions):
    print(p["disturbance"])


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--train_data", type=str, default="", help="Path to the training data.")
  parser.add_argument(
      "--test_data", type=str, default="", help="Path to the test data.")
  parser.add_argument(
      "--predict_data",
      type=str,
      default="",
      help="Path to the prediction data.")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

