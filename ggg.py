from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from six.moves import urllib

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)

LEARNING_RATE = 0.001


def maybe_download(train_data, test_data, predict_data):

  if train_data:
    train_file_name = train_data
  else:
    train_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "file:///C:/Users/chak282/AppData/Local/Programs/Python/Python35/abalone_train.csv",
        train_file.name)
    train_file_name = train_file.name
    train_file.close()
    print("Training data is downloaded to %s" % train_file_name)

  if test_data:
    test_file_name = test_data
  else:
    test_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "file:///C:/Users/chak282/AppData/Local/Programs/Python/Python35/abalone_test.csv", test_file.name)
    test_file_name = test_file.name
    test_file.close()
    print("Test data is downloaded to %s" % test_file_name)

  if predict_data:
    predict_file_name = predict_data
  else:
    predict_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "file:///C:/Users/chak282/AppData/Local/Programs/Python/Python35/abalone_predict.csv",
        predict_file.name)
    predict_file_name = predict_file.name
    predict_file.close()
    print("Prediction data is downloaded to %s" % predict_file_name)

  return train_file_name, test_file_name, predict_file_name



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

  feature_columns = [tf.contrib.layers.real_valued_column("", 7)]


  nn = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                       hidden_units=[10, 10, 10],
                                       activation_fn=tf.nn.relu,
                                       dropout=0.2,
                                       n_classes=3,
                                       optimizer="Adam")
  
  def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    return x, y
  

  nn.fit(input_fn=get_train_inputs, steps=10000)

 
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

