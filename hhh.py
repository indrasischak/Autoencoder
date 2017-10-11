from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib.request

import tensorflow as tf
import numpy as np

IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_RRL = "http://download.tensorflow.org/data/iris_test.csv"

if not os.path.exists(IRIS_TRAINING):
    raw = urllib.request.urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, 'w') as f:
        f.write(raw)

if not os.path.exists(IRIS_TEST):
    raw = urllib.request.urlopen(IRIS_TEST_RRL).read()
    with open(IRIS_TEST, 'w') as f:
        f.write(raw)

# load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32
)

# Specify that all features have real_valued data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# Build 3 layers DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 30],
                                            n_class=3,
                                            model_dir="/tem/iris_model")

# Define the training imputs
def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)

    return x, y

# Fit model
classifier.fit(input_fn=get_train_inputs(), steps=2000)

# Define the test inputs
def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)

    return x, y

# Evaluate accuracy
accuracy_score = classifier.evaluate(input_fn=get_test_inputs(), steps=1)["accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
