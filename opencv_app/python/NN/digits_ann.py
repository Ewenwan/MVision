import cv2
import cPickle
import numpy as np
import gzip

"""OpenCV ANN Handwritten digit recognition example

Wraps OpenCV's own ANN by automating the loading of data and supplying default paramters,
such as 20 hidden layers, 10000 samples and 1 training epoch.

The load data code is taken from http://neuralnetworksanddeeplearning.com/chap1.html
by Michael Nielsen
"""

def load_data():
  mnist = gzip.open('./data/mnist.pkl.gz', 'rb')
  training_data, classification_data, test_data = cPickle.load(mnist)
  mnist.close()
  return (training_data, classification_data, test_data)

def wrap_data():
  tr_d, va_d, te_d = load_data()
  training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
  training_results = [vectorized_result(y) for y in tr_d[1]]
  training_data = zip(training_inputs, training_results)
  validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
  validation_data = zip(validation_inputs, va_d[1])
  test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
  test_data = zip(test_inputs, te_d[1])
  return (training_data, validation_data, test_data)

def vectorized_result(j):
  e = np.zeros((10, 1))
  e[j] = 1.0
  return e

def create_ANN(hidden = 20):
  ann = cv2.ml.ANN_MLP_create()
  ann.setLayerSizes(np.array([784, hidden, 10]))
  ann.setTrainMethod(cv2.ml.ANN_MLP_RPROP)
  ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
  ann.setTermCriteria(( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1 ))
  return ann

def train(ann, samples = 10000, epochs = 1):
  tr, val, test = wrap_data()
  
  
  for x in xrange(epochs):
    counter = 0
    for img in tr:
      
      if (counter > samples):
        break
      if (counter % 1000 == 0):
        print "Epoch %d: Trained %d/%d" % (x, counter, samples)
      counter += 1
      data, digit = img
      ann.train(np.array([data.ravel()], dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array([digit.ravel()], dtype=np.float32))
    print "Epoch %d complete" % x
  return ann, test
  
def test(ann, test_data):
  sample = np.array(test_data[0][0].ravel(), dtype=np.float32).reshape(28, 28)
  cv2.imshow("sample", sample)
  cv2.waitKey()
  print ann.predict(np.array([test_data[0][0].ravel()], dtype=np.float32))

def predict(ann, sample):
  resized = sample.copy()
  rows, cols = resized.shape
  if (rows != 28 or cols != 28) and rows * cols > 0:
    resized = cv2.resize(resized, (28, 28), interpolation = cv2.INTER_LINEAR)
  return ann.predict(np.array([resized.ravel()], dtype=np.float32))

"""
usage:
ann, test_data = train(create_ANN())
test(ann, test_data)
"""


