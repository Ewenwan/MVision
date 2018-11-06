import cv2
import numpy as np


ann = cv2.ml.ANN_MLP_create()

ann.setLayerSizes(np.array([64, 16, 3], dtype=np.float32))

num0 = [
  0, 1, 1, 1, 1, 1, 1, 0, 
  1, 1, 0, 0, 0, 0, 1, 1, 
  1, 1, 0, 0, 0, 0, 1, 1,
  1, 1, 0, 0, 0, 0, 1, 1,
  1, 1, 0, 0, 0, 0, 1, 1,
  1, 1, 0, 0, 0, 0, 1, 1,
  1, 1, 0, 0, 0, 0, 1, 1,
  0, 1, 1, 1, 1, 1, 1, 0
]

num4 = [
  0, 0, 0, 0, 1, 1, 1, 1, 
  0, 0, 0, 1, 1, 0, 1, 1, 
  0, 0, 1, 1, 0, 0, 1, 1,
  0, 1, 1, 0, 0, 0, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1,
  0, 0, 0, 0, 0, 0, 1, 1,
  0, 0, 0, 0, 0, 0, 1, 1,
  0, 0, 0, 0, 0, 0, 1, 1
]

num1 = [
  0, 0, 0, 0, 1, 1, 0, 0, 
  0, 0, 0, 1, 1, 1, 0, 0, 
  0, 0, 0, 1, 1, 1, 0, 0, 
  0, 0, 0, 0, 1, 1, 0, 0, 
  0, 0, 0, 0, 1, 1, 0, 0, 
  0, 0, 0, 0, 1, 1, 0, 0, 
  0, 0, 0, 0, 1, 1, 0, 0, 
  0, 0, 0, 1, 1, 1, 1, 0
]


train_data = [
  (num0, [1, 0, 0]),
  (num1, [0, 1, 0]),
  (num4, [0, 0, 1])
]

ann.setTrainMethod(cv2.ml.ANN_MLP_RPROP)

for x in range(0, 500):
  print x
  for t, r in train_data:
    ann.train(np.array([t], dtype=np.float32), 
      cv2.ml.ROW_SAMPLE,
      np.array([r], dtype=np.float32)
    )

print ann.predict(np.array([num0], dtype=np.float32))
print ann.predict(np.array([num1], dtype=np.float32))
print ann.predict(np.array([num4], dtype=np.float32))
