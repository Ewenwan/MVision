import cv2
import numpy as np
from matplotlib import pyplot as plt

camera = cv2.VideoCapture(0)

while True:
  ret, img = camera.read()
  color = ('b','g','r')
  for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    plt.show()
  #cv2.imshow("frame", img)
  # k = cv2.waitKey(30) & 0xff
  # if k == 27:
  #     break

camera.release()
cv2.destroyAllWindows()
