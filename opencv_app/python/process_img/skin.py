import numpy as np
import cv2

camera = cv2.VideoCapture(0)

# determine upper and lower HSV limits for (my) skin tones
lower = np.array([0, 100, 0], dtype="uint8")
upper = np.array([50,255,255], dtype="uint8")

while (True):
  ret, frame = camera.read()
  if not ret:
    continue
  # switch to HSV
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  # find mask of pixels within HSV range
  skinMask = cv2.inRange(hsv, lower, upper)
  # denoise
  skinMask = cv2.GaussianBlur(skinMask, (9, 9), 0)
  # kernel for morphology operation
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
  # CLOSE (dilate / erode)
  skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_CLOSE, kernel, iterations = 3)
  # denoise the mask
  skinMask = cv2.GaussianBlur(skinMask, (9, 9), 0)
  # only display the masked pixels
  skin = cv2.bitwise_and(frame, frame, mask = skinMask)
  cv2.imshow("HSV", skin)
  # quit or save frame
  key = cv2.waitKey(1000 / 12) & 0xff
  if key == ord("q"):
    break
  if key == ord("p"):
    cv2.imwrite("skin.jpg", skin) 

cv2.destroyAllWindows()
