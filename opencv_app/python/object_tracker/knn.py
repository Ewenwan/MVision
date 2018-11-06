import cv2
import numpy as np

knn = cv2.createBackgroundSubtractorKNN(detectShadows = True)
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,12))
camera = cv2.VideoCapture("/home/d3athmast3r/Downloads/traffic.flv")

def drawCnt(fn, cnt):
  if cv2.contourArea(cnt) > 1400:
    (x, y, w, h) = cv2.boundingRect(cnt)
    cv2.rectangle(fn, (x, y), (x + w, y + h), (255, 255, 0), 2)

while True:
  ret, frame = camera.read()
  if not ret:
    break
  fg = knn.apply(frame.copy())
  fg_bgr = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
  bw_and = cv2.bitwise_and(fg_bgr, frame)
  draw = cv2.cvtColor(bw_and, cv2.COLOR_BGR2GRAY)
  draw = cv2.GaussianBlur(draw, (21, 21), 0)
  draw = cv2.threshold(draw, 10, 255, cv2.THRESH_BINARY)[1]
  draw = cv2.dilate(draw, es, iterations = 2)
  image, contours, hierarchy = cv2.findContours(draw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  for c in contours:
    drawCnt(frame, c)
  cv2.imshow("motion detection", frame)
  if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
      break

camera.release()
cv2.destroyAllWindows()
