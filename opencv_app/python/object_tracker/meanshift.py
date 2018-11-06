import numpy as np
import cv2

cap = cv2.VideoCapture(0)
# capture the first frame
ret,frame = cap.read()
# mark the ROI
r,h,c,w = 10, 200, 10, 200
# wrap in a tuple
track_window = (c,r,w,h)

# extract the ROI for tracking
roi = frame[r:r+h, c:c+w]
# switch to HSV
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# create a mask with upper and lower boundaries of colors you want to track
mask = cv2.inRange(hsv_roi, np.array((100., 30.,32.)), np.array((180.,120.,255.)))
# calculate histograms of roi
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        print dst
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break

    else:
        break

cv2.destroyAllWindows()
cap.release()
