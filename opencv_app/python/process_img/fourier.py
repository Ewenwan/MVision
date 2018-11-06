import cv2
import numpy as np
from matplotlib import pyplot as plot

img = cv2.imread('../images/bb.jpg', 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

row, cols = img.shape
crow, ccol = row / 2, cols / 2
fshift[crow - 30: crow+30, ccol - 30: ccol + 30] = 0

f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plot.subplot(221), plot.imshow(img, cmap = "gray")
plot.title("Input"), plot.xticks([]), plot.yticks([])

plot.subplot(222), plot.imshow(magnitude_spectrum, cmap = "gray")
plot.title('magnitude_spectrum'), plot.xticks([]), plot.yticks([])

plot.subplot(223), plot.imshow(img_back, cmap = "gray")
plot.title("Input in JET"), plot.xticks([]), plot.yticks([])
plot.show()
