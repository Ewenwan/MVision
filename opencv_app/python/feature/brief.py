import numpy as np
import cv2
from matplotlib import pyplot as plt

"""FeatureDetector_create and DescriptorExtractor_create do not exists in 3.0.0-rc1"""

img = cv2.imread('images/coat_of_arms_single.jpg',0)

# Initiate STAR detector
star = cv2.FeatureDetector_create("STAR")

# Initiate BRIEF extractor
brief = cv2.DescriptorExtractor_create("BRIEF")

# find the keypoints with STAR
kp = star.detect(img,None)

# compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)

print brief.getInt('bytes')
print des.shape
