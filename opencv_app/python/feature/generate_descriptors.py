import os
import cv2
import numpy as np
from os import walk
from os.path import join
import sys

def create_descriptors(folder):
  files = []
  for (dirpath, dirnames, filenames) in walk(folder):
    files.extend(filenames)
  for f in files:
    save_descriptor(folder, f, cv2.xfeatures2d.SIFT_create())

def save_descriptor(folder, image_path, feature_detector):
  print "reading %s" % image_path
  if image_path.endswith("npy"):
    return
  img = cv2.imread(join(folder, image_path), 0)
  keypoints, descriptors = feature_detector.detectAndCompute(img, None)
  descriptor_file = image_path.replace("jpg", "npy")
  np.save(join(folder, descriptor_file), descriptors)

dir = sys.argv[1]

create_descriptors(dir)
