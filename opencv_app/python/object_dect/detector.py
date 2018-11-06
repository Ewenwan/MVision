import cv2
import numpy as np

datapath = "/home/d3athmast3r/dev/python/CarData/TrainImages/"
SAMPLES = 400

def path(cls,i):
    return "%s/%s%d.pgm"  % (datapath,cls,i+1)

def get_flann_matcher():
  flann_params = dict(algorithm = 1, trees = 5)
  return cv2.FlannBasedMatcher(flann_params, {})

def get_bow_extractor(extract, match):
  return cv2.BOWImgDescriptorExtractor(extract, match)

def get_extract_detect():
  return cv2.xfeatures2d.SIFT_create(), cv2.xfeatures2d.SIFT_create()

def extract_sift(fn, extractor, detector):
  im = cv2.imread(fn,0)
  return extractor.compute(im, detector.detect(im))[1]
    
def bow_features(img, extractor_bow, detector):
  return extractor_bow.compute(img, detector.detect(img))

def car_detector():
  pos, neg = "pos-", "neg-"
  detect, extract = get_extract_detect()
  matcher = get_flann_matcher()
  #extract_bow = get_bow_extractor(extract, matcher)
  print "building BOWKMeansTrainer..."
  bow_kmeans_trainer = cv2.BOWKMeansTrainer(12)
  extract_bow = cv2.BOWImgDescriptorExtractor(extract, matcher)

  print "adding features to trainer"
  for i in range(SAMPLES):
    print i
    bow_kmeans_trainer.add(extract_sift(path(pos,i), extract, detect))
    #bow_kmeans_trainer.add(extract_sift(path(neg,i), extract, detect))
    
  vocabulary = bow_kmeans_trainer.cluster()
  extract_bow.setVocabulary(vocabulary)

  traindata, trainlabels = [],[]
  print "adding to train data"
  for i in range(SAMPLES):
    print i
    traindata.extend(bow_features(cv2.imread(path(pos, i), 0), extract_bow, detect))
    trainlabels.append(1)
    traindata.extend(bow_features(cv2.imread(path(neg, i), 0), extract_bow, detect))
    trainlabels.append(-1)

  svm = cv2.ml.SVM_create()
  svm.setType(cv2.ml.SVM_C_SVC)
  svm.setGamma(1)
  svm.setC(35)
  svm.setKernel(cv2.ml.SVM_RBF)

  svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))
  return svm, extract_bow
