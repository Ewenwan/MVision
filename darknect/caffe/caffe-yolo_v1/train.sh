#!/usr/bin/env sh

CAFFE_HOME=../..
# learning rate paramter
SOLVER=./gnet_solver.prototxt
#WEIGHTS=/your/path/to/bvlc_googlenet.caffemodel
#WEIGHTS=/home/wanyouwen/ewenwan/software/caffe_yolo/weights/bvlc_googlenet.caffemodel
# WEIGHTS=/home/wanyouwen/ewenwan/software/caffe_yolo/caffe-yolov1_oldVer/examples/yolo/yolov1_models/yolov1_models_iter_26000.solverstate
#WEIGHTS=/home/wanyouwen/ewenwan/software/caffe_yolo/caffe-yolov1_oldVer/examples/yolo/yolov1_models/yolov1_models_iter_26000.caffemodel
WEIGHTS=/home/wanyouwen/ewenwan/software/caffe_yolo/caffe-yolov1_oldVer/examples/yolo/yolov1_models/yolov1_models_iter_32000.caffemodel

# training  --gpu=0,1,2,3,4,5,6,7
$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER --weights=$WEIGHTS \
    --gpu=2,3,4,5 2>&1 | tee -a train_yolov1.log
