#!/usr/bin/env sh
# caffe main path
CAFFE_HOME=../..
# learning rate paramter
SOLVER=./gnet_solver.prototxt
# init weights 
WEIGHTS=/home/wanyouwen/ewenwan/software/caffe-yolo/weights/bvlc_googlenet.caffemodel
# training  --gpu=0,1,2,3,4,5,6,7
$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER --weights=$WEIGHTS \
	--gpu=0,1,2,3 2>&1 | tee train_yolov1.log
