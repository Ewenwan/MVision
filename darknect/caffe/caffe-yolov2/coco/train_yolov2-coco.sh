#!/usr/bin/env sh

CAFFE_HOME=../..

SOLVER=./yolov2_coco-solver.prototxt
#WEIGHTS=../../../../yolov2/yolov2.caffemodel
#WEIGHTS=./yolov2-coco.caffemodel
WEIGHTS=/home/wanyouwen/ewenwan/software/caffe_yolo/weights/bvlc_googlenet.caffemodel
$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER \
    --weights=$WEIGHTS \
    --gpu=4
    # --gpu=6 2>&1 | tee -a train_yolov2.log
