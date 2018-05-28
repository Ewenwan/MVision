#!/usr/bin/env sh

CAFFE_HOME=../../..

SOLVER=./my_solver_darknet_v2.prototxt
#WEIGHTS=/home/wanyouwen/ewenwan/software/caffe_yolo/weights/bvlc_googlenet.caffemodel
#SNAPSHOT=./models/gnet_yolo_region_darknet_v3_multifixed_iter_2000.solverstate
#WEIGHTS=./gnet_yolo_region_darknet_v3_pretrain_iter_600000.caffemodel
#SNAPSHOT=./models/gnet_yolo_region_darknet_v3_pretrain_rectify_iter_120000.solverstate
#WEIGHTS=./models/yolov2_models_iter_25000.caffemodel
WEIGHTS=../../../../yolov2/yolov2.caffemodel
$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER --weights=$WEIGHTS \
    --gpu=6,7 2>&1 | tee -a train_yolov2.log

