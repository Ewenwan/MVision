#!/usr/bin/env sh

CAFFE_HOME=../../..

PROTO=./gnet_region_test_darknet_v3.prototxt
#MODEL=./models/gnet_yolo_region_darknet_v3_iter_410000.caffemodel
#MODEL=./models/gnet_yolo_region_darknet_v3_multifixed_iter_98000.caffemodel
#MODEL=./models/gnet_yolo_region_darknet_v3_multifixed_0_anchor_iter_200000.caffemodel
MODEL=./models/gnet_yolo_region_darknet_v3_pretrain_rectify_iter_200000.caffemodel
ITER=310
GPU_ID=1

$CAFFE_HOME/build/tools/test_detection \
    --model=$PROTO --iterations=$ITER \
    --weights=$MODEL --gpu=$GPU_ID 2>&1 | tee test_darknet_v3_rectify.log

