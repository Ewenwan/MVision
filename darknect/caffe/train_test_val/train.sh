#!/usr/bin/env sh
# test 
CAFFE_HOME=../..

PROTO=./gnet_test.prototxt
MODEL=$1
ITER=500
GPU_ID=$2

$CAFFE_HOME/build/tools/test_detection \
    --model=$PROTO --iterations=$ITER \
    --weights=$MODEL --gpu=$GPU_ID
