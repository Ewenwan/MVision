#!/bin/bash
# cafe main path
CAFFE_ROOT=/home/wanyouwen/ewenwan/software/caffe-yolo/caffe
echo "caffe path: "
echo $CAFFE_ROOT
#/your/path/to/voc/
ROOT_DIR=/home/wanyouwen/ewenwan/software/darknet/data/voc/
# string label Map to int label
LABEL_FILE=$CAFFE_ROOT/data/yolo/label_map.txt

echo "voc class label: "
echo $LABEL_FILE

# 2007 + 2012 trainval
# source img
# train
# LIST_FILE=$CAFFE_ROOT/data/yolo/trainval.txt
# 2007 test
LIST_FILE=$CAFFE_ROOT/data/yolo/test_2007.txt

echo "voc img list txt: "
echo $LIST_FILE

# date base file  one 
LMDB_DIR=$CAFFE_ROOT/data/yolo/trainval_lmdb
# shuff;e
SHUFFLE=true
# remove
rm -rf $LMDB_DIR
# 2007 test
# LIST_FILE=$CAFFE_ROOT/data/yolo/test_2007.txt
# LMDB_DIR=./test2007_lmdb
# SHUFFLE=false
# resize size
RESIZE_W=448
RESIZE_H=448

# execute
$CAFFE_ROOT/build/tools/convert_box_data --resize_width=$RESIZE_W --resize_height=$RESIZE_H \
  --label_file=$LABEL_FILE $ROOT_DIR $LIST_FILE $LMDB_DIR --encoded=true --encode_type=jpg --shuffle=$SHUFFLE
