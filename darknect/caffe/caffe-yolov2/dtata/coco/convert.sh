#!/usr/bin/env sh

CAFFE_ROOT=/home/wanyouwen/ewenwan/software/caffe_yolo/caffe-yolov2_oldVer
ROOT_DIR=./

# 标签 id
LABEL_FILE=/home/wanyouwen/ewenwan/software/caffe_yolo/caffe-yolov2_oldVer/data/coco/label_map.txt

# 训练集
LIST_FILE=/home/wanyouwen/ewenwan/software/caffe_yolo/caffe-yolov2_oldVer/data/coco/train2017_img_xml_label.txt
# 生成目录
LMDB_DIR=/data4/quantization/coco/coco2017/coco/lmdb/yolov2/trainval_lmdb
SHUFFLE=false

RESIZE_W=416
RESIZE_H=416

$CAFFE_ROOT/build/tools/convert_box_data --resize_width=$RESIZE_W --resize_height=$RESIZE_H \
  --label_file=$LABEL_FILE $ROOT_DIR $LIST_FILE $LMDB_DIR --encoded=true --encode_type=jpg --shuffle=$SHUFFLE

# 测试集
 LIST_FILE=/home/wanyouwen/ewenwan/software/caffe_yolo/caffe-yolov2_oldVer/data/coco/val2017_img_xml_label.txt
# 生成目录
 LMDB_DIR=/data4/quantization/coco/coco2017/coco/lmdb/yolov2/test2007_lmdb
 SHUFFLE=false

$CAFFE_ROOT/build/tools/convert_box_data \
  --resize_width=$RESIZE_W \
  --resize_height=$RESIZE_H \
  --label_file=$LABEL_FILE \
  $ROOT_DIR $LIST_FILE $LMDB_DIR --encoded=true --encode_type=jpg --shuffle=$SHUFFLE
