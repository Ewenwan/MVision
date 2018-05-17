# 使用 caffe 训练 yolo



## 训练时用的 模型框架文件： gnet_train.prototxt
##  训练时用的 学习参数    ： gnet_solver.prototxt

## 测试集上使用的测试模型：  gnet_test.prototxt

## 检测识别模型：            gnet_deploy.prototxt


## 不调用检测识别模型文件 和 权重文件 得到识别结果 show_det.py

## 训练时使用的 bash脚本文件 : train.sh
      需要提供 一个 预训练的 模型权重文件
      bvlc_googlenet.caffemodel
      下载： http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel

## 再一个 voc / coco 数据集 的图片需要转换成 数据库 lmdb/leveldb 文件 这样读写会比较快

      使用 ./convert.sh 转换 图片数据 到 lmdb/leveldb
        cd cafe/data/yolo
        ln -s /your/path/to/VOCdevkit/ .
        python ./get_list.py
        # change related path in script convert.sh
        ./convert.sh 

### 上述有需要改动的地方

      /////////////////////////////////
      convert.sh 
      ////////////
      #!/usr/bin/env sh
      # cafe main path
      CAFFE_ROOT=../..
      #/your/path/to/vocroot/
      ROOT_DIR=/home/wanyouwen/ewenwan/software/darknet/data/voc/VOCdevkit
      # string label Map to int label
      LABEL_FILE=$CAFFE_ROOT/data/yolo/label_map.txt

      # 2007 + 2012 trainval
      # source img
      LIST_FILE=$CAFFE_ROOT/data/yolo/trainval.txt
      # date base file
      LMDB_DIR=./lmdb/trainval_lmdb
      # shuff;e
      SHUFFLE=true

      # 2007 test
      # LIST_FILE=$CAFFE_ROOT/data/yolo/test_2007.txt
      # LMDB_DIR=./lmdb/test2007_lmdb
      # SHUFFLE=false
      # resize size
      RESIZE_W=448
      RESIZE_H=448

      # execute
      $CAFFE_ROOT/build/tools/convert_box_data  --resize_width=$RESIZE_W --resize_height=$RESIZE_H \
        --label_file=$LABEL_FILE $ROOT_DIR $LIST_FILE $LMDB_DIR --encoded=true --encode_type=jpg --shuffle=$SHUFFLE

      /////////////////////////////// 

#### 需要在 tools下新建 转换 voc标签框的文件 convert_box_data.cpp
      见 https://github.com/yeahkun/caffe-yolo/blob/master/tools/convert_box_data.cpp

      需要对
      caffe/util/io.hpp 
      caffe/src/caffe/util/io.cpp
      做一定修改 

## 测试时使用的 bash脚本文件 :test.sh
