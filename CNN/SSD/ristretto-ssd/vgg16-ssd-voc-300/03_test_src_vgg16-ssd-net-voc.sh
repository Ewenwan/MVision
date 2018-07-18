#!/usr/bin/env sh
# test 测试
MODELS="../../../models/ssd-voc-300/ssd_voc_300_train_val.prototxt"
WEIGHTS="../../../models/ssd-voc-300/ssd_voc_300.caffemodel"

../../../build/tools/caffe test \
    --model=$MODELS \
    --weights=$WEIGHTS \
    --iterations=2000 \
    --net_type=ssd_detection \
    --gpu=6 2>&1 | tee -a ./ssd_voc_300_src_test.log
