#!/usr/bin/env sh
# test

MODELS="../../../models/ssd-voc-300/ssd_voc_300_quantized.prototxt"
WEIGHTS="../../../models/ssd-voc-300/ssd-voc-300-quantized-weights_iter_2000.caffemodel"

../../../build/tools/caffe test \
    --model=$MODELS \
    --weights=$WEIGHTS \
    --iterations=2000 \
    --net_type=ssd_detection \
    --gpu=6 2>&1 | tee -a ./ssd_voc_300_quantized_test.log
