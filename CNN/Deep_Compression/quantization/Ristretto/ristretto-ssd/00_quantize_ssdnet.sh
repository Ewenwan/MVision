#!/usr/bin/env sh
#CaffeRoot = "../.."

../../build/tools/ristretto quantize \
        --model=./ssd_33_train_val.prototxt \
        --weights=../../../../caffe-ssd/examples/ssd/SSD_300x300/models/ssd_33_iter_50000.caffemodel \
        --model_quantized=./quantized.prototxt \
        --trimming_mode=dynamic_fixed_point --gpu=5 --iterations=2000 \
        --error_margin=3
