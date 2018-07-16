#!/usr/bin/env sh
#CaffeRoot = "../.."

../../../build/tools/ristretto quantize \
        --model=./ssd_33_voc_train_val.prototxt \
        --weights=../src/models/ssd_33_iter_800.caffemodel \
        --model_quantized=./ssd_33_voc_quantized.prototxt \
        --trimming_mode=dynamic_fixed_point --gpu=4 --iterations=2000 \
        --error_margin=3 \
        --net_type=ssd_detection
