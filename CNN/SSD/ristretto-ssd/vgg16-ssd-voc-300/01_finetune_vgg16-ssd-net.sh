#!/usr/bin/env sh
# finetune 微调

SOLVER="../../../examples/ssd/ssd-voc-300/solver_finetune.prototxt"
WEIGHTS="../../../models/ssd-voc-300/ssd_voc_300.caffemodel"

../../../build/tools/caffe train \
    --solver=$SOLVER \
    --weights=$WEIGHTS \
    --net_type=ssd_detection \
    --gpu=4 2>&1 | tee -a ./ssd_voc_300_quantized_finetune.log
