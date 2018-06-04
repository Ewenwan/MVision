
CAFFE_HOME=../../..

SOLVER="./MN2_ssd_33_solver.prototxt"

#WEIGHTS="/home/wanyouwen/ewenwan/software/caffe_yolo/weights/VGG_ILSVRC_16_layers_fc_reduced.caffemodel"
#WEIGHTS="./models/ssd_33_iter_19000.caffemodel"
WEIGHTS="./mobilenet_v2.caffemodel"
$CAFFE_HOME/build/tools/caffe train \
        --solver=$SOLVER --weights=$WEIGHTS \
        --gpu=3 2>&1 | tee -a ./MN2_VOC0712_SSD_300x300.log
