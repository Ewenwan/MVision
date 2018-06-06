
CAFFE_HOME=../../..

SOLVER="./SHN_ssd_33_solver.prototxt"

#WEIGHTS="/home/wanyouwen/ewenwan/software/caffe_yolo/weights/VGG_ILSVRC_16_layers_fc_reduced.caffemodel"
#WEIGHTS="./models/SHN_ssd_33_iter_19000.caffemodel"
WEIGHTS="./models/SHN_ssd_33_iter_3000.caffemodel"
$CAFFE_HOME/build/tools/caffe train \
        --solver=$SOLVER --weights=$WEIGHTS \
        --gpu=4,5,6,7 2>&1 | tee -a ./VGG_VOC0712_SSD_300x300.log
