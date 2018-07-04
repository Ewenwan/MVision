# cd /home/wanyouwen/ewenwan/software/caffe-ssd/examples/ssd/SSD_300x300

CAFFE_HOME=../../..
SOLVER = "models/VGG_coco_SSD_300x300_solver.prototxt"
WEIGHTS="/home/wanyouwen/ewenwan/software/caffe_yolo/weights/VGG_ILSVRC_16_layers_fc_reduced.caffemodel"

$CAFFE_HOME/build/tools/caffe train \
--solver= $SOLVER \
--weights= WEIGHTS \
--gpu 0,1,2,3 2>&1 | tee -a ./VGG_coco_SSD_300x300.log
