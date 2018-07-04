# cd /home/wanyouwen/ewenwan/software/caffe-ssd/examples/ssd/SSD_300x300

SOLVER = "models/VGG_coco_SSD_300x300_solver.prototxt"
WEIGHT = "models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel"
./build/tools/caffe train \
--solver= $SOLVER \
--weights= $WEIGHT \
--gpu 0,1,2,3 2>&1 | tee -a ./VGG_coco_SSD_300x300.log
