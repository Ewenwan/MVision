# darknet yolo cfg 转换到 caffe model
# yolov1
## 转模型文件 
    python yolo_cfg_to_caffe_prototxt_v1.py yolov1_test.cfg yolov1_caffe_test.prototxt
    python yolo_cfg_to_caffe_prototxt_v1.py yolov1_test.cfg yolov1_bn_caffe_test.prototxt
## 可视化
    cd caffe 
    python python/draw_net.py models/yolov1_caffe_test.prototxt yolov1_caffenet.png  
    open yolov1_caffenet.png
## 转weight文件
    python yolo_weight_to_caffemodel_v1.py -m yolov1_caffe_test.prototxt -w yolov1.weights -o yolov1.caffemodel
## Convert yolo2 model to caffe
### convert tiny-yolo from darknet to caffe
    1. download tiny-yolo-voc.weights : https://pjreddie.com/media/files/tiny-yolo-voc.weights
    https://github.com/pjreddie/darknet/blob/master/cfg/tiny-yolo-voc.cfg
    2. python darknet2caffe.py tiny-yolo-voc.cfg tiny-yolo-voc.weights tiny-yolo-voc.prototxt tiny-yolo-voc.caffemodel
    3. download voc data and process according to https://github.com/marvis/pytorch-yolo2
    python valid.py cfg/voc.data tiny-yolo-voc.prototxt tiny-yolo-voc.caffemodel
    4. python scripts/voc_eval.py results/comp4_det_test_
    VOC07 metric? Yes
    AP for aeroplane = 0.6094
    AP for bicycle = 0.6781
    AP for bird = 0.4573
    AP for boat = 0.3786
    AP for bottle = 0.2081
    AP for bus = 0.6645
    AP for car = 0.6587
    AP for cat = 0.6720
    AP for chair = 0.3245
    AP for cow = 0.4902
    AP for diningtable = 0.5549
    AP for dog = 0.5905
    AP for horse = 0.6871
    AP for motorbike = 0.6695
    AP for person = 0.5833
    AP for pottedplant = 0.2535
    AP for sheep = 0.5374
    AP for sofa = 0.4878
    AP for train = 0.7004
    AP for tvmonitor = 0.5754
    Mean AP = 0.5391
    5. python detect.py tiny-yolo-voc.prototxt tiny-yolo-voc.caffemodel data/dog.jpg 

### convert tiny-yolo from darknet to caffe without bn
    1. python darknet.py tiny-yolo-voc.cfg tiny-yolo-voc.weights tiny-yolo-voc-nobn.cfg tiny-yolo-voc-nobn.weights
    2. python darknet2caffe.py tiny-yolo-voc-nobn.cfg tiny-yolo-voc-nobn.weights tiny-yolo-voc-nobn.prototxt tiny-yolo-voc-nobn.caffemodel
    3. python valid.py cfg/voc.data tiny-yolo-voc-nobn.prototxt tiny-yolo-voc-nobn.caffemodel
    4. python scripts/voc_eval.py results/comp4_det_test_
    VOC07 metric? Yes
    AP for aeroplane = 0.6094
    AP for bicycle = 0.6781
    AP for bird = 0.4573
    AP for boat = 0.3786
    AP for bottle = 0.2081
    AP for bus = 0.6645
    AP for car = 0.6587
    AP for cat = 0.6720
    AP for chair = 0.3245
    AP for cow = 0.4902
    AP for diningtable = 0.5549
    AP for dog = 0.5905
    AP for horse = 0.6871
    AP for motorbike = 0.6695
    AP for person = 0.5833
    AP for pottedplant = 0.2535
    AP for sheep = 0.5374
    AP for sofa = 0.4878
    AP for train = 0.7004
    AP for tvmonitor = 0.5754
    Mean AP = 0.5391
    5. python detect.py tiny-yolo-voc-nobn.prototxt tiny-yolo-voc-nobn.caffemodel data/dog.jpg 


# yolov2
# yolov3
