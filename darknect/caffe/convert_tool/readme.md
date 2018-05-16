# darknet yolo cfg 转换到 caffe model
# yolov1
## 转模型文件 
    python yolo_cfg_to_caffe_prototxt_v1.py yolov1_test.cfg yolov1_caffe_test.prototxt
## 可视化
    cd caffe 
    python python/draw_net.py models/yolov1_caffe_test.prototxt yolov1_caffenet.png  
    open yolov1_caffenet.png
## 转weight文件
    python yolo_weight_to_caffemodel_v1.py -m yolov1_caffe_test.prototxt -w yolov1.weights -o yolov1.caffemodel

# yolov2
# yolov3
