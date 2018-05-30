# 基于.weights转到caffemodel下进行训练

# 模型转换 见
[yolo_weight_to_caffemodel_v2](https://github.com/Ewenwan/MVision/blob/master/darknect/caffe/convert_tool/yolo_weight_to_caffemodel_v2.py)

# yolov2.weights 去官网下载

# 训练基于 yolov2.weights 作为初始权重开始训练效果会好

# 训练
    ./my_train_darknet_v2.sh
# 检测图片
    python2 show_dect_yolo_v2.py
# 测试数据集
     python2 test_yolo_v2_write_result.py
