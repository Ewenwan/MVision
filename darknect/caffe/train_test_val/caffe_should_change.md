# caffe 需要做的修改

# 1. 数据转换 时

    caffe/util/io.hpp 
    caffe/src/caffe/util/io.cpp 做一定修改

    /tools v添加
    convert_box_data.cpp

# 2.训练时 
    a. 数据层
    caffe/include/caffe/layer 下添加 
      box_data_layer.hpp
    caffe/src/caffe/layers/  下添加 
      box_data_layer.cpp
    b. 检测loss层
      detection_loss_layer.hpp
      detection_loss_layer.cpp
    c. 评估检测结果层
      eval_detection_layer.hpp
      eval_detection_layer.cpp
    d. 数据读取层
    caffe/include/caffe 下添加
        data_reader.hpp
    caffe/src/caffe/ 下添加 
        data_reader.cpp

