# caffe 需要做的修改

# 1. 数据转换 时

    caffe/util/io.hpp 
    caffe/src/caffe/util/io.cpp 做一定修改

    /tools v添加
    convert_box_data.cpp

# 2.训练时 
    caffe/include/caffe/layer 下添加 
      box_data_layer.hpp
    caffe/src/caffe/layers/  下添加 
      box_data_layer.cpp

      detection_loss_layer.hpp
      detection_loss_layer.cpp

      eval_detection_layer.hpp
      eval_detection_layer.cpp

    caffe/include/caffe 下添加
    data_reader.hpp
