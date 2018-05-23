# 需要添加的 文件  层
## yolov1  
### 1. 带有的边框 数据层
    box_data_layer.cpp   需要修改
    box_data_layer.hpp
    
### 2. 训练误差检测层
    detection_loss_layer.cpp
    detection_loss_layer.hpp
### 3. 评估误差层
    eval_detection_layer.cpp
    eval_detection_layer.hpp
### 需要重要修改的文件
    base_data_layer.cpp
    base_data_layer.cu
    base_data_layer.hpp

## yolov2
### 1. passtrough层
    reorg_layer.cpp
    reorg_layer.cu
    reorg_layer.hpp
### 2. 区域检测 region层
    region_layer.cpp
    region_layer.cu
    region_layer.hpp
