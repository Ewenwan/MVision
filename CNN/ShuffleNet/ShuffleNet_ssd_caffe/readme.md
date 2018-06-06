# ShuffleNet_ssd_caffe
[参考1](https://github.com/FreeApe/VGG-or-MobileNet-SSD/tree/master/examples/shufflenet_ssd_head_shoulder)

[参考2](https://github.com/linchaozhang/shufflenet-ssd)

# 新建通道重排层
    添加三个文件 ：
    shuffle_channel_layer.cpp
    shuffle_channel_layer.cu
    shuffle_channel_layer.hpp
# 修改 caffe.proto文件
    message LayerParameter {
    ...
    optional ShuffleChannelParameter shuffle_channel_param = 164;
    ...
    }
    ...
    message ShuffleChannelParameter {
      optional uint32 group = 1[default = 1]; // The number of group
    }
# 重新编译
    make clean
    make all -j
    make pycaffe
# 通道重排三部曲 

| group 1  | group 2  |  group 3  |  
| :------: | :------: | :-------: |  
| 1     2  | 3     4  |  5     6  |   

Each nubmer represents a channel of the feature map  
    
## step 1: Reshape  按分组变形成列矩阵
1  2  
3  4   
5  6 
## step 2: transpose  转置
1 3 5  
2 4 6　　
## step 3: flatten    按分组数量 平滑成行向量

| group 1  | group 2  |  group 3  |  
| :-----:  | :------: | :-------: |  
| 1     3  | 5     2  |  4     6  |  


## shuffleNet 模型框架

    conv1：  3*3/2  24输出  + bn + scale + relu  +      3*3/2 MAXpooling

    ----> resx1_match_conv（3*3/2 AVGpooling） 
    ----> resx1_conv1 1*1*54  + bn + scale + relu  


