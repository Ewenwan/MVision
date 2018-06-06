# ShuffleNet_ssd_caffe
[参考1](https://github.com/FreeApe/VGG-or-MobileNet-SSD/tree/master/examples/shufflenet_ssd_head_shoulder)

[参考2](https://github.com/linchaozhang/shufflenet-ssd)

# 新建通道层
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
# 
