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

       conv1：  3*3/2  24输出  + bn + scale + relu  +   3*3/2 MAXpooling

               ----> resx1_match_conv（3*3/2 AVGpooling pad填充重要）                               --->
    ！ resx1：  ----> resx1_conv1 1*1*54  + bn + scale + relu    3*3/2 *54dw 分组=54  + 1*1*216    ---> concat -----> relu

                ----> 无降采样                                                                          --->
       resx2：  ----> resx2_conv1 1*1*60 分组=3  ->通道重排CS ->  3*3 *60dw 分组=60  + 1*1*240 分组=3    --->add  Eltwise--> relu

                 ----> 无降采样                                                                          --->
       resx3：  ----> resx3_conv1 1*1*60 分组=3  ->通道重排CS ->  3*3 *60dw 分组=60  + 1*1*240 分组=3    --->add  Eltwise--> relu

                 ----> 无降采样                                                                          --->
       resx4：  ----> resx4_conv1 1*1*60 分组=3  ->通道重排CS ->  3*3 *60dw 分组=60  + 1*1*240 分组=3    --->add  Eltwise--> relu

               ----> resx5_match_conv（3*3/2 AVGpooling pad填充重要）                               --->
    ！ resx5：  ----> resx5_conv1 1*1*60 分组=3  ->通道重排CS ->   3*3/2 *60dw 分组=60  + 1*1*240    ---> concat--> relu


                 ----> 无降采样                                                                            --->
       resx6：  ----> resx6_conv1 1*1*120 分组=3  ->通道重排CS ->  3*3 *120dw 分组=120  + 1*1*480 分组=3    --->add  Eltwise--> relu

                   ----> 无降采样                                                                            --->
       resx7：  ----> resx7_conv1 1*1*120 分组=3  ->通道重排CS ->  3*3 *120dw 分组=120  + 1*1*480 分组=3    --->add  Eltwise--> relu   

                   ----> 无降采样                                                                            --->
       resx8：  ----> resx8_conv1 1*1*120 分组=3  ->通道重排CS ->  3*3 *120dw 分组=120  + 1*1*480 分组=3    --->add  Eltwise--> relu    

                   ----> 无降采样                                                                            --->
       resx9：  ----> resx9_conv1 1*1*120 分组=3  ->通道重排CS ->  3*3 *120dw 分组=120  + 1*1*480 分组=3    --->add  Eltwise--> relu    

                   ----> 无降采样                                                                            --->
       resx10：  ----> resx10_conv1 1*1*120 分组=3  ->通道重排CS ->  3*3 *120dw 分组=120  + 1*1*480 分组=3    --->add  Eltwise--> relu 

                   ----> 无降采样                                                                            --->
       resx11：  ----> resx11_conv1 1*1*120 分组=3  ->通道重排CS ->  3*3 *120dw 分组=120  + 1*1*480 分组=3    --->add  Eltwise--> relu 

                     ----> 无降采样                                                                         --->
       resx12：  ----> resx12_conv1 1*1*120 分组=3  ->通道重排CS ->  3*3 *120dw 分组=120  + 1*1*480 分组=3    --->add  Eltwise--> relu 

                ----> resx13_match_conv（3*3/2 AVGpooling pad填充重要）                                  --->
    ！ resx13：  ----> resx13_conv1 1*1*120 分组=3  ->通道重排CS ->   3*3/2 *120dw 分组=120  + 1*1*480    ---> concat--> relu

                     ----> 无降采样                                                                        --->
       resx14：  ----> resx14_conv1 1*1*40 分组=3  ->通道重排CS ->  3*3 *240dw 分组=240  + 1*1*960 分组=3    --->add  Eltwise--> relu 

                          ----> 无降采样                                                                  --->
       resx15：  ----> resx15_conv1 1*1*40 分组=3  ->通道重排CS ->  3*3 *240dw 分组=240  + 1*1*960 分组=3    --->add  Eltwise--> relu 

                          ----> 无降采样                                                                   --->
    !  resx16：  ----> resx16_conv1 1*1*40 分组=3  ->通道重排CS ->  3*3 *240dw 分组=240  + 1*1*960 分组=3    --->add  Eltwise--> relu 

       conv18   1*1*256  + 3*3/2*512

       conv19   1*1*128  + 3*3/2*256  

       conv20   1*1*128  + 3*3/2*256  

       conv21   1*1*64  + 3*3/2*128 

    #============dect 

       resx12 ------>   loc  1*1*12
                        conf 1*1*63  (3*(20+1)=63)
                        box

       resx16 ------>   loc  1*1*24
                        conf 1*1*126  (6*(20+1)=126)
                        box

       conv18 ------>   loc  1*1*24
                        conf 1*1*126  (6*(20+1)=126)
                        box       

       conv19 ------>   loc  1*1*24
                        conf 1*1*126  (6*(20+1)=126)
                        box  

       conv20 ------>   loc  1*1*24
                        conf 1*1*126  (6*(20+1)=126)
                        box  

       conv21 ------>   loc  1*1*24
                        conf 1*1*126  (6*(20+1)=126)
                        box       

    # ==== dect concat 

       concat loc
       concat conf
       concat box


    # 一下根据不同阶段 使用不同的结构
    # dect loss                            训练train
       MultiBoxLoss

    # Reshape  +  Softmax  +  Flatten  + DetectionOutput  + DetectionEvaluate  测试test
    # Reshape  +  Softmax  +  Flatten  + DetectionOutput  开发 deploy


