# SSD 
# caffe code
[caffeSSD ](https://github.com/weiliu89/caffe/tree/ssd)

[VGG16与训练权重](https://download.csdn.net/download/zhayushui/10132277)

# VGG16 结构图 13个卷积层+3个全链接层=16
```asm
输入图像  3*300*300
|

    3*3*3*64  3*3卷积 步长1 填充1 64输出  卷积+relu  conv1_1
    3*3*3*64  3*3卷积 步长1 填充1 64输出  卷积+relu  conv1_2
        | 输出特征图大小 64*300*300       
    2*2最大值池化 步长2                     pool1
    
| 输出特征图大小 64*150*150

    3*3*64*128  3*3卷积 步长1 填充1 128输出 卷积+relu  conv2_1
    3*3*64*128  3*3卷积 步长1 填充1 128输出 卷积+relu  conv2_2
        | 输出特征图大小 128*150*150
    2*2最大值池化 步长2                      pool2
    
| 输出特征图大小 128*75*75

    3*3*128*256  3*3卷积 步长1 填充1 256输出                      卷积+relu  conv3_1
    3*3*128*256  3*3卷积 步长1 填充1 256输出                      卷积+relu  conv3_2
    1*1*128*256 / 3*3*128*256   3*3/1*1卷积 步长1 填充1/0 256输出 卷积+relu  conv3_3
        | 输出特征图大小 256*75*75
    2*2最大值池化 步长2                      pool3
    
| 输出特征图大小 256*38*38

    3*3*256*512  3*3卷积 步长1 填充1 512输出                      卷积+relu  conv4_1
    3*3*256*512  3*3卷积 步长1 填充1 512输出                      卷积+relu  conv4_2
    1*1*256*512 / 3*3*256*512   3*3/1*1卷积 步长1 填充1/0 512输出 卷积+relu  conv4_3
        | 输出特征图大小 256*38*38
    2*2最大值池化 步长2                      pool4
    
| 输出特征图大小 512*19*19

    3*3*512*512  3*3卷积 步长1 填充1 512输出                      空洞卷积+relu  conv5_1
    3*3*512*512  3*3卷积 步长1 填充1 512输出                      空洞卷积+relu  conv5_2
    1*1*512*512 / 3*3*512*512   3*3/1*1卷积 步长1 填充1/0 512输出 空洞卷积+relu  conv5_3
        | 输出特征图大小 512*19*19
    2*2最大值池化 步长2                      pool5
    
| 输出特征图大小 512*10*10

全连接层FC6 1024输出   3*3卷积  填充6  空洞卷积间隔6  空洞卷积 + relu
全连接层FC7 1024输出   1*1卷积  填充0                卷积 + relu 
----全连接层FC8 1000输出    没有了

somtmax 指数映射回归分类
```

# vgg16-ssd 结构

```asm
输入图像  3*300*300
|

    3*3*3*64  3*3卷积 步长1 填充1 64输出  卷积+relu  conv1_1
    3*3*3*64  3*3卷积 步长1 填充1 64输出  卷积+relu  conv1_2
        | 输出特征图大小 64*300*300       
    2*2最大值池化 步长2                     pool1
    
| 输出特征图大小 64*150*150

    3*3*64*128  3*3卷积 步长1 填充1 128输出 卷积+relu  conv2_1
    3*3*64*128  3*3卷积 步长1 填充1 128输出 卷积+relu  conv2_2
        | 输出特征图大小 128*150*150
    2*2最大值池化 步长2                      pool2
    
| 输出特征图大小 128*75*75

    3*3*128*256  3*3卷积 步长1 填充1 256输出                      卷积+relu  conv3_1
    3*3*128*256  3*3卷积 步长1 填充1 256输出                      卷积+relu  conv3_2
    1*1*128*256 / 3*3*128*256   3*3/1*1卷积 步长1 填充1/0 256输出 卷积+relu  conv3_3
        | 输出特征图大小 256*75*75
    2*2最大值池化 步长2                      pool3
    
| 输出特征图大小 256*38*38

    3*3*256*512  3*3卷积 步长1 填充1 512输出                      卷积+relu  conv4_1
    3*3*256*512  3*3卷积 步长1 填充1 512输出                      卷积+relu  conv4_2
    1*1*256*512 / 3*3*256*512   3*3/1*1卷积 步长1 填充1/0 512输出 卷积+relu  conv4_3
        | 输出特征图大小 256*38*38
    2*2最大值池化 步长2                      pool4
    
| 输出特征图大小 512*19*19

    3*3*512*512  3*3卷积 步长1 填充1 512输出                      空洞卷积+relu  conv5_1
    3*3*512*512  3*3卷积 步长1 填充1 512输出                      空洞卷积+relu  conv5_2
    1*1*512*512 / 3*3*512*512   3*3/1*1卷积 步长1 填充1/0 512输出 空洞卷积+relu  conv5_3
        | 输出特征图大小 512*19*19
    2*2最大值池化 步长2                      pool5
    
| 输出特征图大小 512*10*10

    全连接层FC6 1024输出   3*3卷积  填充6  空洞卷积间隔6   空洞卷积 + relu  ---> 1024*10*10
    全连接层FC7 1024输出    1*1卷积  填充0                卷积 + relu      ---> 1024*10*10

--------------新添加-------------
| 输出特征图大小 1024*10*10

    1*1*1024*256  1*1卷积 步长1 填充0 256输出        卷积+relu  conv6_1
    3*3*256*512   3*3卷积 步长2 填充1 512输出        卷积+relu  conv6_2
    
| 输出特征图大小 512*5*5

    1*1*512*128  1*1卷积 步长1 填充0 128输出         卷积+relu  conv7_1
    3*3*128*256  3*3卷积 步长2 填充1 256输出         卷积+relu  conv7_2
    
| 输出特征图大小 256*3*3

    1*1*256*128  1*1卷积 步长1 填充0 128输出         卷积+relu  conv8_1
    3*3*128*256  3*3卷积 步长1 填充1 256输出         卷积+relu  conv8_2
    
| 输出特征图大小 256*3*3

    1*1*256*128  1*1卷积 步长1 填充0 128输出         卷积+relu  conv9_1
    3*3*128*256  3*3卷积 步长1 填充1 256输出         卷积+relu  conv9_2
    
| 输出特征图大小 256*3*3

---------------正则化层
1.
         输入为 conv4_3层的输出： 256*38*38
         conv4_3_norm               Normalize    scale_filler  value: 20
         conv4_3_norm_mbox_loc      Convolution  3*3*256*16  16通道输出   --->  16*38*38
         conv4_3_norm_mbox_loc_perm Permute      0  2  3  1
         conv4_3_norm_mbox_loc_flat Flatten      axis: 1 
         
         
         conv4_3_norm_mbox_conf       Convolution    输入 conv4_3_norm 256*38*38
                                                     3*3*256*84  84通道输出   --->  84*38*38
         conv4_3_norm_mbox_conf_perm  Permute        0  2  3  1
         conv4_3_norm_mbox_conf_flat  Flatten        axis: 1 
         
         
         conv4_3_norm_mbox_priorbox   PriorBox      输入 conv4_3_norm 256*38*38
                                                    输入 data
                        min_size: 30.0
                        max_size: 60.0
                        aspect_ratio: 2
                        flip: true
                        clip: false
                        variance: 0.1
                        variance: 0.1
                        variance: 0.2
                        variance: 0.2
                        step: 8
                        offset: 0.5                     
              
  2.             
          输入为  fc7      输入  1024*10*10  
          fc7_mbox_loc       Convolution   3*3*1024*24  24通道输出   --->  24*10*10
          fc7_mbox_loc_perm  Permute       0  2  3  1
          fc7_mbox_loc_flat  Flatten       axis: 1
          
          fc7_mbox_conf      Convolution   3*3*24*126  126通道输出   --->  126*10*10
          fc7_mbox_conf_perm Permute       0  2  3  1
          fc7_mbox_conf_flat Flatten       axis: 1
          
          fc7_mbox_priorbox  PriorBox      输入  fc7     1024*10*10 
                                           输入 data
             
                        min_size: 60.0
                        max_size: 111.0
                        aspect_ratio: 2
                        aspect_ratio: 3
                        flip: true
                        clip: false
                        variance: 0.1
                        variance: 0.1
                        variance: 0.2
                        variance: 0.2
                        step: 16
                        offset: 0.5                           
          
3. 
          输入 conv6_2            512*5*5
          conv6_2_mbox_loc        Convolution   3*3*512*24  24通道输出   --->  24*5*5
          conv6_2_mbox_loc_perm   Permute       0  2  3  1
          conv6_2_mbox_loc_flat   Flatten       axis: 1   
          
          conv6_2_mbox_conf       Convolution   3*3*24*126  126通道输出   --->  126*5*5
          conv6_2_mbox_conf_perm  Permute       0  2  3  1
          conv6_2_mbox_conf_flat  Flatten      axis: 1
          
          conv6_2_mbox_priorbox            输入  conv6_2      512*5*5
                                           输入  data
                        min_size: 111.0
                        max_size: 162.0
                        aspect_ratio: 2
                        aspect_ratio: 3
                        flip: true
                        clip: false
                        variance: 0.1
                        variance: 0.1
                        variance: 0.2
                        variance: 0.2
                        step: 32
                        offset: 0.5              
              
 4. 
          输入 conv7_2           256*3*3
          conv7_2_mbox_loc       Convolution  3*3*256*24  24通道输出   --->  24*3*3
          conv7_2_mbox_loc_perm  Permute       0  2  3  1 
          conv7_2_mbox_loc_flat  Flatten       axis: 1   
          
          conv7_2_mbox_conf       Convolution   3*3*24*126  126通道输出   --->  126*3*3
          conv7_2_mbox_conf_perm  Permute       0  2  3  1
          conv7_2_mbox_conf_flat  Flatten      axis: 1
          
          conv7_2_mbox_priorbox            输入  conv7_2       256*3*3
                                           输入  data
                        min_size: 162.0
                        max_size: 213.0
                        aspect_ratio: 2
                        aspect_ratio: 3
                        flip: true
                        clip: false
                        variance: 0.1
                        variance: 0.1
                        variance: 0.2
                        variance: 0.2
                        step: 64
                        
5.     
          输入 conv8_2           256*3*3                             
          conv8_2_mbox_loc       Convolution  3*3*256*16  16通道输出   --->  16*3*3
          conv8_2_mbox_loc_perm  Permute       0  2  3  1 
          conv8_2_mbox_loc_flat  Flatten       axis: 1   
          
          conv8_2_mbox_conf      Convolution   3*3*16*84  84通道输出   --->  84*3*3
          conv8_2_mbox_conf_perm Permute       0  2  3  1
          conv8_2_mbox_conf_flat Flatten       axis: 1
          
          conv8_2_mbox_priorbox            输入  conv8_2       256*3*3
                                           输入  data
          
                        min_size: 213.0
                        max_size: 264.0
                        aspect_ratio: 2
                        flip: true
                        clip: false
                        variance: 0.1
                        variance: 0.1
                        variance: 0.2
                        variance: 0.2
                        step: 100
                        offset: 0.5
                        
6.       
          输入 conv8_2           256*3*3          
          conv9_2_mbox_loc       Convolution   3*3*256*16  16通道输出   --->  16*3*3
          conv9_2_mbox_loc_perm  Permute       0  2  3  1 
          conv9_2_mbox_loc_flat  Flatten       axis: 1  
          
          conv9_2_mbox_conf      Convolution   3*3*16*84  84通道输出   --->  84*3*3
          conv9_2_mbox_conf_perm Permute  
          conv9_2_mbox_conf_flat Flatten
          
          conv9_2_mbox_priorbox            输入  conv9_2       256*3*3
                                           输入  data
                        min_size: 264.0
                        max_size: 315.0
                        aspect_ratio: 2
                        flip: true
                        clip: false
                        variance: 0.1
                        variance: 0.1
                        variance: 0.2
                        variance: 0.2
                        step: 300
                        offset: 0.5
                        
7.  

    mbox_loc        type:  Concat 
    
            bottom: "conv4_3_norm_mbox_loc_flat"
            bottom: "fc7_mbox_loc_flat"
            bottom: "conv6_2_mbox_loc_flat"
            bottom: "conv7_2_mbox_loc_flat"
            bottom: "conv8_2_mbox_loc_flat"
            bottom: "conv9_2_mbox_loc_flat"
            
            top: "mbox_loc"
    
   mbox_conf        type: Concat
            bottom: "conv4_3_norm_mbox_conf_flat"
            bottom: "fc7_mbox_conf_flat"
            bottom: "conv6_2_mbox_conf_flat"
            bottom: "conv7_2_mbox_conf_flat"
            bottom: "conv8_2_mbox_conf_flat"
            bottom: "conv9_2_mbox_conf_flat"
            
            top: "mbox_conf"             
          
    mbox_priorbox     type: Concat
            bottom: "conv4_3_norm_mbox_priorbox"
            bottom: "fc7_mbox_priorbox"
            bottom: "conv6_2_mbox_priorbox"
            bottom: "conv7_2_mbox_priorbox"
            bottom: "conv8_2_mbox_priorbox"
            bottom: "conv9_2_mbox_priorbox"
            
            top: "mbox_priorbox"   

    mbox_conf_reshape   type:  Reshape      输入 mbox_conf
              dim: 0 
              dim: -1
              dim: 21
    
    mbox_conf_softmax   type: Softmax   axis: 2  输入  mbox_conf_reshape 
    
    mbox_conf_flatten   type: Flatten   axis: 1  输入  mbox_conf_softmax
    
    
    detection_out 
            type: "DetectionOutput"
            bottom: "mbox_loc"
            bottom: "mbox_conf_flatten"
            bottom: "mbox_priorbox"
            
            num_classes: 21
            share_location: true
            background_label_id: 0
            nms_param {
              nms_threshold: 0.45
              top_k: 400
            }
            code_type: CENTER_SIZE
            keep_top_k: 200
            confidence_threshold: 0.01   
```   
