# 基于SSD目标价检测框架使用 SuqeezeNet骨骼网络
# 参考
[SqueezeNet-SSD](https://github.com/chuanqi305/SqueezeNet-SSD)

# 预训练的权重 
[squeezenet_v1.1.caffemodel](https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel)

# squeezeNet 1.1网络结构
```asm
3*300*300 图像输入
0. conv1  3*3*3*64 3*3卷积 3通道输入 4通道输出 滑动步长2  relu激活  输出：150*150*64                           
1. 最大值池化 maxpool1 3*3池化核尺寸 滑动步长2                      输出：75*75*64          
2. fire2 
       squeeze层 1*1*64*16 16个输出通道  1*1卷积+relu，
       expand层  1*1*16*64 64个输出通道  1*1卷积+relu，
                 3*3*16*64 64个输出通道  3*3卷积+relu，
       concat expand1*1 + expand3*3 ---> 75*75*128
   
3. fire3 
       squeeze层 1*1*128*16 16个输出通道， 1*1卷积+relu，
       expand层  1*1*16*64  64个输出通道  1*1卷积+relu，
                 3*3*16*64  64个输出通道  3*3卷积+relu，
       concat expand1*1 + expand3*3 ---> 75*75*128
       
4. 最大值池化 maxpool3  3*3池化核尺寸 滑动步长2  ---> 38*38*128
 
5. fire4 
       squeeze层 1*1*128*32  32个输出通道， 1*1卷积+relu，
       expand层  1*1*32*128  128个输出通道  1*1卷积+relu，
                 3*3*32*128 128个输出通道  3*3卷积+relu，
       concat expand1*1 + expand3*3 ---> 38*38*256

7. fire5 
       squeeze层 1*1*256*32  32个输出通道， 1*1卷积+relu，
       expand层  1*1*32*128  128个输出通道  1*1卷积+relu，
                 3*3*32*128 128个输出通道  3*3卷积+relu，
       concat expand1*1 + expand3*3 ---> 38*38*256
       
8. 最大值池化 maxpool5  3*3池化核尺寸 滑动步长2  ---> 19*19*256
 
9. fire6  
       squeeze层 1*1*256*48  48个输出通道， 1*1卷积+relu，
       expand层  1*1*48*192  192个输出通道  1*1卷积+relu，
                 3*3*48*192  192个输出通道  3*3卷积+relu，
       concat expand1*1 + expand3*3 ---> 19*19*384
       
10. fire7  
       squeeze层 1*1*384*48  48个输出通道， 1*1卷积+relu，
       expand层  1*1*48*192  192个输出通道  1*1卷积+relu，
                 3*3*48*192  192个输出通道  3*3卷积+relu，
       concat expand1*1 + expand3*3 ---> 19*19*384
       
11. fire8 
       squeeze层 1*1*384*64  64个输出通道， 1*1卷积+relu，
       expand层  1*1*64*256  256个输出通道  1*1卷积+relu，
                 3*3*64*256  256个输出通道  3*3卷积+relu，
       concat expand1*1 + expand3*3 ---> 19*19*512
       
12.  
       squeeze层 1*1*512*64  64个输出通道， 1*1卷积+relu，
       expand层  1*1*64*256  256个输出通道  1*1卷积 + bn (bn+scale) + relu ，
                 3*3*64*256  256个输出通道  3*3卷积 + bn (bn+scale) + relu，
       concat expand1*1 + expand3*3 ---> 19*19*512
       
13. maxpool8 最大值池化 maxpool1 3*3池化核尺寸 滑动步长2
14. fire9 squeeze层 64个输出通道， expand层  256个输出通道
15. 随机失活层 dropout 神经元以0.5的概率不输出
16. conv10 类似于全连接层 1*1的点卷积 将输出通道 固定为 1000类输出 + relu激活
17. avgpool10 13*13的均值池化核尺寸 13*13*1000 ---> 1*1*1000
18. softmax归一化分类概率输出 
```

# VGG16 结构图 13个卷积层+3个全链接层=16
```asm
输入图像  3*300*300
|

1.  3*3*3*64  3*3卷积 步长1 填充1 64输出                         卷积+relu  conv1_1
    3*3*3*64  3*3卷积 步长1 填充1 64输出                         卷积+relu  conv1_2
        | 输出特征图大小 64*300*300       
    2*2最大值池化 步长2                     pool1
    
| 输出特征图大小 64*150*150

2.  3*3*64*128  3*3卷积 步长1 填充1 128输出                      卷积+relu  conv2_1
    3*3*64*128  3*3卷积 步长1 填充1 128输出                      卷积+relu  conv2_2
        | 输出特征图大小 128*150*150
    2*2最大值池化 步长2                      pool2
    
| 输出特征图大小 128*75*75

3.  3*3*128*256  3*3卷积 步长1 填充1 256输出                      卷积+relu  conv3_1
    3*3*128*256  3*3卷积 步长1 填充1 256输出                      卷积+relu  conv3_2
    1*1*128*256 / 3*3*128*256   3*3/1*1卷积 步长1 填充1/0 256输出 卷积+relu  conv3_3
        | 输出特征图大小 256*75*75
    2*2最大值池化 步长2                      pool3
    
| 输出特征图大小 256*38*38

4.  3*3*256*512  3*3卷积 步长1 填充1 512输出                      卷积+relu  conv4_1
    3*3*256*512  3*3卷积 步长1 填充1 512输出                      卷积+relu  conv4_2
    1*1*256*512 / 3*3*256*512   3*3/1*1卷积 步长1 填充1/0 512输出 卷积+relu  conv4_3
    
        | 输出特征图大小 512*38*38
        
    2*2最大值池化 步长2                      pool4
    
| 输出特征图大小 512*19*19

5.  3*3*512*512  3*3卷积 步长1 填充1 512输出                      空洞卷积+relu  conv5_1
    3*3*512*512  3*3卷积 步长1 填充1 512输出                      空洞卷积+relu  conv5_2
    1*1*512*512 / 3*3*512*512   3*3/1*1卷积 步长1 填充1/0 512输出 空洞卷积+relu  conv5_3
        | 输出特征图大小 512*19*19
    3*3最大值池化 步长1   pad: 1             pool5
    
| 输出特征图大小 512*19*19

全连接层FC6 1024输出   3*3卷积  填充6  空洞卷积间隔6  空洞卷积 + relu
        | 输出特征图大小 1024*19*19
全连接层FC7 1024输出   1*1卷积  填充0                卷积 + relu 
        | 输出特征图大小 1024*19*19
----全连接层FC8 1000输出    没有了

somtmax 指数映射回归分类
```

# vgg16-ssd结构
```asm
3*300*300 图像输入
0. conv1  3*3*3*64 3*3卷积 3通道输入 4通道输出 滑动步长2  relu激活  输出：150*150*64                           
1. 最大值池化 maxpool1 3*3池化核尺寸 滑动步长2                      输出：75*75*64          
2. fire2 
       squeeze层 1*1*64*16 16个输出通道  1*1卷积+relu，
       expand层  1*1*16*64 64个输出通道  1*1卷积+relu，
                 3*3*16*64 64个输出通道  3*3卷积+relu，
       concat expand1*1 + expand3*3 ---> 75*75*128
   
3. fire3 
       squeeze层 1*1*128*16 16个输出通道， 1*1卷积+relu，
       expand层  1*1*16*64  64个输出通道  1*1卷积+relu，
                 3*3*16*64  64个输出通道  3*3卷积+relu，
       concat expand1*1 + expand3*3 ---> 75*75*128
       
4. 最大值池化 maxpool3  3*3池化核尺寸 滑动步长2  ---> 38*38*128
 
5. fire4 
       squeeze层 1*1*128*32  32个输出通道， 1*1卷积+relu，
       expand层  1*1*32*128  128个输出通道  1*1卷积+relu，
                 3*3*32*128 128个输出通道  3*3卷积+relu，
       concat expand1*1 + expand3*3 ---> 38*38*256

7. fire5 
       squeeze层 1*1*256*32  32个输出通道， 1*1卷积+relu，
       expand层  1*1*32*128  128个输出通道  1*1卷积+relu，
                 3*3*32*128 128个输出通道  3*3卷积+relu，
       concat expand1*1 + expand3*3 ---> 38*38*256
       
8. 最大值池化 maxpool5  3*3池化核尺寸 滑动步长2  ---> 19*19*256
 
9. fire6  
       squeeze层 1*1*256*48  48个输出通道， 1*1卷积+relu，
       expand层  1*1*48*192  192个输出通道  1*1卷积+relu，
                 3*3*48*192  192个输出通道  3*3卷积+relu，
       concat expand1*1 + expand3*3 ---> 19*19*384
       
10. fire7  
       squeeze层 1*1*384*48  48个输出通道， 1*1卷积+relu，
       expand层  1*1*48*192  192个输出通道  1*1卷积+relu，
                 3*3*48*192  192个输出通道  3*3卷积+relu，
       concat expand1*1 + expand3*3 ---> 19*19*384
       
11. fire8 
       squeeze层 1*1*384*64  64个输出通道， 1*1卷积+relu，
       expand层  1*1*64*256  256个输出通道  1*1卷积+relu，
                 3*3*64*256  256个输出通道  3*3卷积+relu，
       concat expand1*1 + expand3*3 ---> 19*19*512
       
12. fire9  
       squeeze层 1*1*512*64  64个输出通道， 1*1卷积+relu，
       expand层  1*1*64*256  256个输出通道  1*1卷积 + bn (bn+scale) + relu ，
                 3*3*64*256  256个输出通道  3*3卷积 + bn (bn+scale) + relu，
       concat expand1*1 + expand3*3 ---> 19*19*512
       
13. 最大值池化 maxpool9  3*3池化核尺寸 滑动步长2  ---> 10*10*512
 
#===================== add ===================================       
14. fire10  
       squeeze层 1*1*512*96  96个输出通道， 1*1卷积 + bn (bn+scale) + relu，
       expand层  1*1*96*384  384个输出通道  1*1卷积 + bn (bn+scale) + relu ，
                 3*3*96*384  384个输出通道  3*3卷积 + bn (bn+scale) + relu，
       concat expand1*1 + expand3*3 ---> 10*10*768
       
15. 最大值池化 maxpool10  3*3池化核尺寸 滑动步长2  ---> 5*5*768

15. fire11  
       squeeze层 1*1*768*96  96个输出通道， 1*1卷积 + bn (bn+scale) + relu，
       expand层  1*1*96*384  384个输出通道  1*1卷积 + bn (bn+scale) + relu ，
                 3*3*96*384  384个输出通道  3*3卷积 + bn (bn+scale) + relu，
       concat expand1*1 + expand3*3 ---> 5*5*768 
16. conv12  
     conv12_1  1*1*768*128  128个输出通道， 1*1卷积 + bn (bn+scale) + relu，
     conv12_2  3*3*128*256  256个输出通道， 3*3卷积 + bn (bn+scale) + relu，
                            步长2
                                   ---> 2*2*256 
16. conv13  
     conv13_1  1*1*256*64   64个输出通道，  1*1卷积 + bn (bn+scale) + relu，
     conv13_2  3*3*64*128   128个输出通道， 3*3卷积 + bn (bn+scale) + relu，
                            步长2
                                   ---> 1*1*128 
#==================FPN==========================                        
检测输出

fire5
=========bn 归一化 ==========================
    fire5/bn   输入 fire5 concat  38*38*256
    type: "BatchNorm"
    + 
    fire5/scale
    type: "Scale"
    输出 "fire5/normal"
    
    #  位置
    输入 "fire5/normal"
    fire5_mbox_loc  type: "Convolution"  3*3*256*16  输出 38*38*16
    fire5_mbox_loc_perm    Permute      0 2 3 1
    fire5_mbox_loc_flat    Flatten      axis: 1
    
    # 类别置信度
    输入 "fire5/normal"
    fire5_mbox_conf       Convolution  3*3*256*84  输出 38*38*84
    fire5_mbox_conf_perm  Permute      0 2 3 1
    fire5_mbox_conf_flat  Flatten      axis: 1
    
    # 边框
    fire5_mbox_priorbox    PriorBox   输入 fire5/normal 和 data
            min_size: 21.0
            max_size: 45.0
            aspect_ratio: 2.0
            flip: true
            clip: false
            variance: 0.1
            variance: 0.1
            variance: 0.2
            variance: 0.2
            step: 8
            
mbox fire9   19*19*512    
============================================
    #  位置
    输入 "fire9/concat"
    fire9_mbox_loc  type: "Convolution"  3*3*512*24  输出 19*19*24
    fire9_mbox_loc_perm    Permute      0 2 3 1
    fire9_mbox_loc_flat    Flatten      axis: 1
    
    # 类别置信度
    输入 "fire9/concat"
    fire9_mbox_conf       Convolution  3*3*512*126  输出 38*38*126
    fire9_mbox_conf_perm  Permute      0 2 3 1
    fire9_mbox_conf_flat  Flatten      axis: 1
    
    # 边框
    fire9_mbox_priorbox    PriorBox   输入 "fire9/concat" 和 data
            min_size: 45.0
            max_size: 99.0
            aspect_ratio: 2.0
            aspect_ratio: 3.0
            flip: true
            clip: false
            variance: 0.1
            variance: 0.1
            variance: 0.2
            variance: 0.2
            step: 16

mbox fire10   10*10*768  
============================================
    #  位置
    输入 "fire10/concat"
    fire10_mbox_loc  type: "Convolution"  3*3*768*24  输出 10*10*24
    fire10_mbox_loc_perm    Permute      0 2 3 1
    fire10_mbox_loc_flat    Flatten      axis: 1
    
    # 类别置信度
    输入 "fire10/concat"
    fire10_mbox_conf       Convolution  3*3*768*126  输出 10*10*126
    fire10_mbox_conf_perm  Permute      0 2 3 1
    fire10_mbox_conf_flat  Flatten      axis: 1
    
    # 边框
    fire10_mbox_priorbox    PriorBox   输入 "fire10/concat" 和 data
            min_size: 99.0
            max_size: 153.0
            aspect_ratio: 2.0
            aspect_ratio: 3.0
            flip: true
            clip: false
            variance: 0.1
            variance: 0.1
            variance: 0.2
            variance: 0.2
            step: 32

mbox fire11   5*5*768 
============================================
    #  位置
    输入 "fire11/concat"
    fire11_mbox_loc  type: "Convolution"  3*3*768*24  输出 5*5*24
    fire11_mbox_loc_perm    Permute      0 2 3 1
    fire11_mbox_loc_flat    Flatten      axis: 1
    
    # 类别置信度
    输入 "fire9/concat"
    fire11_mbox_conf       Convolution  3*3*768*126  输出 5*5*126
    fire11_mbox_conf_perm  Permute      0 2 3 1
    fire11_mbox_conf_flat  Flatten      axis: 1
    
    # 边框
    fire11_mbox_priorbox    PriorBox   输入 "fire11/concat" 和 data
            min_size: 153.0
            max_size: 207.0
            aspect_ratio: 2.0
            aspect_ratio: 3.0
            flip: true
            clip: false
            variance: 0.1
            variance: 0.1
            variance: 0.2
            variance: 0.2
            step: 64

 mbox fire12   2*2*256  
============================================
    #  位置
    输入 "fire11/concat"
    fire12_mbox_loc  type: "Convolution"  3*3*256*24  输出 2*2*24
    fire12_mbox_loc_perm    Permute      0 2 3 1
    fire12_mbox_loc_flat    Flatten      axis: 1
    
    # 类别置信度
    输入 "fire11/concat"
    fire12_mbox_conf       Convolution  3*3*256*126  输出 2*2*126
    fire12_mbox_conf_perm  Permute      0 2 3 1
    fire12_mbox_conf_flat  Flatten      axis: 1
    
    # 边框
    fire12_mbox_priorbox    PriorBox   输入 "fire12/concat" 和 data
            min_size: 207.0
            max_size: 261.0
            aspect_ratio: 2.0
            aspect_ratio: 3.0
            flip: true
            clip: false
            variance: 0.1
            variance: 0.1
            variance: 0.2
            variance: 0.2
            step: 100
            
            
 mbox fire13   1*1*128 
============================================
    #  位置
    输入 "fire11/concat"
    fire13_mbox_loc  type: "Convolution"  3*3*128*16  输出 1*1*16
    fire13_mbox_loc_perm    Permute      0 2 3 1
    fire13_mbox_loc_flat    Flatten      axis: 1
    
    # 类别置信度
    输入 "fire11/concat"
    fire13_mbox_conf       Convolution  3*3*128*84  输出 1*1*84
    fire13_mbox_conf_perm  Permute      0 2 3 1
    fire13_mbox_conf_flat  Flatten      axis: 1
    
    # 边框
    fire13_mbox_priorbox    PriorBox   输入 "fire13/concat" 和 data
              min_size: 261.0
              max_size: 315.0
              aspect_ratio: 2.0
              flip: true
              clip: false
              variance: 0.1
              variance: 0.1
              variance: 0.2
              variance: 0.2
              step: 300

#================== 检测结果结合 concat
# 位置
  name: "mbox_loc"
  type: "Concat"
  bottom: "fire5_mbox_loc_flat"
  bottom: "fire9_mbox_loc_flat"
  bottom: "fire10_mbox_loc_flat"
  bottom: "fire11_mbox_loc_flat"
  bottom: "conv12_2_mbox_loc_flat"
  bottom: "conv13_2_mbox_loc_flat"
  top: "mbox_loc"   
  
# 置信度
  name: "mbox_conf"
  type: "Concat"
  bottom: "fire5_mbox_conf_flat"
  bottom: "fire9_mbox_conf_flat"
  bottom: "fire10_mbox_conf_flat"
  bottom: "fire11_mbox_conf_flat"
  bottom: "conv12_2_mbox_conf_flat"
  bottom: "conv13_2_mbox_conf_flat"
  top: "mbox_conf"

# 边框
  name: "mbox_priorbox"
  type: "Concat"
  bottom: "fire5_mbox_priorbox"
  bottom: "fire9_mbox_priorbox"
  bottom: "fire10_mbox_priorbox"
  bottom: "fire11_mbox_priorbox"
  bottom: "conv12_2_mbox_priorbox"
  bottom: "conv13_2_mbox_priorbox"
  top: "mbox_priorbox"

# 总loss
  name: "mbox_loss"
  type: "MultiBoxLoss"
  bottom: "mbox_loc"
  bottom: "mbox_conf"
  bottom: "mbox_priorbox"
  bottom: "label"
  top: "mbox_loss"
  
  multibox_loss_param {
    loc_loss_type: SMOOTH_L1
    conf_loss_type: SOFTMAX
    loc_weight: 1.0
    num_classes: 21
    share_location: true
    match_type: PER_PREDICTION
    overlap_threshold: 0.5
    use_prior_for_matching: true
    background_label_id: 0
    use_difficult_gt: true
    neg_pos_ratio: 3.0
    neg_overlap: 0.5
    code_type: CENTER_SIZE
    ignore_cross_boundary_bbox: false
    mining_type: MAX_NEGATIVE}
    

```
