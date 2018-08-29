# SqueezeNet

[NCNN上的inference分类](https://github.com/Ewenwan/MVision/tree/master/CNN/HighPerformanceComputing)

[参考博客](https://blog.csdn.net/csdnldp/article/details/78648543#fn:1)

[论文](https://arxiv.org/pdf/1602.07360.pdf)

[论文源码 caffe model](https://github.com/DeepScale/SqueezeNet)

    注：代码只放出了prototxt文件和训练好的caffemodel，
    因为整个网络都是基于caffe的，有这两样东西就足够了。 
 

[ShuffleNet在Caffe框架下的实现](https://blog.csdn.net/Chris_zhangrx/article/details/78277957)

[PyTorch实现的SqeezeNet](https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py)

# SqueezeNet的工作为以下几个方面：
    1. 提出了新的网络架构Fire Module，通过减少参数来进行模型压缩
    2. 使用其他方法对提出的SqeezeNet模型进行进一步压缩
    3. 对参数空间进行了探索，主要研究了压缩比和3∗3卷积比例的影响
#  常用的模型压缩技术有：
    1. 奇异值分解(singular value decomposition (SVD))
    2. 网络剪枝（Network Pruning）：使用网络剪枝和稀疏矩阵
    3. 深度压缩（Deep compression）：使用网络剪枝，数字化和huffman编码
    4. 硬件加速器（hardware accelerator）
    1. 贝叶斯优化
    2. 模拟退火
    3. 随机搜索 
    4. 遗传算法
    
## SqueezeNet 简化网络模型参数的 设计

## 使用以下三个策略来减少SqueezeNet设计参数

    1. 使用1∗1卷积代替3∗3 卷积：参数减少为原来的1/9
    2. 减少输入通道数量：这一部分使用squeeze layers来实现
    3. 将欠采样操作延后，可以给卷积层提供更大的激活图：更大的激活图保留了更多的信息，可以提供更高的分类准确率
    其中，1 和 2 可以显著减少参数数量，3 可以在参数数量受限的情况下提高准确率。
    
## SqueezeNet 的核心模块 Fire Module
    1. 只使用1∗1 卷积 filter 构建 squeeze convolution layer    减少参数 策略1 
    2. 使用1∗1 和3∗3 卷积 filter的组合 构建的 expand layer
    3. squeeze convolution layer 中 1∗1 卷积 filter数量可调 s1
       expand layer  中 1∗1 卷积 filter数量可调  e1
       expand layer  中 3∗3 卷积 filter数量可调  e2
    4. s1 < e1 + e2                                           减少参数 策略2 
    
### Fire Module 结构

                                     ----->  1 * 1卷积------|
                 输入----->1 * 1卷积（全部）                   ---> concat 通道合并 -------> 输出
                                     ----->  3 * 3卷积  ----|
#### 与Inception module 区别
                                 部分输出 ----->  1 * 1卷积------|
                输入----->1 * 1卷积                               ---> concat 通道合并 -------> 输出 
                                 部分输出 ----->  3 * 3卷积  ----|


## SqueezeNet 网络结构 
    1. SqueezeNet以卷积层（conv1）开始， 
    2. 接着使用8个Fire modules (fire2-9)，
    3. 最后以卷积层（conv10）结束卷积特征提取
    4. 再通过 全局均值池化 + softmax分类得到结果
    
    每个fire module中的filter数量逐渐增加，
    并且在conv1, fire4, fire8, 和 conv10这几层 之后 使用步长为2的max-pooling，
    即将池化层放在相对靠后的位置，这使用了以上的策略（3）。
### 以fire2模块为例
    1. maxpool1层的输出为55∗55∗96 55∗55∗96，一共有96个通道输出。
    2. 之后紧接着的Squeeze层有16个1∗1∗96 卷积核，96个通道输入，16个通道输出，输出尺寸为 55*55*16
    3. 之后将输出分别送到expand层中的1∗1∗16 （64个）和3∗3∗16（64个）进行处理，注意这里不对16个通道进行切分。
    4. 对3∗3∗16 的卷积输入进行尺寸为１的zero padding，分别得到55∗55∗64 和 55∗55∗64 大小相同的两个feature map。
    5. 将这两个feature map连接到一起得到55∗55∗128 大小的feature map。

##  一些改进版本
    还有改进版本 simple bypass 以及  complex bypass的改进版本。
    在中间某些层之间增加 残差网络的结构  结合不同层级的 特征图
## 一些注意点
    以下是网络设计中的一些要点：  
    （1）为了使 1∗1  和 3∗3 filter输出的结果又相同的尺寸，在expand modules中，给3∗3filter的原始输入添加一个像素的边界（zero-padding）。  
    （2）squeeze 和 expand layers中都是用ReLU作为激活函数  
    （3）在fire9 module之后，使用Dropout，比例取50% 
    （4）注意到SqueezeNet中没有全连接层，这借鉴了Network in network的思想  
    （5）训练过程中，初始学习率设置为0.04，，在训练过程中线性降低学习率。

# 网络结构 
```asm
0. conv1  7*7*3*96 7*7卷积 3通道输入 96通道输出 滑动步长2  relu激活                                 
1. 最大值池化 maxpool1 3*3池化核尺寸 滑动步长2           
2. fire2 squeeze层 16个输出通道， expand层  64个输出通道  concat --> 128
3. fire3 squeeze层 16个输出通道， expand层  64个输出通道  concat --> 128
4. fire4 squeeze层 32个输出通道， expand层  128个输出通道 concat --> 256
5. maxpool4 最大值池化 maxpool1 3*3池化核尺寸 滑动步长2
6. fire5 squeeze层 32个输出通道， expand层  128个输出通道 concat --> 256
7. fire6 squeeze层 48个输出通道， expand层  192个输出通道 concat --> 384
8. fire7 squeeze层 48个输出通道， expand层  192个输出通道 concat --> 384
9. fire8 squeeze层 64个输出通道， expand层  256个输出通道 concat --> 512
10. maxpool8 最大值池化 maxpool1 3*3池化核尺寸 滑动步长2
11. fire9 squeeze层 64个输出通道， expand层  256个输出通道concat --> 512
12. 随机失活层 dropout 神经元以0.5的概率不输出
13. conv10 类似于全连接层 1*1的点卷积 将输出通道 固定为 1000类输出 + relu激活
14. avgpool10 13*13的均值池化核尺寸 13*13*1000 ---> 1*1*1000
15. softmax归一化分类概率输出 
```

---
title: squeeze_net的模型优化

date: 2017/7/20 12:04:12

categories:
- 深度学习
tags:
- deeplearning
- 梯度下降法
- 正则化
- 激活函数
- 神经网络
---


<div class="github-widget" data-repo="DragonFive/deep-learning-exercise"></div>

SqueezeNet主要是为了降低CNN模型参数数量而设计的。没有提高运行速度。

<!--more-->

使用的squeezenet的pre-trained model来自[SqueezeNet repo](https://github.com/DeepScale/SqueezeNet)

# 设计原则

（1）替换3x3的卷积kernel为**1x1的卷积kernel**

卷积模板的选择，从12年的AlexNet模型一路发展到2015年底Deep Residual Learning模型，基本上卷积大小都选择在3x3了，因为其有效性，以及设计简洁性。本文替换3x3的卷积kernel为1x1的卷积kernel可以让参数缩小9X。但是为了不影响识别精度，并不是全部替换，而是一部分用3x3，一部分用1x1。具体可以看后面的模块结构图。

（2）减少输入3x3卷积的input feature map数量 
如果是conv1-conv2这样的直连，那么实际上是没有办法**减少conv2的input feature map**数量的。所以作者巧妙地把原本一层conv分解为两层，并且封装为一个**Fire Module**。


（3）**减少pooling **
这个观点在很多其他工作中都已经有体现了，比如GoogleNet以及Deep Residual Learning。

同时也替换fc层为 global avg pooling层
## Fire Module

Fire Module是本文的核心构件，思想非常简单，就是将原来简单的一层conv层变成两层：**squeeze层+expand层**，各自带上Relu激活层。在squeeze层里面全是1x1的卷积kernel，数量记为S11；在expand层里面有1x1和3x3的卷积kernel，数量分别记为E11和E33，**要求S11 < input map number即满足上面的设计原则（2）**。expand层之后将1x1和3x3的卷积output feature maps在**channel维度拼接起来**。

![squeezenet][1]


## 总体网络架构



![squeezenet网络结构][2]

共有**9层fire module**，中间穿插一些max pooling，最后是**global avg pooling代替了fc层**（参数大大减少）。在开始和最后还有两层最简单的单层conv层，保证输入输出大小可掌握。

![squeezenet 参数数量][3]

比较了alexnet，可以看到准确率差不多的情况下，squeezeNet模型参数数量显著降低了（下表倒数第三行），参数减少50X；如果再加上deep compression技术，压缩比可以达到461X！

![参数量比较][4]

## 延迟下采样操作 


在alexnet里，第一层卷积层的stride = 4, 直接下采样了4倍。在一般的CNN中，一般卷积层、池化层都会有下采样(stride>1), 甚至在**前面几层网络的下采样比例**会比较大，这样会导致后面基层的神经元的激活映射区域减少，为了提高精度设计下采样层延迟的慢一点，这也是squeezenet不能提高速度的真正原因。



# reference
[深度学习（六十二）SqueezeNet网络设计思想笔记](http://blog.csdn.net/hjimce/article/details/72809131)

[ 深度学习方法（七）：最新SqueezeNet 模型详解](http://blog.csdn.net/xbinworld/article/details/50897870)


  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1502707058373.jpg
  [2]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1502707143096.jpg
  [3]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1502707297973.jpg
  [4]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1502707773916.jpg
  
