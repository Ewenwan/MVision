# MobileNet 模型结构
# 深度可分解卷积  逐通道卷积 + 普通点卷积和并个通道特征 降低卷积时间
# MobileNet v1 总共28层（1 + 2 × 13 + 1 = 28） 
[参考理解](https://blog.csdn.net/wfei101/article/details/78310226)

[参考代码](https://github.com/Zehaos/MobileNet/blob/master/nets/mobilenet.py)

[V1 论文地址](https://arxiv.org/pdf/1704.04861.pdf)

[MobileNet V2 论文地址](https://arxiv.org/pdf/1801.04381.pdf)

[有tensorflow的实现 v1](https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md)

[caffe也有人实现 v1 v2](https://github.com/shicai/MobileNet-Caffe)

[非官方Caffe v1](https://github.com/shicai/MobileNet-Caffe)

[MobileNetV2-pytorch](https://github.com/Randl/MobileNetV2-pytorch)

[MobileNetV2-caffe](https://github.com/suzhenghang/MobileNetv2/tree/master/.gitignore)

[MobileNet-v2-caffe](https://github.com/austingg/MobileNet-v2-caffe)

[MobileNet-v2-tf](https://github.com/neuleaf/MobileNetV2)

[MobileNet-SSD目标检测](https://github.com/chuanqi305/MobileNet-SSD)

[MobileNet-量化](https://arxiv.org/pdf/1803.08607.pdf)

      是Google针对手机等嵌入式设备(  移动和嵌入式视觉应用 mobile and embedded vision applications)
      提出的一种轻量级的深层神经网络，取名为MobileNets。
      个人感觉论文所做工作偏向于模型压缩方面，
      核心思想就是卷积核的巧妙分解，可以有效减少网络参数。

       MobileNets是基于一个流线型的架构 streamlined architecture 
       轻量级的深层神经网络 light  weight  deep neural  networks.
       一些嵌入式平台上的应用比如机器人和自动驾驶，它们的硬件资源有限，
       就十分需要一种轻量级、低延迟（同时精度尚可接受）的网络模型

       在建立小型和有效的神经网络上，已经有了一些工作，比如SqueezeNet，Google Inception，Flattened network等等。
       大概分为压缩预训练模型和直接训练小型网络两种。
       MobileNets主要关注优化延迟，同时兼顾模型大小，不像有些模型虽然参数少，但是也慢的可以。
 
##  因此，在小型化方面常用的手段有：
      （1）卷积核分解，使用1×N和N×1的卷积核代替N×N的卷积核
      （2）使用bottleneck结构，以SqueezeNet为代表
      （3）以低精度浮点数保存，例如Deep Compression
      （4）冗余卷积核剪枝及哈弗曼编码


      将标准卷积分解成一个深度卷积和一个点卷积（1 × 1卷积核）。深度卷积将每个卷积核应用到每一个通道，
      而1 × 1卷积用来组合通道卷积的输出。后文证明，这种分解可以有效减少计算量，降低模型大小
### 模型简化思想
      3 × 3 × 3 ×16 3*3的卷积 3通道输入  16通道输出
       ===== 3 × 3 × 1 × 3的深度卷积(3个3*3的卷积核，每一个卷积核对输入通道分别卷积后叠加输出) 输出3通道   1d卷积
       ===== + 1 × 1 × 3 ×16的1 ×1点卷积 1*1卷积核 3通道输入  16通道输出
      参数数量 75/432 = 0.17

      3*3*输入通道*输出通道 -> BN -> RELU
      =======>
      3*3*1*输入通道 -> BN -> RELU ->    1*1*输入通道*输出通道 -> BN -> RELU


### A 模型的第一个超参数，即宽度乘数：
        为了构建更小和更少计算量的网络，作者引入了宽度乘数  ，
        作用是改变输入输出通道数，减少特征图数量，让网络变瘦。
###   B 第二个超参数是分辨率乘数  ，
        分辨率乘数用来改变输入数据层的分辨率，同样也能减少参数。
### v1 网络结构 
      """
      1. 普通3d卷积层 3*3*3*round(32 * width_multiplier) 3*3卷积核 3通道输入 输出通道数量 随机确定1~32个
      2. 13个 depthwise_separable_conv2d 层 3*3*1*输入通道 -> BN -> RELU ->  1*1*输入通道*输出通道 -> BN -> RELU
      3. 均值池化层	 7*7核	+ squeeze 去掉维度为1的维
      4. 全连接层 输出  -> [N, 1000]
      5. softmax分类输出到 0~1之间
      """
### v2 结构 借鉴 ResNet结构
#### 残差模块
      f(x) + W*x
      f(x) 为 2个 3x3的卷积
      实际中，考虑计算的成本，对残差块做了计算C，即将2个3x3的卷积层替换为 1x1 + 3x3 + 1x1 。
      新结构中的中间3x3的卷积层首先在一个降维1x1卷积层下减少了计算，然后在另一个1x1的卷积层下做了还原，既保持了精度又减少了计算量。
      
      _____________________________________>
      |                                     +  f(x) + x
      x-----> 1x1 + 3x3标准 + 1x1 卷积 ----->  
           压缩”→“卷积提特征”→“扩张”
#### MobileNet v2
      DW    逐通道卷积  每个卷积核之和一个通道卷积 k*k*1 卷积核数量为 输入通道数量 
      之后使用 1*1的普通卷积（点卷积） 合并个通道特征
      在v1 的 Depth-wise convolution之前多了一个1*1的“扩张”层，目的是为了提升通道数，获得更多特征；
      最后不采用Relu，而是Linear，目的是防止Relu破坏特征。
      结合 x (中间3x3DW 步长为1结合x  步长为2时不结合x )
      1. 步长为1结合x shortcut
      ___________________________________>
      |                                     -->  f(x) + x
      x-----> 1x1 + 3x3DW + 1x1 卷积 ----->  
           “扩张”→“卷积提特征”→ “压缩”
      ResNet是：压缩”→“卷积提特征”→“扩张”，MobileNetV2则是Inverted residuals,即：“扩张”→“卷积提特征”→ “压缩”

      2. 步长为2时不结合x 
      x-----> 1x1 + 3x3DW(步长为2) + 1x1 卷积 ----->   输出

#### v2 网络结构 
      1. 2D卷积块  =  2D卷积层 + BN + RELU6  3*3*3*32 步长2 32个通道输出
      2. 1个自适应残差深度可拆解模块  中间层扩张倍数为1为32*1  16个通道输出
      3. 2个自适应残差深度可拆解模块  中间层扩张倍数为6为16*6  24个通道输出
      4. 3个自适应残差深度可拆解模块  中间层扩张倍数为6为24*6  32个通道输出 
      5. 4个自适应残差深度可拆解模块  中间层扩张倍数为6为32*6  64个通道输出
      6. 3个自适应残差深度可拆解模块  中间层扩张倍数为6为64*6  96个通道输出
      7. 3个自适应残差深度可拆解模块  中间层扩张倍数为6为96*6  160个通道输出 
      8. 1个自适应残差深度可拆解模块  中间层扩张倍数为6为160*6  320个通道输出
      9. 1个 1*1点卷积块 = 1*1 PW点卷积 +  BN + RELU6 1280个通道输出
      10. 全局均值 池化 average_pooling2d
      11. 1*1 PW点卷积 后 展开成1维
      12. softmax 分类得到分类结果

      
---
title: mobile_net的模型优化

date: 2017/7/23 12:04:12

categories:
- 深度学习
tags:
- deeplearning
- 网络优化
- 神经网络
---
[TOC]


论文出自google的 MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications。
源代码和训练好的模型: [tensorflow版本](https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md)

![enter description here][1]

mobilenet 对于alexnet运行速度提高了10倍，参数量降低了50倍！而squeezenet虽然参数量也降低了50倍，但是速度提升很小。

<!--more-->



在建立小型和有效的神经网络上，已经有了一些工作，比如SqueezeNet，Google Inception，Flattened network等等。大概分为压缩预训练模型和直接训练小型网络两种。MobileNets主要关注优化延迟，同时兼顾模型大小。

# mobileNets模型结构
只有一个avg pooling层，用来替换fc层，少用fc和pooling层就能减少参数量。
## 深度可分解卷积 
MobileNets模型基于**深度可分解的卷积**，它可以**将标准卷积分解成一个深度卷积和一个点卷积（1 × 1卷积核）**。标准卷积核为：a × a × c，其中a是卷积核大小，c是卷积核的通道数，本文将其一分为二，一个卷积核是a × a × 1，一个卷积核是1 ×1 × c。简单说，就是标准卷积同时完成了**2维卷积计算和改变特征数量**两件事，本文把这两件事分开做了。后文证明，这种分解可以有效减少计算量，降低模型大小。


标准的卷积核是一步到位，直接计算输出，跨通道的意思是：包含了图征途之间的加权混合。而可分离卷积层把标准卷积层分成两个步骤：

1. 各个卷积层单独卷积 
2. $1x1$卷积核心(1,1,M,N)跨通道结合


![深度可分解的卷积][2]


首先是标准卷积，假定输入F的维度是 DF×DF×M ，经过标准卷积核K得到输出G的维度 DG×DG×N ，卷积核参数量表示为 DK×DK×M×N 。如果计算代价也用数量表示，应该为 DK×DK×M×N×DF×DF 。

现在将卷积核进行分解，那么按照上述计算公式，可得深度卷积的计算代价为 DK×DK×M×DF×DF ，点卷积的计算代价为 M×N×DF×DF 。

![参数量][3]


## 模型结构和训练 

![模型][4]

![mobilenet架构][5]




MobileNet将95％的计算时间用于有75％的参数的1×1卷积。

![1x1卷积计算量大][6]


## 宽度参数  Width Multiplier

宽度乘数 α ，作用是改变输入输出通道数，减少**特征图数量，让网络变瘦**。α 取值是0~1，应用宽度乘数可以进一步减少计算量，大约有 $α^2$ 的优化空间。在 α 参数作用下，MobileNets某一层的计算量为：$D_K×D_K×αM×D_F×D_F+αM×αN×D_F×D_F$



## 分辨率参数 Resolution Multiplier


分辨率乘数用来改变输入数据层的分辨率，同样也能减少参数。在 α 和 ρ 共同作用下，MobileNets某一层的计算量为：$D_K×D_K×αM×ρD_F×ρD_F+αM×αN×ρD_F×ρD_F$

ρ 是隐式参数，ρ 如果为{1，6/7，5/7，4/7}，则对应输入分辨率为{224，192，160，128}，ρ 参数的优化空间同样是 $ρ^2$ 左右.



没有使用这两个参数的mobilenet是vGG的1/30 
![mobilenet参数量][7]

,$\alpha = 0.5, \rho = \frac{5}{7}$时是alexnet的1/50,精度提升了0.3%

![与alexnet对比][8]




[tensorflow官网](https://www.tensorflow.org/mobile/)给出了部署方式，支持android,ios,raspberry Pi等。





**持续更新中。。。。。。。。。。。**

# reference

[github源码](https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md)

[官方的部署方式](https://www.tensorflow.org/mobile/)

[ 深度学习（六十五）移动端网络MobileNets](http://blog.csdn.net/hjimce/article/details/72831171)

[MobileNets 论文笔记](http://blog.csdn.net/Jesse_Mx/article/details/70766871)

[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications 论文理解](http://www.jianshu.com/p/2fd0c007a560)

[tensorflow训练好的模型怎么调用？](https://www.zhihu.com/question/58287577)

[如何用TensorFlow和TF-Slim实现图像分类与分割](https://www.ctolib.com/topics-101544.html)


  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1500434910512.jpg
  [2]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502675769608.jpg
  [3]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502676514289.jpg
  [4]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502677244854.jpg
  [5]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502677189961.jpg
  [6]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502677324886.jpg
  [7]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502695710122.jpg
  [8]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502696170111.jpg
  
