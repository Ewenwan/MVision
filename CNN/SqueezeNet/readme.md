# SqueezeNet

[参考博客](https://blog.csdn.net/csdnldp/article/details/78648543#fn:1)

[论文](https://arxiv.org/pdf/1602.07360.pdf)

[论文源码 caffe model](https://github.com/DeepScale/SqueezeNet)

# SqueezeNet的工作为以下几个方面：
    1. 提出了新的网络架构Fire Module，通过减少参数来进行模型压缩
    2. 使用其他方法对提出的SqeezeNet模型进行进一步压缩
    3. 对参数空间进行了探索，主要研究了压缩比和3∗3卷积比例的影响
#  常用的模型压缩技术有：
    1. 奇异值分解(singular value decomposition (SVD))
    2. 网络剪枝（Network Pruning）：使用网络剪枝和稀疏矩阵
    3. 深度压缩（Deep compression）：使用网络剪枝，数字化和huffman编码
    4. 硬件加速器（hardware accelerator）
    
# 深度网络结构 优化设计 超参数繁多，深度神经网络具有很大的设计空间（design space）。

# 通常进行设计空间探索的方法有： 
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
 
## SqueezeNet 网络结构 
    1. SqueezeNet以卷积层（conv1）开始， 
    2. 接着使用8个Fire modules (fire2-9)，
    3. 最后以卷积层（conv10）结束卷积特征提取
    4. 再通过 全局均值池化 + softmax分类得到结果
    
    每个fire module中的filter数量逐渐增加，
    并且在conv1, fire4, fire8, 和 conv10这几层 之后 使用步长为2的max-pooling，
    即将池化层放在相对靠后的位置，这使用了以上的策略（3）。
    
    
    
   
