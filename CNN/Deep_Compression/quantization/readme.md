# 量化策略

![](http://file.elecfans.com/web1/M00/55/79/pIYBAFssV_SAPOcSAACWBTome1c039.png)

[其他博客参考](https://github.com/ICEORY/iceory.gitbook.io/tree/master/Network%20Quantization)

[论文合集](https://github.com/Ewenwan/MVision/blob/master/CNN/Deep_Compression/quantization/quantizedNN_paper.md)

[低数值精度深度学习推理与训练](https://software.intel.com/zh-cn/articles/lower-numerical-precision-deep-learning-inference-and-training)



# 具体量化方法
[参考](https://github.com/Ewenwan/pytorch-playground/blob/master/utee/quant.py)



```python
# 线性量化
def linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    # 一位
    if bits == 1:
        return torch.sign(input) - 1
    
    delta = math.pow(2.0, -sf)# 小数位 位宽 量化精度
    bound = math.pow(2.0, bits-1)
    min_val = - bound    # 上限制值
    max_val = bound - 1  # 下限值
    rounded = torch.floor(input / delta + 0.5)# 扩大后取整

    clipped_value = torch.clamp(rounded, min_val, max_val) * delta# 再缩回
    return clipped_value
# 对数线性量化
def log_linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input), 0.0, 0.0

    s = torch.sign(input)# 正负号
    input0 = torch.log(torch.abs(input) + 1e-20)# 求对数 获取 比特位
    v = linear_quantize(input0, sf, bits)# 对比特位进行线性量化
    v = torch.exp(v) * s# 再指数 回 原数
    return v
# 双曲正切量化
def tanh_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input)
	
    input = torch.tanh(input) # 双曲正切 映射 [-1, 1]
    input_rescale = (input + 1.0) / 2 #  再 映射到 [0, 1]
    n = math.pow(2.0, bits) - 1       # 固定比特位 放大系数
    v = torch.floor(input_rescale * n + 0.5) / n # 放大后取整
    v = 2 * v - 1 # [-1, 1]                      # 再放回原来的范围

    v = 0.5 * torch.log((1 + v) / (1 - v))       # 反双曲正切 回原数 arctanh
    return v
```


# NN的INT8计算

## 概述

	NN的INT8计算是近来NN计算优化的方向之一。
	相比于传统的浮点计算，整数计算无疑速度更快，
	而NN由于自身特性，对单点计算的精确度要求不高，
	且损失的精度还可以通过retrain的方式恢复大部分，
	因此通常的科学计算的硬件（没错就是指的GPU）并不太适合NN运算，尤其是NN Inference。

>传统的GPU并不适合NN运算，因此Nvidia也好，还是其他GPU厂商也好，通常都在GPU中又集成了NN加速的硬件，因此虽然商品名还是叫做GPU，但是工作原理已经有别于传统的GPU了。

这方面的文章以Xilinx的白皮书较为经典：

https://china.xilinx.com/support/documentation/white_papers/c_wp486-deep-learning-int8.pdf

利用Xilinx器件的INT8优化开展深度学习

## INT量化

论文：

《On the efficient representation and execution of deep acoustic models》

![](https://github.com/Ewenwan/antkillerfarm.github.com/tree/master/images/img2/INT8.png)

一个浮点数包括底数和指数两部分。将两者分开，就得到了一般的INT量化。

## UINT量化

论文：

《Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference》

![](https://github.com/Ewenwan/antkillerfarm.github.com/tree/master/images/img2/INT8_2.png)

UINT量化使用bias将数据搬移到正数区间。

这篇论文的另一个贡献在于：原先的INT8量化是针对已经训练好的模型。而现在还可以在训练的时候就进行量化——前向计算进行量化，而反向的误差修正不做量化。

## NN硬件的指标术语

MACC：multiply-accumulate，乘法累加。

FLOPS：Floating-point Operations Per Second，每秒所执行的浮点运算次数。

显然NN的INT8计算主要以MACC为单位。

## 低精度数据计算库 gemmlowp

gemmlowp是Google提出的一个支持低精度数据的GEMM（General Matrix Multiply）库。

代码：

https://github.com/google/gemmlowp

## 论文

《Quantizing deep convolutional networks for efficient inference: A whitepaper》

## 参考

https://www.chiphell.com/thread-1620755-1-1.html

新Titan X的INT8计算到底是什么鬼

https://mp.weixin.qq.com/s/S9VcoS_59nbZWe_P3ye2Tw

减少模型半数内存用量：百度&英伟达提出混合精度训练法

https://zhuanlan.zhihu.com/p/35700882

CNN量化技术

https://mp.weixin.qq.com/s/9DXMqiPIK5P5wzUMT7_Vfw

基于交替方向法的循环神经网络多比特量化

https://mp.weixin.qq.com/s/PDeChj1hQqUrZiepxXODJg

ICLR oral：清华提出离散化架构WAGE，神经网络训练推理合二为一

http://blog.csdn.net/tangwei2014/article/details/55077172

二值化神经网络介绍

https://mp.weixin.qq.com/s/oumf8l28ijYLxc9fge0FMQ

嵌入式深度学习之神经网络二值化（1）

https://mp.weixin.qq.com/s/tbRj5Wd69n9gvSzW4oKStg

嵌入式深度学习之神经网络二值化（2）

https://mp.weixin.qq.com/s/RsZCTqCKwpnjATUFC8da7g

嵌入式深度学习之神经网络二值化（3）

https://mp.weixin.qq.com/s/tbRj5Wd69n9gvSzW4oKStg

异或神经网络

https://mp.weixin.qq.com/s/KgM1k1bziLTCec67hQ8hlQ

超全总结：神经网络加速之量化模型


# 量化(quantization)。
    对象：对权重量化，对特征图量化(神经元输出)，对梯度量化(训练过程中)
    过程：在inference网络前传，在训练过程(反传)
    一步量化(仅对权重量化)，
    两步量化(对神经元与特征图量化，第一步先对feature map进行量化，第二步再对权重量化)
    
    32位浮点和16位浮点存储的时候，
    第一位是符号位，中间是指数位，后面是尾数。
    英特尔在NIPS2017上提出了把前面的指数项共享的方法，
    这样可以把浮点运算转化为尾数的整数定点运算，从而加速网络训练。
![](https://github.com/Ewenwan/MVision/blob/master/CNN/Deep_Compression/img/flexpoint.jpg)

    分布式训练梯度量化：
![](https://github.com/Ewenwan/MVision/blob/master/CNN/Deep_Compression/img/gradient_quant.jpg)
    

    对权重数值进行聚类，
    量化的思想非常简单。
    CNN参数中数值分布在参数空间，
    通过一定的划分方法，
    总是可以划分称为k个类别。
    然后通过储存这k个类别的中心值或者映射值从而压缩网络的储存。

    量化可以分为
    Low-Bit Quantization(低比特量化)、
    Quantization for General Training Acceleration(总体训练加速量化)和
    Gradient Quantization for Distributed Training(分布式训练梯度量化)。

    由于在量化、特别是低比特量化实现过程中，
    由于量化函数的不连续性，在计算梯度的时候会产生一定的困难。
    对此，阿里巴巴冷聪等人把低比特量化转化成ADMM可优化的目标函数，从而由ADMM来优化。

    从另一个角度思考这个问题，使用哈希把二值权重量化，再通过哈希求解.

    用聚类中心数值代替原权重数值，配合Huffman编码，
    具体可包括标量量化或乘积量化。
    但如果只考虑权重自身，容易造成量化误差很低，
    但分类误差很高的情况。
    因此，Quantized CNN优化目标是重构误差最小化。
    此外，可以利用哈希进行编码，
    即被映射到同一个哈希桶中的权重共享同一个参数值。

    聚类例子：
        例如下面这个矩阵。

        1.2  1.3  6.1
        0.9  0.7  6.9
        -1.0 -0.9 1.0
        设定类别数k=3，通过kmeans聚类。得到：
        A类中心： 1.0 , 映射下标： 1
        B类中心： 6.5 , 映射下标： 2
        C类中心： -0.95 , 映射下标： 3

        所以储存矩阵可以变换为(距离哪个中心近，就用中心的下标替换)：
        1  1  2
        1  1  2
        3  3  1
        当然，论文还提出需要对量化后的值进行重训练，挽回一点丢失的识别率 
        基本上所有压缩方法都有损，所以重训练还是比较必要的。
        
# 1. 深度神经网络压缩 Deep Compression
    为了进一步压缩网络，考虑让若干个权值共享同一个权值，
    这一需要存储的数据量也大大减少。
    在论文中，采用kmeans算法来将权值进行聚类，
    在每一个类中，所有的权值共享该类的聚类质心，
    因此最终存储的结果就是一个码书和索引表。
    
    1.对权值聚类 
        论文中采用kmeans聚类算法，
        通过优化所有类内元素到聚类中心的差距（within-cluster sum of squares ）来确定最终的聚类结果.
        
    2. 聚类中心初始化 

        常用的初始化方式包括3种： 
        a) 随机初始化。
           即从原始数据种随机产生k个观察值作为聚类中心。 

        b) 密度分布初始化。
           现将累计概率密度CDF的y值分布线性划分，
           然后根据每个划分点的y值找到与CDF曲线的交点，再找到该交点对应的x轴坐标，将其作为初始聚类中心。 

        c) 线性初始化。
            将原始数据的最小值到最大值之间的线性划分作为初始聚类中心。 

        三种初始化方式的示意图如下所示： 

![](https://img-blog.csdn.net/20161026183710142)

    由于大权值比小权值更重要（参加HanSong15年论文），
    而线性初始化方式则能更好地保留大权值中心，
    因此文中采用这一方式，
    后面的实验结果也验证了这个结论。 
    
    3. 前向反馈和后项传播 
        前向时需要将每个权值用其对应的聚类中心代替，
        后向计算每个类内的权值梯度，
        然后将其梯度和反传，
        用来更新聚类中心，
        如图： 
        
![](https://img-blog.csdn.net/20161026184233327)

        共享权值后，就可以用一个码书和对应的index来表征。
        假设原始权值用32bit浮点型表示，量化区间为256，
        即8bit，共有n个权值，量化后需要存储n个8bit索引和256个聚类中心值，
        则可以计算出压缩率compression ratio: 
            r = 32*n / (8*n + 256*32 )≈4 
            可以看出，如果采用8bit编码，则至少能达到4倍压缩率。

[通过减少精度的方法来优化网络的方法总结](https://arxiv.org/pdf/1703.09039.pdf)


 
# 降低数据数值范围。
        其实也可以算作量化
        默认情况下数据是单精度浮点数，占32位。
        有研究发现，改用半精度浮点数(16位)
        几乎不会影响性能。谷歌TPU使用8位整型来
        表示数据。极端情况是数值范围为二值
        或三值(0/1或-1/0/1)，
        这样仅用位运算即可快速完成所有计算，
        但如何对二值或三值网络进行训练是一个关键。
        通常做法是网络前馈过程为二值或三值，
        梯度更新过程为实数值。

# 2. 二值量化网络 
[二值化神经网络介绍](https://blog.csdn.net/tangwei2014/article/details/55077172)

![](http://file.elecfans.com/web1/M00/55/79/pIYBAFssV_SAdU6BAACcvDwG5pU677.png)

    上图是在定点表示里面最基本的方法：BNN和BWN。
    在网络进行计算的过程中，可以使用定点的数据进行计算，
    由于是定点计算，实际上是不可导的，
    于是提出使用straight-through方法将输出的估计值直接传给输入层做梯度估计。
    在网络训练过程中会保存两份权值，用定点的权值做网络前向后向的计算，
    整个梯度累积到浮点的权值上，整个网络就可以很好地训练，
    后面几乎所有的量化方法都会沿用这种训练的策略。
    前面包括BNN这种网络在小数据集上可以达到跟全精度网络持平的精度，
    但是在ImageNet这种大数据集上还是表现比较差。


![](https://img-blog.csdn.net/20170214003827832?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGFuZ3dlaTIwMTQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

    二值化神经网络，是指在浮点型神经网络的基础上，
    将其权重矩阵中权重值(线段上) 和 各个 激活函数值(圆圈内) 同时进行二值化得到的神经网络。
        1. 一个是存储量减少，一个权重使用 1bit 就可以，而原来的浮点数需要32bits。
        2. 运算量减少， 原先浮点数的乘法运算，可以变成 二进制位的异或运算。
        

## 2.1. BNN全二值网络

    BNN的 激活函数值 和 权重参数 都被二值化了, 前向传播是使用二值，反向传播时使用全精度梯度。 
    
[ Keras 的实现 实现了梯度的 straight-through estimator](https://github.com/Ewenwan/nn_playground/tree/master/binarynet)

[代码注解 theano 版本 采用确定性（deterministic）的二值化方式](https://github.com/Ewenwan/BinaryNet)

[torch版本 基于概率随机随机化（stochastic）的二值化, 对BN也离散化](https://github.com/Ewenwan/BinaryNet-1)

[pytorch版本](https://github.com/Ewenwan/pytorch_workplace/tree/master/binary)

[inference 前向  基于 CUDA 的 GPU](https://github.com/MatthieuCourbariaux/BinaryNet/blob/master/Run-time/binary_kernels.cu)

[inference 前向  基于 CPU 的实现（其基于 tensorflow 的训练代码有些小问题）](https://github.com/codekansas/tinier-nn/tree/master/eval)

**二值化方法**

    1. 阈值二值化，确定性(sign()函数）
       x =   +1,  x>0
             -1,  其他
![](https://img-blog.csdn.net/20170214005016493?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGFuZ3dlaTIwMTQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)    
             
    2. 概率二值化随机（基于概率）两种二值化方式。
       x = +1,  p = sigmod(x) ,  
           -1,  1-p
![](https://img-blog.csdn.net/20170214005110619?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGFuZ3dlaTIwMTQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_binarized/3.png)

    其实就是最低值0.最大值1，中间是 0.5*x+0.5的斜线段
    第二种方法虽然看起来比第一种更合理，但是在实现时却有一个问题，
    那就是每次生成随机数会非常耗时，所以一般使用第一种方法.
      
**训练二值化网络**

    提出的解决方案是：权值和梯度在训练过程中保持全精度（full precison），
    也即，训练过程中，权重依然为浮点数，
    训练完成后，再将权值二值化，以用于 inference。
    
    在训练过程中，权值为 32 位的浮点数，取值值限制在 [-1, 1] 之间，以保持网络的稳定性,
    为此，训练过程中，每次权值更新后，需要对权值 W 的大小进行检查，W=max(min(1,W),−1)。
    
    前向运算时，我们首先得到二值化的权值：Wkb=sign(Wk),k=1,⋯,n 
    然后，用 Wkb 代替 Wk：
    
    xk=σ(BN(Wkb * xk−1)=sign(BN(Wkb * xk−1))
    
    其中，BN(⋅) 为 Batch Normalization 操作。
    
**前向传播时**

**对权重值W 和 激活函数值a 进行二值化**
    
    Wk = Binary(Wb)   // 权重二值化
    Sk = ak-1 * Wb    // 计算神经元输出
    ak = BN(Sk, Ck)   // BN 方式进行激活
    ak = Binary(ak)   // 激活函数值 二值化

![](https://img-blog.csdn.net/20170214010139607?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGFuZ3dlaTIwMTQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

**在反传过程时**

**计算浮点型权重值对应的梯度和浮点型激活函数值对应的残差**

    虽然BNN的参数和各层的激活值是二值化的，但由于两个原因，
    导致梯度不得不用较高精度的实数而不是二值进行存储。
    两个原因如下：
        1. 梯度的值的量级很小
        2. 梯度具有累加效果，即梯度都带有一定的噪音，而噪音一般认为是服从正态分布的，
          所以，多次累加梯度才能把噪音平均消耗掉。
          
    另一方面，二值化相当于给权重和激活值添加了噪声，而这样的噪声具有正则化作用，可以防止模型过拟合。
    所以，二值化也可以被看做是Dropout的一种变形，
    Dropout是将激活值的一般变成0，从而造成一定的稀疏性，
    而二值化则是将另一半变成1，从而可以看做是进一步的dropout。
    
    
    使用sign函数时，
    对决定化方式中的Sign函数进行松弛化，即前传中是： 
![](https://img-blog.csdn.net/20170214005740059?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGFuZ3dlaTIwMTQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

    sign函数不可导，
    使用直通估计（straight-through estimator）(即将误差直接传递到下一层):
    反传中在已知q的梯度，对r求梯度时，Sign函数松弛为：
    gr=gq1|r|≤1
![](https://img-blog.csdn.net/20170214005816256?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGFuZ3dlaTIwMTQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

    其中1|r|<=1的计算公式就是 Htanh= max(-1, min(1,x))
    其实就是上限是1，下限是-1，中间是 y=x的斜线段

    即当r的绝对值小于等于1时，r的梯度等于q的梯度，否则r的梯度为0。 

     Htanh导数 = 1,  [-1, 1]
                 0,  其他

    直接使用决定式的二值化函数得到二值化的激活值。
    对于权重， 
    在进行参数更新时，要时时刻刻把超出[-1,1]的部分给裁剪了。即权重参数始终是[-1,1]之间的实数。
    在使用参数是，要将参数进行二值化。

    最后求得各层浮点型权重值对应的梯度和浮点型激活函数值对应的残差，
    然后用SGD方法或者其他梯度更新方法对浮点型的权重值进行更新，
    以此不断的进行迭代，直到loss不再继续下降。

    BNN中同时介绍了基于移位（而非乘法）的BatchNormailze和AdaMax算法。 
    实验结果： 
    在MNIST，SVHN和CIFAR-10小数据集上几乎达到了顶尖的水平。 
    在ImageNet在使用AlexNet架构时有较大差距（在XNOR-Net中的实验Δ=29.8%） 
    在GPU上有7倍加速.

**求各层梯度方式如下：**

![](https://img-blog.csdn.net/20170214005928789?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGFuZ3dlaTIwMTQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
    
**梯度更新方式如下：**

![](https://img-blog.csdn.net/20170214010005900?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGFuZ3dlaTIwMTQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
    
    
## 2.2. BCN 混有单精度与二值的神经网络BinaryConnect 与BNN合并**

[BinaryConnect: Training Deep Neural Networks with binary weights](https://arxiv.org/pdf/1511.00363.pdf)

[论文笔记](https://blog.csdn.net/weixin_37904412/article/details/80618102)

[BinaryConnect 代码](https://github.com/Ewenwan/BinaryConnect)

    首先点燃战火的是Matthieu Courbariaux，
    他来自深度学习巨头之一的Yoshua Bengio领导的蒙特利尔大学的研究组。
    他们的文章于2015年11月出现在arxiv.org上。
    与此前二值神经网络的实验不同，Matthieu只关心系数的二值化，
    并采取了一种混和的策略，
    构建了一个混有单精度与二值的神经网络BinaryConnect：
    当网络被用来学习时，系数是单精度的，因此不会受量化噪声影响；
    而当被使用时，系数从单精度的概率抽样变为二值，从而获得加速的好处。
    这一方法在街拍门牌号码数据集(SVHN)上石破天惊地达到超越单精度神经网络的预测准确率，
    同时超越了人类水平，打破了此前对二值网络的一般印象，并奠定了之后一系列工作的基础。
    然而由于只有系数被二值化，Matthieu的BinaryConnect只能消减乘法运算，
    在CPU和GPU上一般只有2倍的理论加速比，但在FPGA甚至ASIC这样的专用硬件上则有更大潜力。

    一石激起千层浪。Matthieu组很快发现自己的工作引起的兴趣超乎想像。
    事实上，3个月后，Itay Hubara在以色列理工的研究组甚至比Matthieu组，
    早了一天在arxiv.org上发表了同时实现系数和中间结果二值化，
    并在SVHN上达到了可观预测准确率的二值网络。
    由于双方的工作太过相似，三个星期后，也就是2016年2月29日，
    双方的论文被合并后以Matthieu与Itay并列一作的方式再次发表到arxiv.org上。
    这个同时实现系数和中间结果二值化的网络被命名为BinaryNet。
    由于达成了中间结果的二值化，BinaryNet的一个样例实现无需额外硬件，
    在现有的GPU上即达成了7倍加速。
  
## 2.3. 二值系数网络 BWN  异或网络XNOR-Net  
[BWN(Binary-Weights-Networks) ](https://arxiv.org/pdf/1603.05279.pdf)

![](http://file.elecfans.com/web1/M00/55/79/pIYBAFssV_SAaYgnAACz9cXw6vE854.png)

    每年的年初是机器学习相关会议扎堆的时段，Matthieu与Itay于3月17日更新了他们的合作论文，
    进行了一些细节的调整，看起来是投稿前的最后准备。
    但就在一天前的3月16日，来自西雅图的Allen institute for AI和
    华盛顿大学的Mohammad Rastegari等人用新方法改进了二值系数网络BinaryConnect和全二值网络BinaryNet
    ，在大规模数据集ImageNet上分别提高预测准确率十几个百分点。
    其中，改进后的 二值系数网络BWN已达到被普遍接受的神经网络质量标准：
    只差单精度AlexNet3个百分点。
    
    而Mohammad改进BinaryNet的产物XNOR-Net，离单精度的AlexNet也只差13个百分点了
    。考虑到XNOR-Net相比AlexNet的惊人的实测58倍运行时加速，
    达到二值神经网络的理论上限的光明未来已近在眼前了。 
    
    Mohammad的方法的关键是达成了计算量与量化噪声间的一个巧妙平衡：
       用二值来进行AlexNet中最昂贵的卷积操作，而用一部分单精度值计算来降低量化噪声。
    也就是说，XNOR-Net不是一个纯粹的二值网络，却保留了二值网络绝大部分的好处。
    从数学的角度，Mohammad提出了一种用二值矩阵与单精度值对角阵之积近似一个单精度值矩阵的算法。
    
    这在数学里中可归为矩阵近似的一种。
    
    矩阵近似包含一大类方法，比如笔者所在的研究组此前提出的Kronecker Fully-Connect方法，
    即用一系列小矩阵对的Kronecker积的和来近似一个大矩阵。
    类似的以减少存储大小和计算量为目的的工作还有利用随机投影的“Deep Fried Network”，
    利用循环矩阵的”Circulant Network”等等。
    由于Mohammad的二值化方法也是一种近似，因此不可避免地会造成预测准确率的降低。
    寻找能快速计算的更好的矩阵近似方法，可能是下一步的主要目标。


    上图展示了ECCV2016上一篇名为XNOR-Net的工作，
    其思想相当于在做量化的基础上，乘了一个尺度因子，这样大大降低了量化误差。
    他们提出的BWN，在ImageNet上可以达到接近全精度的一个性能，
    这也是首次在ImageNet数据集上达到这么高精度的网络。
    

    BWN(Binary-Weights-Networks) 仅有参数二值化了，激活量和梯度任然使用全精度。XNOR-Net是BinaryNet的升级版。 
    主要思想： 
        1. 二值化时增加了缩放因子，同时梯度函数也有相应改变：
        W≈W^=αB=1n∑|W|ℓ1×sign(W)
        ∂C∂W=∂C∂W^(1n+signWα)

        2. XNOR-Net在激活量二值化前增加了BN层 
        3. 第一层与最后一层不进行二值化 
    实验结果： 
        在ImageNet数据集AlexNet架构下，BWN的准确率有全精度几乎一样，XNOR-Net还有较大差距(Δ=11%) 
        减少∼32×的参数大小，在CPU上inference阶段最高有∼58× 的加速。
 
> 对于每一次CNN网络，我们使用一个三元素 《I,W,#》来表示，I 表示卷积输入，W表示滤波器，#表示卷积算子

**BWN

该网络主要是对W 进行二值化，主要是一些数学公式的推导，公式推导如下:
      
      对W进行二值化，使用 B 和缩放比例 a 来近似表达W
![](https://img-blog.csdn.net/20160715113533581)
      
      全精度权重W 和 加权二进制权重 aB 的误差函数，求解缩放比例a和二值权重B，使得误差函数值最小
![](https://img-blog.csdn.net/20160715113542440)

      误差函数展开
![](https://img-blog.csdn.net/20160715113549674)
      
      二值权重B的求解，误差最小，得到 W转置*B最大
![](https://img-blog.csdn.net/20160715113600645)

      缩放比例a的求解，由全精度权重W求解得到
![](https://img-blog.csdn.net/20160715113609159)

**BWN网络的训练


![](https://img-blog.csdn.net/20160715113831914)


**同或网络 XNOR-Networks  对 I(神经元激活输出，下一层的输入) 及 W(权重参数) 都二值化

    XNOR又叫同或门，假设输入是0或1，那么当两个输入相同时输出为1，当两个输入不同时输出为0。

[代码](https://github.com/Ewenwan/XNOR-Net)

     最开始的输入X，权重W, 使用b*H代替X, 使用a*B代替W , a,b为缩放比例，H,B为 二值矩阵。
![](https://img-blog.csdn.net/20160715114052958)
      
     网络中间隐含层的量化，二值化的矩阵相乘，在乘上一个矩阵和一个缩放因子。
![](https://img-blog.csdn.net/20160715114402250)

      主框架:
![](https://img-blog.csdn.net/20160715114256287)

    这里第四行有个符号是圆圈里面有个*，
    表示的是convolutional opration using XNOR and bitcount operation。
    也就是说正常两个矩阵之间的点乘如果用在两个二值矩阵之间，
    那么就可以将点乘换成XNOR-Bitcounting operation，
    从32位浮点数之间的操作直接变成1位的XNOR门操作，这就是加速的核心。

    由于在一般网络下，一层卷积的 kernel 规格是固定的，kernel 和 input 在进行卷积的时候，
    input 会有重叠的地方，所以在进行量化因子的运算时，先对 input 全部在 channel 维求平均，
    得到的矩阵 A，再和一个 w x h 的卷积核 k 进行卷积得到比例因子矩阵 K，

    其中：
    Kij = 1 / (w x h)
    
    实际添加了XNOR的网络:
![](https://img-blog.csdn.net/20170831082647603?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


	 
## HORQ XNOR的改进版
[Performance Guaranteed Network Acceleration viaHigh-Order Residual Quantization](https://arxiv.org/pdf/1708.08687.pdf)

	接下来详细介绍HORQ，因为HORQ可以看做是XNOR的改进版，所以建议先看看XNOR：XNOR-Net算法详解。
	HORQ和XNOR都包含对weight和input做二值化，weight二值化方面基本一样，接下来主要介绍对input的二值化。
	将CNN网络层的输入 进行高精度二值量化，从而实现高精度的二值网络计算。
![](https://img-blog.csdn.net/20170831151129332?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvemhhbmdqdW5oaXQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
	

 
## 2.4. QNN 量化网络 量化激活函数 nbit量化
[QNN Quantized Neural Networks ](https://arxiv.org/pdf/1609.07061.pdf)

        对BNN的简单扩展，
        量化激活函数，
        有线性量化与log量化两种，
        其1-bit量化即为BinaryNet。
        在正向传播过程中加入了均值为0的噪音。 
        BNN约差于XNOR-NET（<3%），
        QNN-2bit activation 略优于DoReFaNet 2-bit activation


    激活函数量 线性量化：

        LinearQuant(x, bitwidth)= Clip(round(x/bitwidth)*bitwidth,  minV, maxV )

        激活函数为 整数阶梯函数  
        最小值 minV
        最大值 maxV
        中间   线性阶梯整数

    log量化：

        LogQuant(x, bitwidth)、 = Clip (AP2(x), minV, maxV )

        AP2(x) = sign(x) × 2^round(log2|x|)
         平方近似
**QCNN**

[QCNN Quantized Convolutional Neural Networks for Mobile Devices ](https://arxiv.org/pdf/1512.06473.pdf)

[代码](https://github.com/Ewenwan/quantized-cnn)

    出了一种量化CNN的方法（Q-CNN），
    量化卷积层中的滤波器和全连接层中的加权矩阵，
    通过量化网络参数，
    用近似内积计算有效地估计卷积和全连接层的响应,
    最小化参数量化期间每层响应的估计误差，更好地保持模型性能。
    在ILSVRC-12实验有4-6倍的加速和15-20倍的压缩，只有一个百分点的分类精度损失。
    
    步骤：
        首先，全连接的层保持不变,用纠错量化所有卷积层。
        其次，利用ILSVRC-12训练集对量化网络的全连接层进行微调，恢复分类精度。
        最后，纠错量化微调的层网络的全连接。

## 2.5. 约束低比特(3比特)量化 Extremely Low Bit Neural Networks 

[论文 Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM](https://arxiv.org/pdf/1707.09870.pdf)

[翻译](https://www.jiqizhixin.com/articles/2018-01-22-6)

[解析2](https://www.jianshu.com/p/c34ec77dae9e)

[ADMM 算法理解 对原函数不好求解，转而求解它的对偶函数，基于对对偶函数的优化，从来解出原问题](https://blog.csdn.net/danyhgc/article/details/76014478)

[ADMM 算法实现](http://web.stanford.edu/~boyd/admm.html)

![](http://file.elecfans.com/web1/M00/55/79/pIYBAFssV_WAdFmiAACFxVTKLmQ760.png)

    上图展示了阿里巴巴冷聪等人做的通过ADMM算法求解binary约束的低比特量化工作。
    从凸优化的角度，在第一个优化公式中，f(w)是网络的损失函数，
    后面会加入一项W在集合C上的loss来转化为一个优化问题。
    这个集合C取值只有正负1，如果W在满足约束C的时候，它的loss就是0；
    W在不满足约束C的时候它的loss就是正无穷。
    为了方便求解还引进了一个增广变量，保证W是等于G的，
    这样的话就可以用ADMM的方法去求解。
    
    提出一种基于低比特表示技术的神经网络压缩和加速算法。
    我们将神经网络的权重表示成离散值，
    并且离散值的形式为 2 的幂次方的形式，比如 {-4，-2，-1，0，1，2，4}。
    这样原始 32 比特的浮点型权重可以被压缩成 1-3 比特的整形权重，
    同时，原始的浮点数乘法操作可以被定点数的移位操作所替代。
    在现代处理器中，定点移位操作的速度和能耗是远远优于浮点数乘法操作的。
    
    {-1，0，1 }, 三值网络，存储只需要2bit，极大地压缩存储空间，
    同时也可以避免乘法运算，只是符号位的变化和加减操作，从而提升计算速度。
![](https://upload-images.jianshu.io/upload_images/2509688-eab1154e07f49554.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/533)
    
    {-2，-1，0，1，2}, 五值网络 和 {-4，-2，-1，0，1，2，4} 七值网络，
    需要3bit存储
    
    首先，我们将离散值权重的神经网络训练定义成一个离散约束优化问题。
    以三值网络为例，其目标函数可以表示为：
![](https://image.jiqizhixin.com/uploads/editor/e02461c9-1369-4f22-ab30-c12dfec03db5/3718302.png)

    在约束条件中引入一个 scale（尺度）参数。
    对于三值网络，我们将约束条件写成 {-a, 0, a}, a>0.
    
    这样做并不会增加计算代价，
    因为在卷积或者全连接层的计算过程中可以先和三值权重 {-1, 0, 1} 进行矩阵操作，
    然后对结果进行一个标量 scale。
    从优化的角度看，增加这个 scale 参数可以大大增加约束空间的大小，
    
    这有利于算法的收敛。如下图所示：
![](https://image.jiqizhixin.com/uploads/editor/fd9ec9ac-50dd-4867-b7bb-b80bafd16c51/5179603.jpg)
    
    对于三值网络而言，scale 参数可以将约束空间从离散的 9 个点扩增到 4 条直线。

    为了求解上述约束优化问题，我们引入 ADMM 算法。
    在此之前，我们需要对目标函数的形式做一个等价变换(对偶变换)。
![](https://image.jiqizhixin.com/uploads/editor/620d2e54-5d9b-49ef-8d11-f3b29ca39794/6118804.png)

![](https://upload-images.jianshu.io/upload_images/2509688-09b68d52c24fac35.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/700)

    其中 Ic 为指示函数，
    如果 G 符合约束条件，则 Ic(G)=0，
    否则 Ic(G) 为无穷大。
    该目标函数的增广拉格朗日形式为( 将条件 引入到 目标函数):
![](https://image.jiqizhixin.com/uploads/editor/f6b5da9e-e441-489e-a53c-da61ad8f4ca9/6905205.png)
    
        ADMM 算法将上述问题分成三个子问题进行求解，即：
![](https://image.jiqizhixin.com/uploads/editor/b47c80ae-2399-48c9-a94e-0073b7a8c716/7818706.jpg)

    与其它算法不同的是，我们在实数空间和离散空间分别求解，
    然后通过拉格朗日乘子的更新将两组解联系起来。
    第一个子问题需要找到一个网络权重最小化
![](https://image.jiqizhixin.com/uploads/editor/9a21f213-5dc6-4bf5-b751-4095fa03f6c1/8625907.png)

    在实验中我们发现使用常规的梯度下降算法求解这个问题收敛速度很慢。
    在这里我们使用 Extra-gradient 算法来对这个问题进行求解。
    Extra-gradient 算法包含两个基本步骤，分别是：
![](https://image.jiqizhixin.com/uploads/editor/13c91f04-9188-462c-9b1a-f46816b5af96/9496608.png)

    第二个子问题在离散空间中进行优化。通过简单的数学变换第二个子问题可以写成：
![](https://image.jiqizhixin.com/uploads/editor/26ce6576-e2e9-4b19-965a-a8962d0e79bb/0162109.png)

    该问题可以通过迭代优化的方法进行求解。当 a 或 Q 固定时，很容易就可以获得 Q 和 a 的解析解。
    
    除上述三值网络外，还有以下几种常用的参数空间：
![](https://upload-images.jianshu.io/upload_images/2509688-d3699da636ddfe3b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/399)

    参数空间中加入2、4、8等值后，仍然不需要乘法运算，只需进行移位操作。
    因此，通过这种方法将神经网络中的乘法操作全部替换为移位和加操作。
     
## 2.6. 哈希函数两比特缩放量化 BWNH 
[论文](https://arxiv.org/pdf/1802.02733.pdf)

[博客解析](https://blog.csdn.net/ajj15120321/article/details/80571748)

![](http://file.elecfans.com/web1/M00/55/79/pIYBAFssV_WAE7dRAACHJnpcRMk945.png)

[保留内积哈希方法是沈老师团队在15年ICCV上提出的 Learning Binary Codes for Maximum Inner Product Search ](https://webpages.uncc.edu/~szhang16/paper/ICCV15_binary.pdf)

    通过Hashing方法做的网络权值二值化工作。
    第一个公式是我们最常用的哈希算法的公式，其中S表示相似性，
    后面是两个哈希函数之间的内积。
    我们在神经网络做权值量化的时候采用第二个公式，
    第一项表示输出的feature map，其中X代表输入的feature map，W表示量化前的权值，
    第二项表示量化后输出的feature map，其中B相当于量化后的权值，
    通过第二个公式就将网络的量化转化成类似第一个公式的Hashing方式。
    通过最后一行的定义，就可以用Hashing的方法来求解Binary约束。
    
    本文在二值化权重(BWN)方面做出了创新，发表在AAAI2018上，作者是自动化所程建团队。
    本文的主要贡献是提出了一个新的训练BWN的方法，
    揭示了哈希与BW(Binary Weights)之间的关联，表明训练BWN的方法在本质上可以当做一个哈希问题。
    基于这个方法，本文还提出了一种交替更新的方法来有效的学习hash codes而不是直接学习Weights。
    在小数据和大数据集上表现的比之前的方法要好。
    
    为了减轻用哈希方法所带来的loss，
    本文将binary codes乘以了一个scaling factor并用交替优化的策略来更新binary codes以及factor.
## 2.7 高阶残差量化网络 二值网络+量化残差二值网络 HORQ
论文：Performance Guaranteed Network Acceleration via High-Order Residual Quantization

[论文链接](https://arxiv.org/abs/1708.08687.pdf)


	本文是对 XNOR-Networks 的改进，将CNN网络层的输入 进行高精度二值量化，
	从而实现高精度的二值网络计算，XNOR-Networks 也是对每个CNN网络层的权值和输入进行二值化，
	这样整个CNN计算都是二值化的，这样计算速度快，占内存小。
	一般对输入做二值化后模型准确率会下降特别厉害，
	而这篇文章提出的对权重和输入做high-order residual quantization的方法,
	可以在保证准确率的情况下大大压缩和加速模型。
	
	XNOR-Networks 对输入进行一级量化
	HORQ          对输入进行多级量化(对上级量化的残差再进行量化)
	
![](https://static.leiphone.com/uploads/new/article/740_740/201710/59e2f036c850f.png?imageMogr2/format/jpg/quality/90)
	

# 3. 三值化网络 

## a. TNN 全三值网络
[Ternary Neural Networks TNN](https://arxiv.org/pdf/1609.00222.pdf)

[代码](https://github.com/Ewenwan/tnn-train)

    训练时激活量三值化，参数全精度 
    infernce时，激活量，参数都三值化（不使用任何乘法） 
    用FPGA和ASIC设计了硬件
## b. TWN 三值系数网络
    权值三值化的核心：
        首先，认为多权值相对比于二值化具有更好的网络泛化能力。
        其次，认为权值的分布接近于一个正态分布和一个均匀分布的组合。
        最后，使用一个 scale 参数去最小化三值化前的权值和三值化之后的权值的 L2 距离。 
        
[caffe-代码](https://github.com/Ewenwan/caffe-twns)


[Ternary weight networks](https://arxiv.org/pdf/1605.04711.pdf)

[论文翻译参考](https://blog.csdn.net/xjtu_noc_wei/article/details/52862282)

[参考2](https://blog.csdn.net/weixin_37904412/article/details/80590746)

    算法
![](https://img-blog.csdn.net/20161019215508291?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

    这个算法的核心是只在前向和后向过程中使用使用权值简化，但是在update是仍然是使用连续的权值。

    简单的说就是先利用公式计算出三值网络中的阈值：
![](https://img-blog.csdn.net/20161019215559323?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

    也就是说，将每一层的权值绝对值求平均值乘以0.7算出一个deta作为三值网络离散权值的阈值，
    具体的离散过程如下：
![](https://img-blog.csdn.net/20161019215719426?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

    其实就是简单的选取一个阈值（Δ），
    大于这个阈值的权值变成 1，小于-阈值的权值变成 -1，其他变成 0。
    当然这个阈值其实是根据权值的分布的先验知识算出来的。
    本文最核心的部分其实就是阈值和 scale 参数 alpha 的推导过程。

    在参数三值化之后，作者使用了一个 scale 参数去让三值化之后的参数更接近于三值化之前的参数。
    根据一个误差函数 推导出 alpha 再推导出 阈值（Δ）

    这样，我们就可以把连续的权值变成离散的（1,0，-1），

    那么，接下来我们还需要一个alpha参数，具体干什么用后面会说（增强表达能力）
    这个参数的计算方式如下：
![](https://img-blog.csdn.net/20161019215844543?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

    |I(deta)|这个参数指的是权值的绝对值大于deta的权值个数，计算出这个参数我们就可以简化前向计算了，
    具体简化过程如下：
![](https://img-blog.csdn.net/20161019215951715?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

    可以看到，在把alpha乘到前面以后，我们把复杂的乘法运算变成了简单的加法运算，从而加快了整个的训练速度。
    
    主要思想就是三值化参数（激活量与梯度精度），参照BWN使用了缩放因子。
    由于相同大小的filter，
    三值化比二值化能蕴含更多的信息，
    因此相比于BWN准确率有所提高。
   
    
    
## c. 训练三值量化 TTQ  训练浮点数量化
[Trained Ternary Quantization  TTQ](https://arxiv.org/pdf/1612.01064.pdf)

[博客参考](https://blog.csdn.net/yingpeng_zhong/article/details/80382704)

    提供一个三值网络的训练方法。
    对AlexNet在ImageNet的表现，相比32全精度的提升0.3%。
    与TWN类似，
    只用参数三值化(训练得到的浮点数)，
    但是正负缩放因子不同，
    且可训练，由此提高了准确率。
    ImageNet-18模型仅有3%的准确率下降。

    对于每一层网络，三个值是32bit浮点的{−Wnl,0,Wpl}，
    Wnl、Wpl是可训练的参数。
    另外32bit浮点的模型也是训练的对象，但是阈值Δl是不可训练的。 
    由公式(6)从32bit浮点的到量化的三值： 
![](https://img-blog.csdn.net/20180520154233906?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lpbmdwZW5nX3pob25n/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

    由(7)算出Wnl、Wpl的梯度:
![](https://img-blog.csdn.net/20180520154430336?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lpbmdwZW5nX3pob25n/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
    
    其中:
![](https://img-blog.csdn.net/20180520154553848?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lpbmdwZW5nX3pob25n/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

    由(8)算出32bit浮点模型的梯度
![](https://img-blog.csdn.net/20180520154744861?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lpbmdwZW5nX3pob25n/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

    由(9)给出阈值，这种方法在CIFAR-10的实验中使用阈值t=0.05。
    而在ImageNet的实验中，并不是由通过钦定阈值的方式进行量化的划分，
    而是钦定0值的比率r，即稀疏度。 
![](https://img-blog.csdn.net/20180520155802296?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lpbmdwZW5nX3pob25n/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
    
    整个流程:
![](https://img-blog.csdn.net/20180520154900829?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lpbmdwZW5nX3pob25n/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


## d. 三值定点变换量化 + 矩阵分解  定点分解神经网络
[Fixed-point Factorized Networks 论文](https://arxiv.org/pdf/1611.01972.pdf)

![](http://file.elecfans.com/web1/M00/55/79/pIYBAFssV_aAHHAsAACFv5V6ARc330.png)

    一般的方法都是先做矩阵（张量）分解然后做定点运算（fixed point）； 
    而这种方法是
    1、先进行不动点分解，
    2、然后进行伪全精度权重复原（pseudo full precision weight recovery），
    3、接下来做权重均衡（weight balancing），
    4、最后再进行微调（fine-tuning）。

    借助了矩阵分解和定点变换的优势，
    对原始权值矩阵直接做一个定点分解，限制分解后的权值只有+1、-1、0三个值。
    将网络变成三层的网络，首先是正常的3×3的卷积，对feature map做一个尺度的缩放，
    最后是1×1的卷积，所有的卷积的操作都有+1、-1、0。

    原矩阵 W = 三值矩阵*D矩阵*三值矩阵
![](https://upload-images.jianshu.io/upload_images/1770756-24fc80630c954833.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/354)
    
    SSD矩阵分解算法：
![](https://upload-images.jianshu.io/upload_images/1770756-8e0634a2f48d91d1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/663)
    
    全精度恢复： Full-precision Weights Recovery
![](https://upload-images.jianshu.io/upload_images/1770756-3c6160b5ec67fc1b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/353)

## 4. 二进制位量化网络 哈希函数的味道啊  仅权重量化
[ShiftCNN](http://cn.arxiv.org/pdf/1706.02393v1)

[代码 caffe-quant-shiftcnn ](https://github.com/Ewenwan/caffe-quant-shiftcnn)

[博客](https://blog.csdn.net/shuzfan/article/details/77856900)

    整个对参数进行量化的流程如下：
![](https://img-blog.csdn.net/20170905204744197?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2h1emZhbg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


    一个利用低精度和量化技术实现的神经网络压缩与加速方案。
    
>最优量化

    量化可以看作用离散码本描述样本分布。 
    优化目标(最大概率准则)和优化方法(L1和L2正则化)通常导致了神经网络参数呈现中心对称的非均匀分布。
    因此，一个最佳的量化码本应当是一个非均匀分布的码本。 
    这也是为什么BinaryNet(-1,+1)、ternary quantization(-1,0,+1)这种策略性能不足的一个重要原因。
    
    需要注意的是，
    量化之前需要对参数进行范围归一化，
    即除以最大值的绝对值，这样保证参数的绝对值都小于1。
    该量化方法具有码本小、量化简单、量化误差小的优点。
    
>量化

    ShiftCNN所采用是一种相当巧妙的类似于残差量化的方法。

    完整的码本包含 N 个子码本。 
    每个码本包含 M=2^B−1 个码字，即每一个码字可以用 B bit 表示。 
    每个码本定义如下：

     Cn=0, ±2^−n+1, ±2^−n, …, ±2^−n−⌊M/2⌋+2
    假设 N=2，B=4，则码本为

    C1=0, ±2^−1, ±2^−2, ±2^−3, ±2^−4, ±2^−5, ±2^−6
    C2=0, ±2^−2, ±2^−3, ±2^−4, ±2^−5, ±2^−6, ±2^−7
    
    于是，每一个权重都可以使用 N*B bit 的索引通过下式求和计算得到：
    wi' = sum(Cn[id(n)])

>卷积计算

    卷积计算的复杂度主要来自乘法计算。
    ShiftCNN采用简单的移位和加法来实现乘法，从而减少计算量。
    比如计算 y=wx, 而 w 通过量化已经被我们表示成了,
    类似于 2^−1 + 2^−2 + 2^−3 这种形式，
    于是 y = x>>1 + x>>2 + x>>3 

![](https://img-blog.csdn.net/20170905211321692?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2h1emZhbg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

    其中累加单元如下：(主要是逻辑控制与求和):
![](https://img-blog.csdn.net/20170905211836994?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2h1emZhbg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

    实验
    ShiftCNN量化后无需再训练。 
    个人觉得再训练应该会更好一点。
    再训练-再量化-再训练-再量化

## 5. 固定点多比特量化
[Fixed Point Quantization of Deep Convolutional Networks ](https://arxiv.org/pdf/1511.06393.pdf)

    r=S(q-Z) 其中q为定点结果，r为对应的浮点数据，S和Z分别为范围和偏移参数



## 6. Ristretto是一个和Caffe类似的框架。

    Ristretto是一个自动化的CNN近似工具，可以压缩32位浮点网络。
    Ristretto是Caffe的扩展，允许以有限的数字精度测试、训练和微调网络。
    
    本文介绍了几种网络压缩的方法，压缩特征图和参数。
    方法包括定点法（Fixed Point Approximation）、
    动态定点法（Dynamic Fixed Point Approximation）、
    迷你浮点法（Minifloat Approximation）和
    乘法变移位法（Turning Multiplications Into Bit Shifts），
    所压缩的网络包括LeNet、CIFAR-10、AlexNet和CaffeNet等。
    注：Ristretto原指一种特浓咖啡（Caffe），本文的Ristretto沿用了Caffe的框架。
    
[Ristretto: Hardware-Oriented Approximation of Convolutional Neural Networks](https://arxiv.org/pdf/1605.06402.pdf)

[caffe+Ristretto 工程代码](https://github.com/pmgysel/caffe)

[代码 主要修改](https://github.com/MichalBusta/caffe/commit/55c64c202fc8fca875e108b48c13993b7fdd0f63)

### Ristretto速览
    Ristretto Tool：
           Ristretto工具使用不同的比特宽度进行数字表示，
           执行自动网络量化和评分，以在压缩率和网络准确度之间找到一个很好的平衡点。
    Ristretto Layers：
           Ristretto重新实现Caffe层并模拟缩短字宽的算术。
    测试和训练：
           由于将Ristretto平滑集成Caffe，可以改变网络描述文件来量化不同的层。
           不同层所使用的位宽以及其他参数可以在网络的prototxt文件中设置。
           这使得我们能够直接测试和训练压缩后的网络，而不需要重新编译。

### 逼近方案
    Ristretto允许以三种不同的量化策略来逼近卷积神经网络：
        1、动态固定点：修改的定点格式。
        2、Minifloat：缩短位宽的浮点数。
        3、两个幂参数：当在硬件中实现时，具有两个幂参数的层不需要任何乘法器。


    这个改进的Caffe版本支持有限数值精度层。所讨论的层使用缩短的字宽来表示层参数和层激活（输入和输出）。
    由于Ristretto遵循Caffe的规则，已经熟悉Caffe的用户会很快理解Ristretto。
    下面解释了Ristretto的主要扩展：
### Ristretto的主要扩展
    1、Ristretto Layers
    
    Ristretto引入了新的有限数值精度层类型。
    这些层可以通过传统的Caffe网络描述文件（* .prototxt）使用。 
    下面给出一个minifloat卷积层的例子：
    
```
    layer {
      name: "conv1"
      type: "ConvolutionRistretto"
      bottom: "data"
      top: "conv1"
      convolution_param {
        num_output: 96
        kernel_size: 7
        stride: 2
        weight_filler {
          type: "xavier"
        }
      }
      quantization_param {
        precision: MINIFLOAT   # MANT:mantissa，尾数(有效数字) 
        mant_bits: 10
        exp_bits: 5
      }
    }
```
    该层将使用半精度（16位浮点）数字表示。
    卷积内核、偏差以及层激活都被修剪为这种格式。
    
    注意与传统卷积层的三个不同之处：
        1、type变成了ConvolutionRistretto；
        2、增加了一个额外的层参数：quantization_param；
        3、该层参数包含用于量化的所有信息。
        
    2、 Blobs
        Ristretto允许精确模拟资源有限的硬件加速器。
        为了与Caffe规则保持一致，Ristretto在层参数和输出中重用浮点Blob。
        这意味着有限精度数值实际上都存储在浮点数组中。
    3、Scoring
        对于量化网络的评分，Ristretto要求
          a. 训练好的32位FP网络参数
          b. 网络定义降低精度的层
        第一项是Caffe传统训练的结果。Ristretto可以使用全精度参数来测试网络。
             这些参数默认情况下使用最接近的方案，即时转换为有限精度。
        至于第二项——模型说明——您将不得不手动更改Caffe模型的网络描述，
             或使用Ristretto工具自动生成Google Protocol Buffer文件。
            # score the dynamic fixed point SqueezeNet model on the validation set*
            ./build/tools/caffe test --model=models/SqueezeNet/RistrettoDemo/quantized.prototxt \
            --weights=models/SqueezeNet/RistrettoDemo/squeezenet_finetuned.caffemodel \
            --gpu=0 --iterations=2000

#### Ristretto: SqueezeNet 示例 构造一个8位动态定点SqueezeNet网络
## 准备数据 和 预训练的 全精度模型权重
    1、下载原始 32bit FP 浮点数 网络权重
       并将它们放入models/SqueezeNet/文件夹中。这些是由DeepScale提供的预训练好的32位FP权重。
[地址](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.0)

    2、微调再训练一个低精度 网络权重 
       我们已经为您fine-tuned了一个8位动态定点SqueezeNet。
       从models/SqueezeNet/RistrettoDemo/ristrettomodel-url提供的链接下载它，并将其放入该文件夹。


    3、对SqueezeNet prototxt（models/SqueezeNet/train_val.prototxt）做两个修改

     a. imagenet 数据集下载    较大 train 100G+
        http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar
        http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar
        解压：
        tar -zxvf 

    b. 标签文件下载
       http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
       解压：
       tar -xf caffe_ilsvrc12.tar.gz
       放在 data/ilsvrc12 下面

    c. 生成 lmdb数据库文件
       ./examples/imagenet/create_imagenet.sh 里面需要修改 上面下载的数据集的路径
       得到两个数据库文件目录：
       examples/imagenet/ilsvrc12_train_leveldb
       examples/imagenet/ilsvrc12_val_leveldb

    d. 生成数据库图片的均值文件
      ./examples/imagenet/make_imagenet_mean.sh
       得到：
       data/ilsvrc12/imagenet_mean.binaryproto

    e. 修改 models/SqueezeNet/train_val.prototxt 
       网络配置文件中 训练和测试数据集数据库格式文件地址
       source: "examples/imagenet/ilsvrc12_train_lmdb"
       source: "examples/imagenet/ilsvrc12_val_lmdb"

##  量化到动态定点
    首先安装Ristretto（make all -j 见最上面的代码 ），并且在Caffe的根路径下运行所有命令。
    SqueezeNet在32位和16位动态定点上表现良好，但是我们可以进一步缩小位宽。
    参数压缩和网络准确性之间有一个折衷。
    Ristretto工具可以自动为网络的每个部分找到合适的位宽：
    
    运行量化网络： 
    ./examples/ristretto/00_quantize_squeezenet.sh
```bash
    ./build/tools/ristretto quantize \       # 工具
	--model=models/SqueezeNet/train_val.prototxt \   # 全精度网络模型
	--weights=models/SqueezeNet/squeezenet_v1.0.caffemodel \ #全精度网络权重
	--model_quantized=models/SqueezeNet/RistrettoDemo/quantized.prototxt \ # 自动生成的量化网络模型文件 
	--trimming_mode=dynamic_fixed_point \ # 量化类型 动态定点
    --gpu=0 \
    --iterations=2000 \
	--error_margin=3
```

    这个脚本将量化SqueezeNet模型。
    你会看到飞现的信息，Ristretto以不同字宽测试量化模型。
    最后的总结将如下所示：
```sh
    I0626 16:56:25.035650 14319 quantization.cpp:260] Network accuracy analysis for
    I0626 16:56:25.035667 14319 quantization.cpp:261] Convolutional (CONV) and fully
    I0626 16:56:25.035681 14319 quantization.cpp:262] connected (FC) layers.
    I0626 16:56:25.035693 14319 quantization.cpp:263] Baseline 32bit float: 0.5768
    I0626 16:56:25.035715 14319 quantization.cpp:264] Dynamic fixed point CONV
    I0626 16:56:25.035728 14319 quantization.cpp:265] weights: 
    I0626 16:56:25.035740 14319 quantization.cpp:267] 16bit: 0.557159
    I0626 16:56:25.035761 14319 quantization.cpp:267] 8bit:  0.555959
    I0626 16:56:25.035781 14319 quantization.cpp:267] 4bit:  0.00568
    I0626 16:56:25.035802 14319 quantization.cpp:270] Dynamic fixed point FC
    I0626 16:56:25.035815 14319 quantization.cpp:271] weights: 
    I0626 16:56:25.035828 14319 quantization.cpp:273] 16bit: 0.5768
    I0626 16:56:25.035848 14319 quantization.cpp:273] 8bit:  0.5768
    I0626 16:56:25.035868 14319 quantization.cpp:273] 4bit:  0.5768
    I0626 16:56:25.035888 14319 quantization.cpp:273] 2bit:  0.5768
    I0626 16:56:25.035909 14319 quantization.cpp:273] 1bit:  0.5768
    I0626 16:56:25.035938 14319 quantization.cpp:275] Dynamic fixed point layer
    I0626 16:56:25.035959 14319 quantization.cpp:276] activations:
    I0626 16:56:25.035979 14319 quantization.cpp:278] 16bit: 0.57578
    I0626 16:56:25.036012 14319 quantization.cpp:278] 8bit:  0.57058
    I0626 16:56:25.036051 14319 quantization.cpp:278] 4bit:  0.0405805
    I0626 16:56:25.036073 14319 quantization.cpp:281] Dynamic fixed point net:
    I0626 16:56:25.036087 14319 quantization.cpp:282] 8bit CONV weights,
    I0626 16:56:25.036100 14319 quantization.cpp:283] 1bit FC weights,
    I0626 16:56:25.036113 14319 quantization.cpp:284] 8bit layer activations:
    I0626 16:56:25.036126 14319 quantization.cpp:285] Accuracy: 0.5516
    I0626 16:56:25.036141 14319 quantization.cpp:286] Please fine-tune.
```
    分析表明，卷积层的激活和参数都可以降低到8位，top-1精度下降小于3％。
    由于SqueezeNet不包含全连接层，因此可以忽略该层类型的量化结果。
    最后，该工具同时量化所有考虑的网络部分。
    结果表明，8位SqueezeNet具有55.16％的top-1精度（与57.68％的基准相比）。
    为了改善这些结果，我们将在下一步中对网络进行微调。
    
## finetune 微调动态固定点参数
    上一步将 32位浮点 SqueezeNet 量化为 8位固定点，
    并生成相应的量化网络描述文件（models/SqueezeNet/RistrettoDemo/quantized.prototxt）。
    现在我们可以微调浓缩的网络，尽可能多地恢复原始的准确度。
    
    在微调期间，Ristretto会保持一组高精度的 权重。
    对于每个训练batch，这些32位浮点权重 随机 四舍五入为 8位固定点。
    然后将8位参数 用于前向 和 后向传播，最后将 权重更新 应用于 高精度权重。
    
    微调程序可以用传统的caffe工具 ./build/tools/caffe train 来完成。
    只需启动以下脚本：
```sh
./examples/ristretto/01_finetune_squeezenet.sh
//////////内容
#!/usr/bin/env sh
# finetune 微调

SOLVER="../../models/SqueezeNet/RistrettoDemo/solver_finetune.prototxt"   # 微调求解器
WEIGHTS="../../models/SqueezeNet/squeezenet_v1.0.caffemodel"              # 原始 全精度权重

./build/tools/caffe train \
    --solver=$SOLVER \
    --weights=$WEIGHTS

``` 
    经过1200次微调迭代（Tesla K-40 GPU〜5小时）， batch大小为32 * 32，
    压缩后的SqueezeNet将具有57％左右的top-1验证精度。
    微调参数位于models/SqueezeNet/RistrettoDemo/squeezenet_iter_1200.caffemodel。
    总而言之，您成功地将SqueezeNet缩减为8位动态定点，精度损失低于1％。
    请注意，通过改进数字格式（即对网络的不同部分 选择整数 和 分数长度），可以获得稍好的最终结果。
    
## SqueezeNet动态固定点基准 测试
    
    在这一步中，您将对现有的动态定点SqueezeNet进行基准测试，我们将为您进行微调。
    即使跳过上一个微调步骤，也可以进行评分。
    该模型可以用传统的caffe-tool进行基准测试。
    所有的工具需求都是一个网络描述文件以及网络参数。
    
```sh
./examples/ristretto/02_benchmark_fixedpoint_squeezenet.sh
//////////内容
./build/tools/caffe test \ # 测试模式
	--model=models/SqueezeNet/RistrettoDemo/quantized.prototxt \ # 量化网络文件
	--weights=models/SqueezeNet/RistrettoDemo/squeezenet_finetuned.caffemodel \ # 量化网络权重
	--gpu=0 \
    --iterations=2000
```

    
## 原始全精度网络测试 与上面的8位定点量化的测试结果 作对比
```sh
./examples/ristretto/benchmark_floatingpoint_squeezenet.sh
//////////内容
./build/tools/caffe test \
	--model=models/SqueezeNet/train_val.prototxt \
	--weights=models/SqueezeNet/squeezenet_v1.0.caffemodel \
	--gpu=0 --iterations=2000
```

## 7. INQ 神经网络无损低比特量化技术
    英特尔中国研究院：INQ神经网络无损低比特量化技术;
    全精度网络输入，输出权值为0或2的整数次幂的网络.
[INCREMENTAL NETWORK QUANTIZATION: TOWARDS LOSSLESS CNNS WITH LOW-PRECISION WEIGHTS](https://arxiv.org/pdf/1702.03044.pdf)

[代码](https://github.com/Ewenwan/Incremental-Network-Quantization)

## 8.其他
## 半波高斯量化的低精度深度学习 
[Deep Learning with Low Precision by Half-wave Gaussian Quantization](https://arxiv.org/pdf/1702.00953.pdf)

##  
[Network Sketching: Exploiting Binary Structure in Deep CNNs ](https://arxiv.org/pdf/1706.02021.pdf)

## 
[Training Quantized Nets: A Deeper Understanding ](https://arxiv.org/pdf/1706.02379.pdf)

## 平衡优化:一种量化神经网络的有效与高效方法 百分位数 直方图均衡化 均匀量化     
[Balanced Quantization: An Effective and Efficient Approach to Quantized Neural Networks](https://arxiv.org/pdf/1706.07145.pdf)
	
	 量化神经网络（Quantized Neural Network）
	 使用低位宽数来表示参数和执行计算，以降低神经网络的计算复杂性，存储大小和存储器使用。
	 在QNN中，参数和激活被均匀地量化为低位宽数，从而可以用高效的位运算来代替更复杂的乘加操作。
	 然而，神经网络中的参数分布通常是钟形的并且包含几个大的异常值，
	 因此从极值确定的均匀量化可能造成对位宽的不充分利用，造成错误率升高。
	 在本文中，我们提出了一种产生平衡分布的量化值的新型量化方法。
	 我们的方法先将参数按百分位数递归划分为一些平衡的箱，
	 再应用均匀量化。这样产生的量化值均匀分布在可能的值之间，从而增加了有效位宽。
	 我们还引入计算上更廉价的百分位数的近似值来减少由平衡量化引入的计算开销。
	 总体而言，我们的方法提高了QNN的预测精度，对训练速度的影响可以忽略不计，且不会在推理过程中引入额外的计算。
	 该方法适用于卷积神经网络和循环神经网络。
	
	
## Google量化方法 r=S(q-Z) 其中q为定点结果，r为对应的浮点数据，S和Z分别为范围和偏移参数 在 TFlite中应用

[Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)

## 阿里巴巴 AQN:一种通过交替量化对深度学习模型压缩以及加速推理的方法
[ALTERNATING MULTI-BIT QUANTIZATION FOR RECURRENT NEURAL NETWORKS](https://arxiv.org/pdf/1802.00150.pdf)

[参考](https://yq.aliyun.com/articles/555997)



## 
[Deep Neural Network Compression with Single and Multiple Level Quantization](https://arxiv.org/pdf/1803.03289.pdf)

## 
[Expectation Backpropagation: Parameter-Free Training of Multilayer Neural Networks with Continuous or Discrete Weights](http://papers.nips.cc/paper/5269-expectation-backpropagation-parameter-free-training-of-multilayer-neural-networks-with-continuous-or-discrete-weights.pdf)
