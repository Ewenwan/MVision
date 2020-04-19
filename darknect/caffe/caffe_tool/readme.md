# caffe 工具
[参考](https://github.com/Ewenwan/caffe_tools)

## 计算模型参数等

## 吸收BN层
[模型优化：BatchNorm合并到卷积中](https://blog.csdn.net/wfei101/article/details/78635557)


      bn层即batch-norm层，一般是深度学习中用于加速训练速度和一种方法，
      一般放置在卷积层（conv层）或者全连接层之后，
      将数据归一化并加速了训练拟合速度。
      但是 bn 层虽然在深度学习模型训练时起到了一定的积极作用，
      但是在预测时因为凭空多了一些层，影响了整体的计算速度并占用了更多内存或者显存空间。
      所以我们设想如果能将ｂｎ层合并到相邻的卷积层或者全连接层之后就好了.
      
      
      源网络 prototxt去除 BN和 scale
      
      每一层的 BN和scale参数 被用来修改 每一层的权重W 和 偏置b
      

## temsorflow 模型转 caffe
[temsorflow 模型](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

      dump_tensorflow_weights.py ： 
      模型优化:BatchNorm合并到卷积中， dump the weights of conv layer and batchnorm layer.
         
      load_caffe_weights.py ：
      load the dumped weights to deploy.caffemodel.
   
##  caffe  coco模型 转 voc模型
      coco2voc.py


## 模型修改
```py
import numpy as np
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

# //修改后的prototxt
src_prototxt = "xxx.prototxt"

# //原始的prototxt
old_prototxt = "s.prototxt"
old_caffemodel = "s.caffemodel"

# 创建网络模型对象
caffe.set_mode_cpu()
net = caffe.Net(src_prototxt, caffe.TEST)
net_old = caffe.Net(old_prototxt, old_caffemodel, caffe.TEST)

src_net_params = caffe_pb2.NetParameter()
text_format.Merge(open(src_prototxt).read(), src_net_params)

#拷贝相同名字层的参数
for k,v in net_old.params.items():
    # print (k,v[0].data.shape)
    # print (np.size(net_old.params[k]))
    if(k in net.layer_dict.keys()):
        print(k, v[0].data.shape)
        print(np.size(net_old.params[k]))
        for i in range(np.size(net_old.params[k])):
           net.params[k][i].data[:] = np.copy(net_old.params[k][i].data[:])
net.save("eur_single.caffemodel")
```


## 模型计算量
[参考](https://github.com/Captain1986/CaptainBlackboard/blob/master/D%230023-CNN%E6%A8%A1%E5%9E%8B%E8%AE%A1%E7%AE%97%E9%87%8F%E4%BC%B0%E8%AE%A1/D%230023.md)

在我们训练的深度学习模型在资源受限的嵌入式设备上落地时，**精度不是我们唯一的考量因素**，我们还需要考虑

1. **安装包的大小**，如果你的模型文件打包进app一起让客户下载安装，那么动辄数百MB的模型会伤害用户的积极性；

2. 模型速度，或者说**计算量的大小**。现在手机设备上的图片和视频的分辨率越来越大，数据量越来越多；对于视频或者游戏，FPS也越来越高，这都要求我们的模型在计算时，速度越快越好，计算量越小越好；

3. 运行时**内存占用大小**，内存一直都是嵌入式设备上的珍贵资源，占用内存小的模型对硬件的要求低，可以部署在更广泛的设备上，降低我们**算法落地的成本**；况且，一些手机操作系统也不会分配过多的内存给单一一个app，当app占用内存过多，系统会kill掉它；

4. **耗电量大小**，智能手机发展到今天，最大的痛点一直是电池续航能力和发热量，如果模型计算量小，内存耗用小的话，自然会降低电量的消耗速度。

### 计算量评价指标

一个朴素的评估模型速度的想法是评估它的计算量。一般我们用FLOPS，即每秒浮点操作次数FLoating point OPerations per Second这个指标来衡量GPU的运算能力。这里我们用MACC，即乘加数Multiply-ACCumulate operation，或者叫MADD，来衡量模型的计算量。

不过这里要说明一下，用MACC来估算模型的计算量只能**大致地**估算一下模型的速度。模型最终的的速度，不仅仅是和计算量多少有
关系，还和诸如**内存带宽**、优化程度、CPU流水线、Cache之类的因素也有很大关系。

为什么要用乘加数来评估计算量呢？因为CNN中很多的计算都是类似于y = w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + ... + w[n-1]*x[n-1]这样的点乘然后累加的形式，其中w和x是向量，结果y是标量。

在CNN中最常见的卷积层和全连接层中，w是学习到的权重值，而x是该层的输入特征图，y是该层的输出特征图。一般来说，每层输出不止一张特征图，所以我们上面的乘加计算也要做多次。这里我们约定w[0]*x[0] + ...算一次乘加运算。这样来算，像上面两个长度为n的向量w和x相乘，就有n次乘法操作和n-1次加法操作，大约可等于n次乘加操作。


### CNN常用层计算量分析

#### 全连接层

全连接层执行的计算就是y = matmul(x, W) + b，这里x是I个输入值的向量，W是包含层权重的IxJ矩阵，b是包含J个元素的偏置值向量。结果y包含由层计算的输出值，也是大小为J的向量。

为了计算MACC的数量，我们看点乘发生的位置matmul(x, W)。矩阵乘法matmul只包含一大堆的点积运算。每个点积都在输入x和矩阵W的一列间发生。两者都有个I元素，因此这算作I个MACC。我们必须计算J个这样的点积，因此MACC的总数IxJ与权重矩阵的大小相同。

加偏置b并不会太影响MACC的数量，毕竟加偏置的操作次数远少于矩阵乘法里面的乘加次数。

总之，一个长度为I的向量与一个I x J维度的矩阵相乘（这就是全连接呀）得到一个长度为J的输出向量，需要I x J次MACC或者(2xI - 1) x J和FLOPS。

如果全连接层直接跟随卷积层，则其输入大小可能不会被指定为单个矢量长度I，但是可能被指定为具有诸如形状(512, 7, 7)的特征图。例如Keras要求你先将这个输入“压扁flatten”成一个向量，这样就可以得到I = 512×7×7个输入。

### 激活函数

通常深度学习模型层的后面会串联一个非线性激活函数，例如ReLU或者Sigmoid函数。这些激活函数自然也会消耗时间。但是我们不用MACC来计算它们的计算量，而是使用FLOPS，因为它们不完全是乘加运算。

有些激活函数的计算比其他激活函数更难，例如，ReLU：y = max(x, 0)，这只是GPU上的一次单次操作。对于一个有J个输出神经元的全连接层来说，ReLU只做J次这样的运算，所以算J次FLOPS。对于Sigmoid函数y = 1 / (1 + exp(-x))来说，因为它涉及到指数运算和倒数，所以它有更多的计算量。当我们计算FLOPS时，我们通常把加、减、乘、除、取幂、求根等看做一次FLOPS。因为Sigmoid函数有四种（减、取幂、加、除），所以它每个输出对应四个FLOPS，对于J个输出单元的全连接层后的Sigmoid激活层，有J x 4次FLPOS。

通常我们不计算激活函数的计算量，因为他们只占整个网络计算量中的很小一部分，我们主要关心大矩阵乘法和点乘运算，直接认为激活函数的运算是免费的。

总结：不需要担忧激活函数。

### 卷积层

卷积层的输入和输出不是矢量，而是三维特征图H × W × C，其中H是特征图的高度，W宽度和C是通道数。

今天使用的大多数卷积层都是方形核。对于具有核大小K的卷积层，MACC的数量为：K × K × Cin × Hout × Wout × Cout。这个公式可以这么理解：

      首先，输出特征图中有Hout × Wout × Cout个像素；

      其次，每个像素对应一个立体卷积核K x K x Cin在输入特征图上做立体卷积卷积出来的；

      最后，而这个立体卷积操作，卷积核上每个点都对应一次MACC操作

      同样，我们在这里为了方便忽略了偏置和激活。
我们不应该忽略的是层的stride，以及任何dilation因子，padding等。这就是为什么我们需要参看层的输出特征图的尺寸Hout × Wout，因它考虑到了stride等因素。

### 深度可分离卷积层

这里对于MobileNet V1中的深度可分离卷积只列个结论，更详细的讨论可见本黑板报我前面写的depthwise separable convolutions in mobilenet一文。MobileNet V1深度可分层的总MACC是：MACC_v1 = (K × K × Cin × Hout × Wout) + (Cin × Hout × Wout × Cout)，其中K是卷积核大小，Cin是输入特征图通道数，Hout, Wout是DW卷积核输出尺寸（PW卷积只改变输出通道数，不改变输入输出尺寸）。深度可分离卷积的计算量和传统卷积计算量的比为(K × K + Cout) / K × K × Cout，约等于 1 / (K x K)。

下面我们详细讨论下MobileNet V2中的MACC。

MobileNet V2相比与V1，主要是由DW+PW两层变成了下面的三层PW+DW+PW：

一个1×1卷积，为特征图添加更多通道（称为expansion layer）

3×3深度卷积，用于过滤数据（depthwise convolution）

1×1卷积，再次减少通道数（projection layer，bottleneck convolution）

这种扩展块中MACC数量的公式：

Cexp = (Cin × expansion_factor)，（expansion_factor用于创建深度层要处理的额外通道，使得Cexp在此块内使用的通道数量）

MACC_expansion_layer = Cin × Hin × Win × Cexp，(参照上面传统卷积，把卷积核设置为1x1即得)

MACC_depthwise_layer = K × K × Cexp × Hout × Wout(参照MoblieNet V1分析)

MACC_projection_layer = Cexp × Hout × Wout × Cout(参照MoblieNet V1分析，或者传统卷积把卷积核设置为1x1即得)

把所有这些放在一起：

MACC_v2 = Cin × Hin × Win × Cexp + (K × K + Cout) × Cexp × Hout × Wout

如果stride = 1，则简化为：

(K × K + Cout + Cin) × Cexp × Hout × Wout


## 模型 内存访问估计  mem acc cost   内存带宽(bandwidth) 
我们对常见层的计算量(MACC，FLOPS)做了分析和估算，但这只是模型性能估计这整个故事的一部分。内存带宽(bandwidth)是另一部分，大部分情况下，它比计算次数更重要！

### 内存访问
在当前的计算机架构中，内存的访问比CPU中执行单个计算要慢得多（需要更多的时钟周期）—— 大约100或更多倍！

对于网络中的每个层，CPU需要：

      1. 首先，从主存储器读取输入向量或特征图；

      2. 然后，计算点积——这也涉及从主存中读取层的权重；

      3. 最后，将计算出的结果作为新的矢量或特征图写回主存储器。

这涉及大量的内存访问。由于内存非常慢（相对于CPU计算速度而言），因此该层执行的内存读/写操作量也会对其速度产生很大影响——可能比计算次数更大。


#### 卷积层和全连接层：读取权重带来的内存访问

网络每层学习的参数或权重存储在主存储器中。通常，模型的权重越少，运行的速度就越快。

> **将权重读入**

**全连接层** 将其权重保持在大小I × J矩阵中，其中I是输入神经元的数量和J是输出的数量。它还有一个大小J的偏置量。所以这一层的权重总共有  **(I + 1) × J**。

**大多数卷积层**都有正方形内核，因此对于具有内核大小K和Cin输入通道的卷积层，每个滤波器都有权重K × K × Cin。该层将具有Cout滤波器/输出通道，因此权重总数 **K × K × Cin × Cout**加上额外的Cout个偏置值。

通常，**卷积层的权重数量小于全连接层。**

很明显，全连接层是内存权重访问的负担！

有用的结论：由于权值共享，卷积层一般占网络更少的权重参数数量，但是更多的计算量。

我们可以使用全连接层实现卷积层，反之亦然。卷积可以看成是一个全连接层，绝大多数连接设置为0——每个输出仅连接到K × K输入
而不是所有输出，并且所有输出对这些连接使用相同的值。这就是卷积层对内存更有效的原因，因为它们不存储未使用的连接的权重。

#### 卷积层：读取特征图、权重参数和写回中间结果带来的内存访问

在文献中，经常会看到模型的复杂性，其中列出了MACC（或FLOPS）的数量和训练参数的数量。但是，这忽略了一个重要的指标：层的输入读取的内存量，以及写入该层输出执行的内存访问次数。

假设卷积层的输入形状是Hin x Win x Cin图像，输出特征图形状Hout x Wout x Cout那么，对于每个输出特征图的像素来说，需要访问输入特征图次数为每个卷积核的参数的个数：K x K x Cin。所以，此卷积层需要访问内存（读取输入特征）的次数为(K × K × Cin) x (Hout x Wout x Cout)。（当然，一个聪明的GPU内核程序员将有办法优化这一点。每个GPU线程可以计算多个输出像素而不是一个，允许它多次重复使用一些输入值，总体上需要更少的内存读取，所有这些优化都将平等地应用于所有模型。因此，即使我的公式不是100％正确，它们的误差是常数级的，因此仍然可用于比较模型。）

对于计算得到的特征图的输出，如果此特定卷积层的步幅为2，滤波器为32个，则它会写入具有112×112×32个值的输出特征图。那么需要112 x 112 x 32 = 401,408次内存访问。

对于本层卷积的参数从内存中读取，因为参数数量很少，可以直接认为只读取一次，存储在缓存中。这里读取次数为K x K x Cin x Cout + Cout。

总结：每个层将进行以下总内存访问：

      1. input = (K × K × Cin) x (Hout x Wout x Cout)  
            一次访问输入数据大小(单个卷积核参数量) * 总共多少次(输出像素数量)
      2. output = Hout × Wout × Cout
            计算一次，输出赋值一次
      3. weights = K × K × Cin × Cout + Cout
            读取一次在缓存，Cout 个 维度为 K × K × Cin 的卷积核
      
具体举例来说，如果是一副输入224 x 224 x 3的图片，经过stride = 2，K = 3的卷积，输出112 x 112 x 32的特征图，那么有：

      input = 3 × 3 × 3 × 112 × 112 × 32 = 10,838,016(96.42%)
      output = 112 × 112 × 32 = 401,408(3.57%)
      weights = 3 × 3 × 3 × 32 + 32 = 896(0.01%)
      total = 11,240,320

有这个例子我们可以看到，卷积层主要的内存访问发生在把输入特征图反复搬运到CPU参与计算（因此有得会重排参数和输入来达到更好的缓存访问效果），把计算得到的输出特征图写入内存和权重的读取带来的内存访问，可以忽略不计。顺便说一句，我们这里假设了权重只被读取一次并缓存在本地CPU/GPU内存中，因此它们可以在CPU/GPU线程之间共享，并将重新用于每个输出像素。

对于网络中较深的层，具有28 x 28 x 256个输入和28 x 28 x 512个输出，K = 3，stride = 1，那么：

      input = 3 × 3 × 256 × 28 × 28 × 512 = 924,844,032(99.83%)
      output = 28 × 28 × 512 = 401,408(0.04%)
      weights = 3 × 3 × 256 × 512 + 512 = 1,180,160(0.13%)
      total = 926,425,600
      
即使特征图的宽度和高度现在较小，它们也会有更多的通道。这就是为什么权重的计算更多，因为由于通道数量的增加，权重会越来越多。但是主要的内存访问依然是把输入特征图反复搬运到CPU参与计算。

#### 深度可分离卷积分析

如果使用深度可分离卷积呢？使用跟前面相同的输入和输出大小，计算3×3深度卷积层和1×1逐点层的内存访问次数：

      DepthWise layer
      input = 3 × 3 × 1 x 28 × 28 × 256 = 1,806,336
      output = 28 × 28 × 256 = 200,704
      weights = 3 × 3 × 1 x 256 + 256 = 2,560
      total = 2,009,600(1.91%)
      PointWise layer
      input = 1 × 1 × 256 × 28 × 28 × 512 = 102,760,448
      output = 28 × 28 × 512 = 401,408
      weights = 1 × 1 × 256 × 512 + 512 = 131,584
      total = 103,293,440(98.09%)
      total of both layers = 105,303,040
      
可以看到深度可分离卷积它的内存访问量减少到大约原来的926425600 / 105303040 = 8.80倍（几乎是K × K倍）,这就是使用深度可分层的好处。还可以看到Depth-Wise层的内存访问成本非常便宜，几乎可以忽略不计。

#### 激活层和BN层：融合

在PyTorch和大多数训练框架中，经常会看到Conv2D层后面跟着一个应用ReLU的激活层。这对训练框架来说很好，提供了灵活性，但是让ReLU成为一个单独的层是浪费的，特别是因为这个函数非常简单。

示例：对28 × 28 × 512卷积层的输出应用ReLU ：

      input = 28 × 28 × 512 = 401,408
      output = 28 × 28 × 512 = 401,408
      weights = 0
      total = 802,816

首先，它需要从卷积层读取特征图每个像素，然后对其应用ReLU，最后将结果写回内存。当然，这非常快，因为它几乎与将数据从一个内存位置复制到另一个内存位置相同，但这样的操作有些浪费。

因此，激活函数通常与卷积层融合。这意味着卷积层在计算出点积之后直接应用ReLU，然后才能写出最终结果。这节省了一次读取和一次写入存储器的昂贵时钟开销。

同理，对于BN层来说，将BN层融合进卷积层也是一种在实践中经常用到的策略。
