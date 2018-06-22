---
title: 目标检测算法总结

date: 2017/7/30 12:04:12

categories:
- 深度学习
tags:
- 目标检测
- 深度学习
- 神经网络
---


 去年总结了一篇关于目标检测的博客 [视频智能之——目标检测](https://dragonfive.github.io/object_detection/)，今年到现在有了新的体会，所以就更新一篇。
 
 ![目标检测题图][1]
<!--more-->

# 目标检测

![map与速度][2]

## 检测算法划分

目标检测的算法大致可以如下划分：
- 传统方法：
1. 基于Boosting框架：Haar/LBP/积分HOG/ICF/ACF等特征+Boosting等
2. 基于SVM：HOG+SVM or DPM等

- CNN方法：
1. 基于region proposal：以Faster R-CNN为代表的R-CNN家族
2. 基于回归区域划分的：YOLO/SSD
3. 给予强化学习的：attentionNet等。 
4. 大杂烩：Mask R-CNN

## selective search 策略

selective search的策略是，因为目标的层级关系，用到了**multiscale**的思想，那我们就尽可能遍历所有的尺度好了，但是不同于暴力穷举，可以先得到小尺度的区域，然后一次次**合并**得到大的尺寸就好了。既然特征很多，那就把我们知道的特征都用上，但是同时也要照顾下**计算复杂度**，不然和穷举法也没啥区别了。最后还要做的是能够**对每个区域进行排序**，这样你想要多少个候选我就产生多少个。

- 使用**Efficient GraphBased** Image Segmentation中的方法来得到region
- 得到所有region之间两两的相似度
- 合并最像的两个region
- 重新计算新合并region与其他region的相似度
- 重复上述过程直到整张图片都聚合成一个大的region
- 使用一种**随机的计分方式**给每个region打分，按照分数进行ranking，取出**top k**的子集，就是selective search的结果

### 区域划分与合并

首先通过**基于图的图像分割方法?**初始化原始区域，就是将图像分割成很多很多的小块。然后我们使用**贪心策略**，计算每两个相邻的区域的相似度，然后每次合并最相似的两块，直到最终只剩下一块完整的图片。然后这其中每次产生的图像块包括合并的图像块我们都保存下来，这样就得到图像的**分层表示**了呢。

优先合并小的区域

### 颜色空间多样性

作者采用了8中不同的颜色方式，主要是为了考虑场景以及光照条件等。这个策略主要应用于【1】中图像分割算法中原始区域的生成。主要使用的颜色空间有：**（1）RGB，（2）灰度I，（3）Lab，（4）rgI（归一化的rg通道加上灰度），（5）HSV，（6）rgb（归一化的RGB），（7）C，（8）H（HSV的H通道）**

使用L1-norm归一化获取图像**每个颜色通道的25 bins**的直方图，这样每个区域都可以得到一个**75维**的向量。![enter description here][3]，区域之间颜色相似度通过下面的公式计算：

![enter description here][4]
 在区域合并过程中使用需要对新的区域进行计算其直方图，计算方法：
 
![enter description here][5]
优先合并小的区域.





### 距离计算多样性

 这里的纹理采用SIFT-Like特征。具体做法是对每个颜色通道的8个不同方向计算方差σ=1的高斯微分（GaussianDerivative），每个通道每个方向获取10 bins的直方图（L1-norm归一化），这样就可以获取到一个240维的向量。![enter description here][6]


还有大小相似度(优先合并小的区域)和吻合相似度(boundingbox是否在一块儿)

### 给区域打分

这篇文章做法是，给予最先合并的**图片块**较大的权重，比如最后一块完整图像权重为1，倒数第二次合并的区域权重为2以此类推。但是当我们策略很多，多样性很多的时候呢，这个权重就会有太多的重合了，排序不好搞啊。文章做法是给他们乘以一个随机数，毕竟3分看运气嘛，然后对于相同的区域多次出现的也叠加下权重，毕竟多个方法都说你是目标，也是有理由的嘛。

区域的分数是区域内图片块权重之和。
### reference

[目标检测（1）-Selective Search](https://zhuanlan.zhihu.com/p/27467369)

[论文笔记 《Selective Search for Object Recognition》](http://blog.csdn.net/csyhhb/article/details/50425114)


## bounding-box regression详解 
(回归/微调的对象是什么

![enter description here][7]

![enter description here][8]


![enter description here][9]


![enter description here][10]

![enter description here][11]



### reference

[bounding-box regression详解](http://caffecn.cn/?/question/160 )



## RCNN 

RCNN作为第一篇目标检测领域的深度学习文章。这篇文章的创新点有以下几点：将CNN用作目标检测的特征提取器、**有监督预训练的方式初始化**CNN、在CNN特征上做**BoundingBox 回归**。

目标检测区别于目标识别很重要的一点是其需要目标的具体位置，也就是BoundingBox。而产生BoundingBox最简单的方法就是滑窗，可以在卷积特征上滑窗。但是我们知道**CNN是一个层次的结构，随着网络层数的加深，卷积特征的步伐及感受野也越来越大**。

对于每一个region proposal 都wrap到固定的大小的scale,227x227(AlexNet Input),对于每一个处理之后的图片，把他都放到CNN上去进行特征提取，得到每个region proposal的feature map,这些特征用固定长度的特征集合feature vector来表示。



本文的亮点在于网络结构和训练集
### 训练集

经典的目标检测算法在区域中提取人工设定的特征（Haar，HOG）。本文则需要训练深度网络进行特征提取。可供使用的有两个数据库： 
一个较大的识别库（ImageNet ILSVC 2012）：标定每张图片中物体的类别。**一千万图像，1000类**。 
一个较小的检测库（PASCAL VOC 2007）：标定每张图片中，物体的类别和位置。**一万图像，20类**。 



### 整体结构 

![RCNN][12]


![网络结构][13]

RCNN的输入为完整图片，首先通过区域建议算法产生一系列的候选目标区域，其中使用的区域建议算法为**Selective Search,选择2K**个置信度最高的区域候选。

然后对这些候选区域预处理成227 × 227 pixel size ，16 pixels of warped image context around the original box 

然后对于这些目标区域候选提取其**CNN特征AlexNet**，并训练**SVM分类**这些特征。最后为了提高定位的准确性在SVM分类后区域基础上进行**BoundingBox回归**。

VGG这个模型的特点是选择比较小的卷积核、选择较小的跨步，这个网络的精度高，不过**计算量是Alexnet的7倍**。Alexnet特征提取部分包含了5个卷积层、2个全连接层，在Alexnet中**p5层神经元个数为9216、 f6、f7的神经元个数都是4096**，通过这个网络训练完毕后，最后提取特征每个输入候选框图片都能得到一个4096维的特征向量。





#### CNN目标特征提取 finetune


**物体检测的一个难点在于，物体标签训练数据少**，如果要直接采用**随机初始化CNN参数**的方法，那么目前的训练数据量是远远不够的。这种情况下，最好的是采用某些方法，把参数初始化了，然后在进行有监督的参数微调，这边文献采用的是有监督的预训练。所以paper在设计网络结构的时候，是**直接用Alexnet**的网络，然后连参数也是直接采用它的参数，作为初始的参数值，然后再fine-tuning训练。
网络优化求解：采用**随机梯度下降法**，学习速率大小为0.001；最后一层全连接层采用参数随机初始化的方法，其它网络层的参数不变。

RCNN使用ImageNet的有标签数据进行有监督的**预训练Alexnet**。直到现在，这种方法已经成为CNN初始化的标准化方法。但是训练CNN的样本量还是不能少的，为了尽可能获取最多的正样本，RCNN将**IOU>0.5（IoU 指重叠程度，计算公式为：A∩B/A∪B）的样本都称为正样本**。每次ieration 的batch_size为128，其中正样本个数为32，负样本为96.其实这种设置是偏向于正样本的，因为正样本的数量实在是太少了。由于CNN需要固定大小的输入，因此对于每一个区域候选，首先将其防缩至227\*227，然后通过CNN提取特征。

特定领域的fine-tuning 为了适应不同场合的识别需要，如VOC，对网络继续使用从VOC图片集上对**region proposals归一化后的图片**进行训练。网络只需要将最后的1000类的分类层换成**21类的分类层（20个VOC中的类别和1个背景类）**，其他都不需要变。为了保证训练只是对网络的微调而不是大幅度的变化，网络的**学习率只设置成了0.001**。计算每个region proposal与人工标注的框的IoU，IoU重叠阈值设为0.5，大于这个阈值的作为正样本，其他的作为负样本。



#### svm训练 
数据是在经过微调的RCNN上取得Fc7层特征，然后训练SVM，并通过BoundingBox回归得到的最终结果。
RCNN的SVM训练将**ground truth样本作为正样本**，而**IOU < 0.3的样本作为负样本**，这样也是SVM困难样本挖掘的方法。

**分类器**
对每一类目标，使用一个线性SVM二类分类器进行判别。输入为深度网络输出的4096维特征，输出是否属于此类。 
由于负样本很多，使用hard negative mining方法。 
**正样本**
本类的真值标定框。 
**负样本**
考察每一个候选框，如果和本类所有标定框的重叠都**小于0.3**，认定其为负样本


一旦CNN f7层特征被提取出来，那么我们将为每个物体累训练一个svm分类器。当我们用CNN提取2000个候选框，可以得到2000x4096这样的特征向量矩阵，然后我们只需要把这样的一个矩阵与svm权值矩阵4096xN点（Therefore，the pool5 need to be set as）乘(N为分类类别数目，因为我们训练的N个svm，每个svm包好了4096个W)，就可以得到结果了


#### 贪婪非极大值抑制

由于有多达2K个区域候选，我们如何筛选得到最后的区域呢？RCNN使用**贪婪非极大值抑制**的方法，假设ABCDEF五个区域候选，首先根据概率从大到小排列。假设为FABCDE。然后从最大的F开始，计算F与ABCDE是否IoU是否超过某个阈值，如果超过则将ABC舍弃。然后再从D开始，直到集合为空。而这个阈值是筛选得到的，通过这种处理之后一般只会剩下几个区域候选了。

![NMS非极大值抑制][14]

定位一个车辆，最后算法就找出了一堆的方框，我们需要判别哪些矩形框是没用的。非极大值抑制：先假设有6个矩形框，根据分类器类别分类概率做排序，从小到大分别属于车辆的概率分别为A、B、C、D、E、F。
(1)从最大概率矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个设定的阈值;
(2)假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记第一个矩形框F，是我们保留下来的。
(3)从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，重叠度大于一定的阈值，那么就扔掉；并标记E是我们保留下来的第二个矩形框。
就这样一直重复，找到所有被保留下来的矩形框。



#### BoundingBox回归

为了进一步提高定位的准确率，RCNN在贪婪非极大值抑制后进行BoundingBox回归，进一步微调BoundingBox的位置。不同于DPM的BoundingBox回归，RCNN是在Pool5层进行的回归。而**BoundingBox是类别相关的，也就是不同类的BoundingBox回归的参数是不同的**。例如我们的区域候选给出的区域位置为：也就是区域的中心点坐标以及宽度和高度。

目标检测问题的衡量标准是重叠面积：许多看似准确的检测结果，往往因为候选框不够准确，重叠面积很小。故需要一个位置精修步骤。 回归器对每一类目标，使用一个线性脊回归器进行精修。正则项λ=10000.
输入为深度网络**pool5层**的4096维特征，输出为xy方向的**缩放和平移**。 训练样本判定为本类的候选框中，和真值**重叠面积大于0.6的候选框**。


### 瓶颈

- 速度瓶颈：重复为每个region proposal提取特征是极其费时的，Selective Search对于每幅图片产生2K左右个region proposal，也就是意味着一幅图片需要经过2K次的完整的CNN计算得到最终的结果。
- 性能瓶颈：对于所有的region proposal**放缩**到固定的尺寸会导致我们不期望看到的**几何形变**，而且由于速度瓶颈的存在，不可能采用多尺度或者是大量的数据增强去训练模型。

作者提到花费在region propasals和提取特征的时间是13s/张-GPU和53s/张-CPU。

r-cnn有点麻烦，他要先过一次classification得到分类的model，继而在得到的model上进行适当的改变又得到了detection的model，最后才开始在detection model cnn上进行边界检测。

### reference

[目标检测（2）-RCNN](https://zhuanlan.zhihu.com/p/27473413)


[RCNN学习笔记(0):rcnn简介](http://blog.csdn.net/u011534057/article/details/51240387)


[ RCNN学习笔记(1):](http://blog.csdn.net/u011534057/article/details/51218218)


[ RCNN学习笔记(2)](http://blog.csdn.net/u011534057/article/details/51218250)

## SPPNet

论文：Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition


![sppnet与rcnn的区别][15]


SPP的优点：

1）任意尺寸输入，固定大小输出
2）层多
3）可对任意尺度提取的特征进行池化。

图可以看出SPPnet和RCNN的区别，首先是输入不需要放缩到指定大小。其次是增加了一个空间金字塔池化层，使得fc层能够固定参数个数。

空间金字塔池化层是SPPNet的核心

![enter description here][16]
其主要目的是对于任意尺寸的输入产生固定大小的输出。思路是对于任意大小的feature map首先分成16、4、1个块，然后在每个块上最大池化，池化后的特征拼接得到一个固定维度的输出。以满足全连接层的需要。不过因为不是针对于目标检测的，所以输入的图像为一整副图像。

为了简便，在fintune的阶段只修改fc层。

multi-size训练，输入尺寸在[180,224]之间，假设最后一个卷积层的输出大小为a×a，若给定金字塔层有n×n 个bins，进行滑动窗池化，窗口尺寸为win=⌈a/n⌉，步长为str=⌊a/n⌋，使用一个网络完成一个完整epoch的训练，之后切换到另外一个网络。只是在训练的时候用到多尺寸，测试时直接将SPPNet应用于任意尺寸的图像。


### reference

[目标检测（3）-SPPNet](https://zhuanlan.zhihu.com/p/27485018)



## fast RCNN

无论是RCNN还是SPPNet，其**训练都是多阶段**的。首先通过ImageNet**预训练网络模型**，然后通过检测数据集**微调模型提取每个区域候选的特征**，之后通过**softmax分类**每个区域候选的种类，最后通过**区域回归**，精细化每个区域的具体位置。为了避免多阶段训练，同时在单阶段训练中提升识别准确率，Fast RCNN提出了**多任务目标函数**，将softmax分类以及区域回归的部分纳入了卷积神经网络中。

所有的层在finetune阶段都是可以更新的，使用了truncated SVD方法，MAP是66%.

### 网络结构

![fast RCNN网络结构][17]

整体框架大致如上述所示,再次几句话总结：

1.用selective search在一张图片中生成约2000个object proposal，即RoI。
2.把它们整体输入到全卷积的网络中，在**最后一个卷积层上对每个ROI求映射关系**，并用一个RoI pooling layer来统一到相同的大小－> (fc)feature vector 即－>提取一个固定维度的特征表示。
3.继续经过两个全连接层（FC）得到特征向量。特征向量经由各自的FC层，得到两个输出向量：第一个是分类，使用softmax，第二个是每一类的bounding box回归。

**ROI pooling**
对比SPPNet，首先是将SPP换成了**ROI Poling**。ROI Poling可以看作是空间金字塔池化的简化版本，它通过将区域候选对应的卷积层特征还分为H\*W个块，然后在每个块上进行最大池化就好了。每个块的划分也简单粗暴，直接使用卷积特征尺寸除块的数目就可以了。空间金字塔池化的特征是多尺寸的，而ROI Pooling是**单一尺度**的。而对于H\*W的设定也是参照网络Pooling层的，例如对于VGG-19，网络全连接层输入是7\*7\*512，因此对应于我们的H,W就分别设置为7，7就可以了。另外一点不同在于网络的输出端，无论是SPPNet还是RCNN，CNN网络都是仅用于特征提取，因此输出端只有网络类别的概率。而Fast RCNN的网络输出是**包含区域回归**的。

Rol pooling layer的作用主要有两个：
1.是将image中的rol定位到feature map中对应patch
2.是用一个单层的SPP layer将这个feature map patch下采样为大小固定的feature再传入全连接层。即RoI pooling layer来统一到相同的大小－> (fc)feature vector 即－>提取一个固定维度的特征表示。


- muti-task 
将网络的**输出改为两个子网络**，一个用以分类（softmax）一个用于回归。最后更改网络的输入，网络的**输入是图片集合以及ROI的集合**

![统一的损失函数][18]

kx是真实类别，式中第一项是分类损失，第二项是定位损失，L由R个输出取均值而来。

1.对于分类loss，是一个N+1路的softmax输出，其中的N是类别个数，1是背景。为何不用SVM做分类器了？在5.4作者讨论了softmax效果比SVM好，因为它引入了类间竞争。（笔者觉得这个理由略牵强，估计还是实验效果验证了softmax的performance好吧 ）
2.对于回归loss，是一个4xN路输出的regressor，也就是说对于每个类别都会训练一个单独的regressor的意思，比较有意思的是，这里regressor的loss不是L2的，而是一个平滑的L1，形式如下：

![enter description here][19]

![smooth图像][20]

- Mini-Batch 采样
Mini-Batch的设置基本上与SPPNet是一致的，不同的在于128副图片中，仅来自于**两幅图片**。其中25%的样本为正样本，也就是IOU大于0.5的，其他样本为负样本，同样使用了**困难负样本挖掘**的方法，也就是负样本的IOU区间为[0.1，0.5），负样本的u=0，[u> 1]函数为艾弗森指示函数，意思是如果是背景的话我们就不进行区域回归了。在训练的时候，每个区域候选都有一个正确的标签以及正确的位置作为监督信息。

- ROI Pooling的反向传播
不同于SPPNet，我们的ROI Pooling是可以反向传播的，让我们考虑下正常的Pooling层是如何反向传播的，以**Max Pooling为例，根据链式法则，对于最大位置的神经元偏导数为1，对于其他神经元偏导数为0**。ROI Pooling 不用于常规Pooling，因为很多的区域建议的感受野可能是相同的或者是重叠的，因此在一个Batch_Size内，我们需要对于这些重叠的神经元偏导数进行求和，然后反向传播回去就好啦。

RoI pooling层计算损失函数对每个输入变量x的偏导数，如下

![ROI pooling反向传播][21]




### 改算法的结构与思考的问题

**四个阶段**
1.**Rol pooling** layer(fc) 
2.**Multi-task** loss(one-stage) 
3.Scale invariance(trade off->single scale(compare with multi-scale for decreasing **1mAP**) )
4.**SVD** on fc layers(speed up training)

**作者提出的问题**
1.Which layers to finetune?
2.Data augment
3.Are more proposals always better

### R-CNN、SPP-net的缺点以及fast-RCnn层的改进

**R-CNN**
1. 训练分多个阶段
2. 每个proposal都要计算convNet特征，并保存在硬盘上。


**SPPnet**
1. 训练分多个阶段
2. finetune的时候只微调FC层，一方面这样简单，另一方面spp层不方便做反向传播

**fast-RCNN**
1. 把多个任务的损失函数写到一起，实现单级的训练过程； 
2. 小批量取样，以image为中心取样
2. 在训练时可更新所有的层； 
3. 不需要在磁盘中存储特征。 

RoI-centric sampling：从所有图片的所有RoI中均匀取样，这样每个SGD的mini-batch中包含了不同图像中的样本。（SPPnet采用） 
FRCN想要解决微调的限制,就要反向传播到spp层之前的层->(reason)反向传播需要计算每一个RoI感受野的卷积层，通常会覆盖整个图像，如果一个一个用RoI-centric sampling的话就又慢又耗内存。
Fast RCNN:->改进了SPP-Net在实现上无法同时tuning在SPP layer两边的卷积层和全连接层

**image-centric sampling**：(solution)mini-batch采用层次取样，先对图像取样，再对RoI取样，同一图像的RoI共享计算和内存。 另外，FRCN在一次微调中联合优化softmax分类器和bbox回归。


### reference

[目标检测（4）-Fast R-CNN](https://zhuanlan.zhihu.com/p/27582096)

[ RCNN学习笔记(4)：fast rcnn](http://blog.csdn.net/u011534057/article/details/51241831)



## Faster RCNN
faster RCNN可以简单地看做“区域生成网络+fast RCNN“的系统,着重解决系统的三个问题：

1. 如何设计区域生成网络 
2. 如何训练区域生成网络 
3. 如何让区域生成网络和fast RCNN网络共享特征提取网络


**每张图只提300个proposals**

Fast RCNN提到如果去除区域建议算法的话，网络能够接近实时，而 **selective search方法进行区域建议的时间一般在秒级**。产生差异的原因在于卷积神经网络部分运行在GPU上，而selective search运行在CPU上，所以效率自然是不可同日而语。一种可以想到的解决策略是将selective search通过GPU实现一遍，但是这种实现方式忽略了接下来的**检测网络可以与区域建议方法共享计算**的问题。因此Faster RCNN从提高区域建议的速度出发提出了region proposal network 用以通过GPU实现快速的区域建议。通过**共享卷积，RPN在测试时的速度约为10ms**，相比于selective search的秒级简直可以忽略不计。Faster RCNN整体结构为RPN网络产生区域建议，然后直接传递给Fast RCNN。

### faster rcnn 结构



对于一幅图片的处理流程为：图片-卷积特征提取-RPN产生proposals-Fast RCNN分类proposals。

![enter description here][22]


![网络结构][23]

### feature extraction 特征提取

原始特征提取（上图灰色方框）包含若干层conv+relu，直接套用ImageNet上常见的分类网络，额外添加一个conv+relu层，输出51*39*256维特征（feature）。


### region proposal network

[RCNN,Fast RCNN,Faster RCNN 总结](http://shartoo.github.io/RCNN-series/)


![faster RCNN的结构][24]

区域建议算法一般分为两类：基于超像素合并的（selective search、CPMC、MCG等），基于滑窗算法的。由于卷积特征层一般很小，所以得到的滑窗数目也少很多。但是产生的滑窗准确度也就差了很多，毕竟感受野也相应大了很多。

![区域建议算法][25]

RPN对于feature map的每个位置进行**滑窗**，通过**不同尺度以及不同比例的K个anchor**产生K个256维的向量，然后分类每一个region是否包含目标以及通过**回归**得到目标的具体位置。

供RPN网络输入的特征图经过RPN网络得到区域建议和区域得分，并对**区域得分采用非极大值抑制【阈值为0.7】，输出其Top-N【文中为300】得分的区域建议给RoI**池化层；

**单个RPN网络结构**

![RPN网络结构][26]

上图中卷积层/全连接层表示卷积层或者全连接层，作者在论文中表示这两层实际上是全连接层，但是网络在所有滑窗位置**共享全连接层**，可以很自然地用n×n卷积核【论文中设计为3×3】跟随两个并行的1×1卷积核实现

RPN的作用：RPN在CNN卷积层后增加滑动窗口操作以及两个卷积层完成区域建议功能，第一个卷积层将特征图**每个滑窗位置编码成一个特征向量**，第二个卷积层对应每个滑窗位置输出k个区域得分和k个回归后的区域建议，并对得分区域进行**非极大值抑制**后输出得分**Top-N【文中为300】区域**，告诉检测网络应该注意哪些区域

**anchor**

Anchors是一组大小固定的参考窗口：三种尺度{ 1282，2562，51221282，2562，5122 }×三种长宽比{1:1，1:2，2:1}，如下图所示，表示RPN网络中对特征图滑窗时每个滑窗位置所对应的原图区域中9种可能的大小。 继而**根据图像大小计算滑窗中心点对应原图区域的中心点**，通过中心点和size就可以得到滑窗位置和原图位置的映射关系，由此原图位置并**根据与Ground Truth重复率贴上正负标签**，让RPN学习该Anchors是否有物体即可。因为ground truth 是在原图上的所以要做映射。

**平移不变性**

Anchors这种方法具有平移不变性，就是说在图像中平移了物体，窗口建议也会跟着平移。

**RPN网络的训练**

$$L(\{p_i\},\{t_i\})=\frac{1}{N_{cls}}\sum _i L_{cls}(p_i,p_i^*)+\lambda \frac{1}{N_{reg}}\sum _i p_i ^* L_{reg}(t_i,t_i^*)$$


Lcls是  分类的损失（classification loss），是一个二值分类器（是object或者不是）的softmax loss。其公式为 $$Lcls(p_i,p_i^*)=−log[p_i*p^*_i+(1−p^⋆_i)(1−pi)]$$

LregLreg 回归损失（regression loss）:


RPN网络被ImageNet网络【ZF或VGG-16】进行了有监督预训练，利用其训练好的网络参数初始化； 用标准差0.01均值为0的高斯分布对新增的层随机初始化

PASCAL VOC 数据集中既有物体类别标签，也有物体位置标签； 正样本仅表示前景，负样本仅表示背景； 回归操作仅针对正样本进行； 训练时弃用所有超出图像边界的anchors，否则在训练过程中会产生较大难以处理的修正误差项，导致训练过程无法收敛； 对去掉超出边界后的anchors集采用非极大值抑制，**最终一张图有2000个anchors用于训练**。文中提到对于**1000×600**的一张图像，大约有20000(~60×40×9)个anchors，忽略**超出边界**的anchors剩下6000个anchors，利用**非极大值抑制**去掉重叠区域，剩2000个区域建议用于训练； **测试时在2000个区域建议中选择Top-N【文中为300】个区域建议**用于Fast R-CNN检测。

![解释][27]


### 训练方法 

4-Step Alternating Training交替训练的方法，思路和迭代的Alternating training有点类似，但是细节有点差别：

第一步：用ImageNet模型初始化，独立训练一个RPN网络；
第二步：仍然用ImageNet模型初始化，但是使用上一步RPN网络产生的proposal作为输入，训练一个Fast-RCNN网络，至此，两个网络每一层的参数完全不共享；
第三步：使用第二步的Fast-RCNN网络参数初始化一个新的RPN网络，但是把RPN、Fast-RCNN共享的那些卷积层的learning rate设置为0，也就是不更新，仅仅更新RPN特有的那些网络层，重新训练，此时，两个网络已经共享了所有公共的卷积层；
第四步：仍然固定共享的那些网络层，把Fast-RCNN特有的网络层也加入进来，形成一个unified network，继续训练，fine tune Fast-RCNN特有的网络层，此时，该网络已经实现我们设想的目标，即网络内部预测proposal并实现检测的功能。

### RPN的boundingbox和fast-rcnn的回归的区别

 Fast R-CNN中基于RoI的bounding-box回归所输入的特征是在特征图上对任意size的**RoIs进行Pool操作提取的**，所有size RoI共享回归参数，而在RPN中，用来bounding-box回归所输入的特征是在特征图上相同的空间size【3×3】上提取的，为了解决不同尺度变化的问题，同时训练和学习了**k个不同的回归器**，依次对应为上述9种anchors，这k个回归量**并不分享权重**。


### Faster R-CNN中三种尺度怎么解释：

原始尺度：原始输入的大小，不受任何限制，不影响性能；

归一化尺度：输入特征提取网络的大小，在测试时设置，源码中opts.test_scale=600。anchor在这个尺度上设定，这个参数和anchor的相对大小决定了想要检测的目标范围；

网络输入尺度：输入特征检测网络的大小，在训练时设置，源码中为224×224。


### reference

[目标检测（5）-Faster RCNN](https://zhuanlan.zhihu.com/p/27988828)

[ RCNN学习笔记(5) faster rcnn](http://blog.csdn.net/u011534057/article/details/51247371)

[简单粗暴地使用自己数据集训练Faster-RCNN模型](http://blog.leanote.com/post/braveapple/%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8%E8%87%AA%E5%B7%B1%E6%95%B0%E6%8D%AE%E9%9B%86%E8%AE%AD%E7%BB%83Faster-RCNN%E6%A8%A1%E5%9E%8B)

[Faster-RCNN+ZF用自己的数据集训练模型(Python版本)](http://blog.csdn.net/sinat_30071459/article/details/51332084)

[目标检测--Faster RCNN2](https://saicoco.github.io/object-detection-4/)

[如何在faster—rcnn上训练自己的数据集（单类和多类](https://www.zhihu.com/question/39487159)




## YoLo
YOLO的核心思想就是利用整张图作为网络的输入，直接在输出层回归bounding box的位置和bounding box所属的类别。该方法采用单个神经网络直接预测物品边界和类别概率，实现端到端的物品检测。同时，该方法检测速非常快，基础版可以达到**45帧/s的实时检测；FastYOLO可以达到155帧/s**。由于可以看到图片的全局信息，所以YOLO的背景预测的假阳性优于当前最好的方法。

Resize成$448*448$，图片分割得到$7*7$网格(cell)
CNN提取特征和预测：卷积不忿负责提特征。全链接部分负责预测：
a) $7*7*2=98$个bounding box(bbox) 的坐标$x_{center},y_{center},w,h$ 和是否有物体的conﬁdence 。 
b) $7*7=49个cell$所属20个物体的概率。
过滤bbox（通过nms）


### yolo的实现方法

网络结构类似于 **GoogleNet**

预训练分类网络： 在 ImageNet 1000-class competition dataset上预训练一个分类网络，这个网络是Figure3中的前20个卷机网络+average-pooling layer+ fully connected layer （此时网络输入是224*224）。

训练检测网络：转换模型去执行检测任务，《Object detection networks on convolutional feature maps》提到说在预训练网络中增加卷积和全链接层可以改善性能。在他们例子基础上添加4个卷积层和2个全链接层，随机初始化权重。检测要求细粒度的视觉信息，所以把网络输入也又$224*224$变成$448*448$。


将一幅图像分成SxS个网格(grid cell)，如果某个object的中心 落在这个网格中，则这个网格就负责预测这个object。

- 每个网格要预测B个bounding box，每个bounding box除了要回归自身的位置之外，还要附带预测一个confidence值。 这个confidence代表了所预测的box中含有object的置信度和这个box预测的有多准两重信息，其值是这样计算的：
$$confidence = Pr(Object) \ast IOU^{truth}_{pred}$$


其中如果有object落在一个grid cell里，第一项取1，否则取0。 第二项是预测的bounding box和实际的groundtruth之间的IoU值。

- 每个bounding box要预测(x, y, w, h)和confidence共5个值，每个网格还要预测一个类别信息，记为C类。则SxS个网格，每个网格要预测B个bounding box还要预测C个categories。输出就是S x S x (5xB+C)的一个tensor。 
注意：class信息是针对每个网格的，confidence信息是针对每个bounding box的。

- PASCAL VOC中，图像输入为**448x448**，**取S=7，B=2，一共有20个类别(C=20)**。则输出就是7x7x30的一个tensor。 

![yolo的网络结构][28]

- 在test的时候，每个网格预测的class信息和bounding box预测的confidence信息相乘，就得到每个bounding box的class-specific confidence score: 
等式左边第一项就是每个网格预测的类别信息，第二三项就是每个bounding box预测的confidence。这个乘积即encode了预测的box属于某一类的概率，也有该box准确度的信息。

- 得到每个box的class-specific confidence score以后，设置阈值，滤掉得分低的boxes，对保留的boxes进行**NMS处理**，就得到最终的检测结果


### 实现细节 

![inference][29]

每个grid有30维，这30维中，8维是回归box的坐标，2维是box的confidence，还有20维是类别。 其中坐标的x,y用对应网格的offset归一化到0-1之间，w,h用图像的width和height**归一化**到0-1之间。

对不同大小的box预测中，相比于大box预测偏一点，小box预测偏一点肯定更不能被忍受的。而sum-square error loss中对同样的偏移loss是一样。 
为了缓和这个问题，作者用了一个比较取巧的办法，就是将box的width和height取平方根代替原本的height和width。这个参考下面的图很容易理解，小box的横轴值较小，发生偏移时，反应到y轴上相比大box要大。

作者向预训练模型中加入了4个卷积层和两层全连接层，提高了模型输入分辨率（224×224->448×448）。顶层预测类别概率和bounding box协调值。bounding box的宽和高通过输入图像宽和高归一化到0-1区间。顶层采用linear activation，其它层使用 leaky rectified linear。作者采用sum-squared error为目标函数来优化，增加bounding box loss权重，减少置信度权重。


![enter description here][30]

更重视8维的坐标预测，给这些损失前面赋予更大的loss weight, 记为 $\lambda_{coord}$ 在pascal VOC训练中取5。
对没有object的box的confidence loss，赋予小的loss weight，记为 $\lambda_noobj$ 在pascal VOC训练中取0.5。
有object的box的confidence loss和类别的loss的loss weight正常取1。

![损失函数][31]



这个损失函数中： 
只有当某个网格中有object的时候才对classification error进行惩罚。
只有当某个box predictor对某个ground truth box负责的时候，才会对**box的coordinate error**进行惩罚，而对哪个ground truth box负责就看其预测值和ground truth box的IoU是不是在那个cell的所有box中最大。
其他细节，例如使用激活函数使用leak RELU，模型用ImageNet预训练等等

### yolo的缺点

YOLO对相互靠的很近的物体，还有**很小的群体 检测效果**不好，这是因为**一个网格中只预测了两个框，并且只属于一类**。

对测试图像中，同一类物体出现的新的不常见的长宽比和其他情况是。泛化能力偏弱。

由于损失函数的问题，定位误差是影响检测效果的主要原因。尤其是大小物体的处理上，还有待加强。



### reference
[YOLO：实时快速目标检测](https://zhuanlan.zhihu.com/p/25045711)

[RCNN学习笔记(6)：You Only Look Once(YOLO)](http://blog.csdn.net/u011534057/article/details/51244354)

[YOLO有史以来讲的最清楚的PPT](https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.g137784ab86_4_1252)


## SSD

采用 **VGG16** 的基础网络结构，使用前面的前 5 层，然后利用 astrous 算法将 fc6 和 fc7 层转化成两个卷积层。再格外增加了 3 个卷积层，和一个 average pool层。不同层次的 feature map 分别用于 default box 的偏移以及不同类别得分的预测，最后通过 nms得到最终的检测结果。

该论文是在 ImageNet 分类和定位问题上的已经训练好的 **VGG16** 模型中 fine-tuning 得到，使用 SGD，初始学习率为 **10^{-3}**, 冲量为 0.9，权重衰减为 **0.0005**，batchsize 为 32。不同数据集的学习率改变策略不同。新增加的卷积网络采用 **xavier** 的方式进行初始化，输入图片大小是300x300



![网络大致结构][32]

这些增加的卷积层的 feature map 的大小变化比较大，允许能够检测出**不同尺度**下的物体： 在低层的feature map,感受野比较小，高层的感受野比较大，在不同的feature map进行卷积，可以达到多尺度的目的。

### 六尺度检测器 


![多尺度feature map][33]

多尺度feature map得到 default boxs及其 4个位置偏移和21个类别置信度。

对于不同尺度feature map（ 上图中 38x38x512，19x19x512, 10x10x512, 5x5x512, 3x3x512, 1x1x256） 的上的所有特征点： 以5x5x256为例 它的#defalut_boxes = 6。

![检测器生成5x5x6个结果][34]

1. 按照不同的 scale 和 ratio 生成，k 个 default boxes，这种结构有点类似于 Faster R-CNN 中的 Anchor。(此处k=6所以：5x5x6 = 150 boxes)
scale: 假定使用 m 个不同层的feature map 来做预测，最底层的 feature map 的 scale 值为 $s_{min} = 0.2$，最高层的为 $s_{max} = 0.95$，其他层通过下面公式计算得到 $s_k = s_{min} + \frac{s_{max} - s_{min}}{m - 1}(k-1), k \in [1,m]$
ratio: 使用不同的 ratio值$a_r \in \left\{1, 2, \frac{1}{2}, 3, \frac{1}{3} \right \}$ 计算 default box 的宽度和高度：$w_k^{a} = s_k\sqrt{a_r}，h_k^{a} = s_k/\sqrt{a_r}$。另外对于 ratio = 1 的情况，额外再指定 scale 为$s_k{'} = \sqrt{s_ks_{k+1}}$ 也就是总共有 6 中不同的 default box。
default box中心：上每个 default box的中心位置设置成  $( \frac{i+0.5}{  \left| f_k \right| },\frac{j+0.5}{\left| f_k \right| }  )$ ，其中 $\left| f_k \right|$ 表示第k个特征图的大小 $i,j \in [0, \left| f_k \right| )$ 。因为有**6个ratio**，所以每个位置有6个defaul box.
2. 新增加的每个卷积层的 feature map 都会通过一些小的卷积核操作，得到每一个 default boxes 关于物体类别的21个置信度 ($c_1,c_2 ,\cdots, c_p$ 20个类别和1个背景) 和4偏移 (shape offsets) 。
假设feature map 通道数为 p 卷积核大小统一为 $3*3*p$ （此处p=256）。个人猜想作者为了使得卷积后的feature map与输入尺度保持一致必然有 padding = 1， stride = 1 ： $$\frac{ inputFieldSize - kernelSize + 2 \cdot padding }{stride}  + 1 = \frac{5 - 3 + 2 \cdot 1}{1} + 1 = 5 $$
假如feature map 的size 为$m*n$, 通道数为 p，使用的卷积核大小为 $3*3*p$。每个 feature map 上的每个特征点对应 k 个 default boxes，物体的类别数为 c，那么一个feature map就需要使用 $k(c+4)$个这样的卷积滤波器，最后有 $(m*n) *k* (c+4)$个输出。

### 训练策略 

**正负样本**

==正负样本==： 给定输入图像以及每个物体的 ground truth,首先找到每个ground true box对应的default box中**IOU最大的做为**（与该ground true box相关的匹配）正样本。然后，在剩下的default box中找到那些与任意一个ground truth box 的 **IOU 大于 0.5的default box**作为（与该ground true box相关的匹配）正样本。下图的例子是：给定输入图像及 ground truth，分别在两种不同尺度(feature map 的大小为 8*8，4*4)下的匹配情况。有两个 default box 与猫匹配（$8*8$），一个 default box 与狗匹配（$4*4$）。

![feature map][35]


![anchor在faster-rcnn与ssd的区别][36]


目标函数，和常见的 Object Detection 的方法目标函数相同，分为两部分：计算相应的 default box 与目标类别的 score(置信度)以及相应的回归结果（位置回归）。置信度是采用 Softmax Loss（Faster R-CNN是log loss），位置回归则是采用 Smooth L1 loss （与Faster R-CNN一样采用 offset_PTDF靠近 offset_GTDF的策略

![损失函数][37]





### reference


[RCNN学习笔记(10)：SSD:Single Shot MultiBox Detector](http://blog.csdn.net/u011534057/article/details/52733686)

[晓雷机器学习笔记-SSD](https://zhuanlan.zhihu.com/p/24954433)

[最好的SSD课程](https://deepsystems.ai/en/reviews)


# 其它reference


[CVPR2017-目标检测相关论文](https://zhuanlan.zhihu.com/p/28088956)


[深度学习实践经验：用Faster R-CNN训练行人检测数据集Caltech——准备工作](http://jacobkong.github.io/posts/2093106769/)


[深度学习实践经验：用Faster R-CNN训练行人检测数据集Caltech](http://jacobkong.github.io/posts/464905881/)


  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501725558357.jpg
  [2]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1503912071330.jpg
  [3]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501731400655.jpg
  [4]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501731416622.jpg
  [5]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501731430205.jpg
  [6]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501731713492.jpg
  [7]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1503827108571.jpg
  [8]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1503827140439.jpg
  [9]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1503827159776.jpg
  [10]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1503827186738.jpg
  [11]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1503827198922.jpg
  [12]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501733503711.jpg
  [13]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1503646999361.jpg
  [14]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1503820950848.jpg
  [15]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501816659717.jpg
  [16]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501819431880.jpg
  [17]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501820819379.jpg
  [18]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1503827726497.jpg
  [19]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1503827873218.jpg
  [20]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1503889360944.jpg
  [21]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1503827982062.jpg
  [22]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1503843755653.jpg
  [23]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1508381821403.jpg
  [24]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501831614917.jpg
  [25]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501832022067.jpg
  [26]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1508382052921.jpg
  [27]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1508384487090.jpg
  [28]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1503905518100.jpg
  [29]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1503978258508.jpg
  [30]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1503905912862.jpg
  [31]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1503905928952.jpg
  [32]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1503975757154.jpg
  [33]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1503976032747.jpg
  [34]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1503976224714.jpg
  [35]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1503976848017.jpg
  [36]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1503976974477.jpg
  [37]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1503977129813.jpg
