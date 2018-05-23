# Video Analysis 相关领域 之Action Recognition(行为识别)
[论文总结参考](https://blog.csdn.net/whfshuaisi/article/details/79116265)
# 1. 任务特点及分析
## 目的
    给一个视频片段进行分类，类别通常是各类人的动作

## 特点
    简化了问题，一般使用的数据库都先将动作分割好了，一个视频片断中包含一段明确的动作，
    时间较短（几秒钟）且有唯一确定的label。
    所以也可以看作是输入为视频，输出为动作标签的多分类问题。
    此外，动作识别数据库中的动作一般都比较明确，周围的干扰也相对较少（不那么real-world）。
    有点像图像分析中的Image Classification任务。
## 难点/关键点
    强有力的特征：
        即如何在视频中提取出能更好的描述视频判断的特征。
        特征越强，模型的效果通常较好。
    特征的编码（encode）/融合（fusion）：
        这一部分包括两个方面，
        第一个方面是非时序的，在使用多种特征的时候如何编码/融合这些特征以获得更好的效果；
        另外一个方面是时序上的，由于视频很重要的一个特性就是其时序信息，
             一些动作看单帧的图像是无法判断的，只能通过时序上的变化判断，
             所以需要将时序上的特征进行编码或者融合，获得对于视频整体的描述。
    算法速度：
        虽然在发论文刷数据库的时候算法的速度并不是第一位的。
        但高效的算法更有可能应用到实际场景中去.
# 2. 常用数据库
    行为识别的数据库比较多，这里主要介绍两个最常用的数据库，也是近年这个方向的论文必做的数据库。
    1. UCF101:来源为YouTube视频，共计101类动作，13320段视频。
       共有5个大类的动作：
                1)人-物交互；
                2)肢体运动；
                3)人-人交互；
                4)弹奏乐器；
                5)运动。
[数据库主页](http://crcv.ucf.edu/data/UCF101.php)
                
    2. HMDB51:来源为YouTube视频，共计51类动作，约7000段视频。
      HMDB: a large human motion database
[数据库主页](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

    3. 在Actioin Recognition中，实际上还有一类骨架数据库，
    比如MSR Action 3D，HDM05，SBU Kinect Interaction Dataset等。
    这些数据库已经提取了每帧视频中人的骨架信息，基于骨架信息判断运动类型。 

# 3. 研究进展
## 3.1 传统方法
### 密集轨迹算法(DT算法) iDT（improved dense trajectories)特征
     ”Action recognition with improved trajectories”
[iDT算法](https://blog.csdn.net/wzmsltw/article/details/53023363) 
[iDT算法用法与代码解析](https://blog.csdn.net/wzmsltw/article/details/53221179)
[Stacked Fisher Vector 编码 基本原理 ](https://blog.csdn.net/wzmsltw/article/details/52050112)

    基本思路：
            DT算法的基本思路为利用光流场来获得视频序列中的一些轨迹，
            再沿着轨迹提取HOF，HOG，MBH，trajectory4种特征，其中HOF基于灰度图计算，
            另外几个均基于dense optical flow计算。
            最后利用FV（Fisher Vector）方法对特征进行编码，再基于编码结果训练SVM分类器。
            而iDT改进的地方在于它利用前后两帧视频之间的光流以及SURF关键点进行匹配，
            从而消除/减弱相机运动带来的影响，改进后的光流图像被成为warp optical flow

## 3.2 深度学习方法
### 时空双流网络结构  Two Stream Network及衍生方法
#### 提出 
[“Two-Stream Convolutional Networks for Action Recognition in Videos”（2014NIPS）](https://arxiv.org/pdf/1406.2199.pdf)

    Two Stream方法最初在这篇文章中被提出，
    基本原理为:
        1. 对视频序列中每两帧计算密集光流，得到密集光流的序列（即temporal信息）。
        2. 然后对于视频图像（spatial）和密集光流（temporal）分别训练CNN模型，
           两个分支的网络分别对动作的类别进行判断，
        3. 最后直接对两个网络的class score进行fusion（包括直接平均和svm两种方法），得到最终的分类结果。
    注意，对与两个分支使用了相同的2D CNN网络结构，其网络结构见下图。
    实验效果：UCF101-88.0%，HMDB51-59.4% 

#### 改进1 CNN网络进行了spatial以及temporal的融合
[Convolutional Two-Stream Network Fusion for Video Action Recognition“（2016CVPR）](https://arxiv.org/pdf/1604.06573.pdf)

    这篇论文的主要工作为:
        1. 在two stream network的基础上，
           利用CNN网络进行了spatial以及temporal的融合，从而进一步提高了效果。
        2. 此外，该文章还将基础的spatial和temporal网络都换成了VGG-16 network。
    实验效果：UCF101-92.5%，HMDB51-65.4% 

#### TSN 结构 
[Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/pdf/1608.00859.pdf)

    这篇文章是港中文Limin Wang大神的工作，他在这方面做了很多很棒的工作，
    可以followt他的主页：http://wanglimin.github.io/ 。

    这篇文章提出的TSN网络也算是spaital+temporal fusion，结构图见下图。

    这篇文章对如何进一步提高two stream方法进行了详尽的讨论，主要包括几个方面（完整内容请看原文）： 
        1. 据的类型：除去two stream原本的RGB image和 optical flow field这两种输入外，
           章中还尝试了RGB difference及 warped optical flow field两种输入。
            终结果是 RGB+optical flow+warped optical flow的组合效果最好。
        2. 构：尝试了GoogLeNet,VGGNet-16及BN-Inception三种网络结构，其中BN-Inception的效果最好。
        3. 包括 跨模态预训练，正则化，数据增强等。
        4. 果：UCF101-94.2%，HMDB51-69.4% 
#### TSN改进版本之一  加权融合
    改进的地方主要在于fusion部分，不同的片段的应该有不同的权重，而这部分由网络学习而得，最后由SVM分类得到结果。
[Deep Local Video Feature for Action Recognition 【CVPR2017】](https://arxiv.org/pdf/1701.07368.pdf)

#### TSN改进版本二  时间推理
    这篇是MIT周博磊大神的论文，作者是也是最近提出的数据集 Moments in time 的作者之一。
    该论文关注时序关系推理。
    对于哪些仅靠关键帧（单帧RGB图像）无法辨别的动作，如摔倒，其实可以通过时序推理进行分类。
    除了两帧之间时序推理，还可以拓展到更多帧之间的时序推理。
    通过对不同长度视频帧的时序推理，最后进行融合得到结果。
    该模型建立TSN基础上，在输入的特征图上进行时序推理。
    增加三层全连接层学习不同长度视频帧的权重，及上图中的函数g和h。

    除了上述模型外，还有更多关于时空信息融合的结构。
    这部分与connection部分有重叠，所以仅在这一部分提及。
    这些模型结构相似，区别主要在于融合module的差异，细节请参阅论文。
[Temporal Relational Reasoning in Videos](https://arxiv.org/pdf/1711.08496.pdf)
    
#### LSTM 结构融合双流特征
[Beyond Short Snippets: Deep Networks for Video Classification Joe](https://arxiv.org/pdf/1503.08909.pdf)

    这篇文章主要是用LSTM来做two-stream network的temporal融合。效果一般
    实验效果：UCF101-88.6% 
    
###  3D卷积 C3D Network
#### 提出 C3D
[Learning spatiotemporal features with 3d convolutional networks](https://arxiv.org/pdf/1412.0767.pdf)

[C3D论文笔记](https://blog.csdn.net/wzmsltw/article/details/61192243)

[C3D_caffe 代码](https://github.com/facebook/C3D)

    C3D是facebook的一个工作，采用3D卷积和3D Pooling构建了网络。
    通过3D卷积，C3D可以直接处理视频（或者说是视频帧的volume）
    实验效果：UCF101-85.2% 可以看出其在UCF101上的效果距离two stream方法还有不小差距。
             我认为这主要是网络结构造成的，C3D中的网络结构为自己设计的简单结构，如下图所示。

    速度：
            C3D的最大优势在于其速度，在文章中其速度为314fps。而实际上这是基于两年前的显卡了。
    用Nvidia 1080显卡可以达到600fps以上。
    所以C3D的效率是要远远高于其他方法的，个人认为这使得C3D有着很好的应用前景。

#### 改进  I3D[Facebook]
    即基于inception-V1模型，将2D卷积扩展到3D卷积。
[Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/pdf/1705.07750.pdf)

#### T3D 时空 3d卷积
[Temporal 3D ConvNets:New Architecture and Transfer Learning for Video Classificati](https://arxiv.org/pdf/1711.08200.pdf)

    该论文值得注意的，
        一方面是采用了3D densenet，区别于之前的inception和Resnet结构；
        另一方面，TTL层，即使用不同尺度的卷积（inception思想）来捕捉讯息。
#### P3D  [MSRA]
[Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/10/iccv_p3d_camera.pdf)

    改进ResNet内部连接中的卷积形式。然后，超深网络，一般人显然只能空有想法，望而却步。

### 其他方法 
#### PP3D  Temporal Pyramid Pooling
[End-to-end Video-level Representation Learning for Action Recognition](https://arxiv.org/pdf/1711.04161.pdf)

    Pooling。时空上都进行这种pooling操作，旨在捕捉不同长度的讯息。
    
#### TLE  时序线性编码层

    1. 本文主要提出了“Temporal Linear Encoding Layer” 时序线性编码层，主要对视频中不同位置的特征进行融合编码。
       至于特征提取则可以使用各种方法，文中实验了two stream以及C3D两种网络来提取特征。

    2. 实验效果：UCF101-95.6%，HMDB51-71.1% （特征用two stream提取）。
       应该是目前为止看到效果最好的方法了（CVPR2017里可能会有更好的效果） 
[Deep Temporal Linear Encoding Networks](https://arxiv.org/pdf/1611.06678.pdf)



