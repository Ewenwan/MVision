# Video Analysis 相关领域 之Action Recognition(行为识别)
[论文总结参考](https://blog.csdn.net/whfshuaisi/article/details/79116265)

[博客参考2](https://blog.csdn.net/wzmsltw/article/details/70239000)

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
[【数据集整理】人体行为识别和图像识别](https://blog.csdn.net/liuxiao214/article/details/78889662)

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
    如今人体行为识别是计算机视觉研究的一个热点，
    人体行为识别的目标是从一个未知的视频或者是图像序列中自动分析其中正在进行的行为。
    简单的行为识别即动作分类，给定一段视频，只需将其正确分类到已知的几个动作类别，
    复杂点的识别是视频中不仅仅只包含一个动作类别，而是有多个，
    系统需自动的识别出动作的类别以及动作的起始时刻。
    行为识别的最终目标是分析视频中
    哪些人       who
    在什么时刻    when
    什么地方，    where
    干什么事情，  what
    即所谓的“W4系统”
    
    人体行为识别应用背景很广泛，主要集中在智能视频监控，
    病人监护系统，人机交互，虚拟现实，智能家居，智能安防，
    运动员辅助训练，另外基于内容的视频检索和智能图像压缩等
    有着广阔的应用前景和潜在的经济价值和社会价值，
    其中也用到了不少行为识别的方法。
## 3.1 传统方法
### 特征综述

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
    总结：        
    A. 利用光流场来获得视频序列中的一些轨迹;
        a.	通过网格划分的方式在图片的多个尺度上分别密集采样特征点,滤除一些变换少的点；
        b.	计算特征点邻域内的光流中值来得到特征点的运动速度，进而跟踪关键点；
    B. 沿轨迹提取HOF,HOG,MBH,trajectory,4种特征
        其中HOG基于灰度图计算，
        另外几个均基于稠密光流场计算。
        a.	HOG, 方向梯度直方图，分块后根据像素的梯度方向统计像素的梯度幅值。
        b.	HOF, 光流直方图，光流通过当前帧梯度矩阵和相邻帧时间上的灰度变换矩阵计算得到，
            之后再对光流方向进行加权统计。
        c.	MBH，运动边界直方图，实质为光流梯度直方图。
        d.	Trajectories, 轨迹特征，特征点在各个帧上位置点的差值，构成轨迹变化特征。
    C．特征编码—Bag of Features;
        a. 对训练集数据提取上述特征，使用K_means聚类算法，对特征进行聚类，得到特征字典；
        b.  使用字典单词对测试数据进行量化编码，得到固定长度大小的向量，可使用VQ或则SOMP算法。
    D. 使用SVM进行分类
        对编码量化之后的特征向量使用SVM支持向量机进行分类。
#### iDT（improved dense trajectories) 改进
    1. 剔除相机运动引起的背景光流
        a. 使用SURF特征算法匹配前后两帧的 匹配点对，这里会使用人体检测，剔除人体区域的匹配点，运动量大，影响较大；
        b. 利用RANSAC 随机采样序列一致性算法估计 前后两帧的 单应投影变换矩阵 I(t+1) = H * I(t)；
        c. 计算相机运动引起的光流Mt [u',v'] = I(t+1)' - I(t)
        d. 使用光流算法计算 t帧的全局光流 Ft
           假设1：光照亮度恒定：
                  I(x, y, t) =  I(x+dx, y+dy, t+dt) 
                 泰勒展开：
                  I(x+dx, y+dy, t+dt) =  
                                        I(x, y, t) + dI/dx * dx + dI/dy * dy + dI/dt * dt
                                      =  I(x, y, t) + Ix * dx  + Iy * dy + It * dt
                 得到：
                      Ix * dx  + Iy * dy + It * dt = 0
                 因为 像素水平方向的运动速度 u=dx/dt,  像素垂直方向的运动速度 v=dy/dt
                 等式两边同时除以 dt ,得到：
                      Ix * dx/dt  + Iy * dy/dt + It = 0
                      Ix * u  + Iy * v + It = 0
                 写成矩阵形式：
                      [Ix, Iy] * [u; v] = -It,  式中Ix, Iy为图像空间像素差值(梯度), It 为时间维度，像素差值
           假设2：局部区域 运动相同
                 对于点[x,y]附近的点[x1,y1]  [x2,y2]  , ... , [xn,yn]  都具有相同的速度 [u; v]
                 有：
                  [Ix1, Iy1;                      [It1
                   Ix2, Iy2;                       It2
                   ...               *  [u; v] = - ...
                   Ixn, Iyn;]                      Itn]
                 写成矩阵形式：
                  A * U = b
                 由两边同时左乘 A逆 得到：
                  U = A逆 * b
                 由于A矩阵的逆矩阵可能不存在，可以曲线救国改求其伪逆矩阵
                  U = (A转置*A)逆 * A转置 * b
           得到像素的水平和垂直方向速度以后，可以得到:
               速度幅值： 
                        V = sqrt(u^2 + v^2)
               速度方向：Cet = arctan(v/u)      
        e. 使用全局光流 Ft 减去 运动光流Mt 得到去除运动噪声后的 光流

    2. 特征归一化方式
        在iDT算法中，对于HOF,HOG和MBH特征采取了与DT算法（L2范数归一化）不同的方式。
           L2范数归一化  : Xi' = Xi/sqrt(X1^2 + ... + Xn^2)
        L1正则化后再对特征的每个维度开平方。
           L1范数归一化  : Xi' = Xi/(abs(X1) + ... + abs(Xn))
           Xi'' = sqrt(Xi')
        这样做能够给最后的分类准确率带来大概0.5%的提升
        
    3. 特征编码—Fisher Vector
        特征编码阶段iDT算法不再使用Bag of Features/ BOVM方法，
        (提取图像的SIFT特征，通过（KMeans聚类）,VQ矢量量化，构建视觉词典（码本）)
        
        而是使用效果更好的Fisher Vector编码.
        FV采用混合高斯模型（GMM）构建码本，
        但是FV不只是存储视觉词典的在一幅图像中出现的频率，
        并且FV还统计视觉词典与局部特征（如SIFT）的差异
        
        Fisher Vector同样也是先用大量特征训练码书，再用码书对特征进行编码。

        在iDT中使用的Fisher Vector的各个参数为：
           1.  用于训练的特征长度：trajectory+HOF+HOG+MBH = 30+96+108+192 = 426维
           2.  用于训练的特征个数：从训练集中随机采样了256000个
           3.  PCA降维比例：2，即维度除以2，降维后特征长度为 D = 426 / 2 = 213。
               先降维，后编码
           4. Fisher Vector中 高斯聚类的个数K：K=256
        故编码后得到的特征维数为2*K*D个，即109056维。
        在编码后iDT同样也使用了SVM进行分类。
        在实际的实验中，推荐使用liblinear，速度比较快。
    4. 其他改进思想 
       原先是沿着轨迹提取手工设计的特征，可以沿着轨迹利用CNN提取特征。
       
#### Fisher Vector 特征编码 主要思想是使用高斯分布来拟合单词 而不是简简单单的聚类产生中心点
[VLFeat数学推导](http://www.vlfeat.org/api/fisher-fundamentals.html)

    FV采用GMM构建视觉词典，为了可视化，这里采用二维的数据，
    然这里的数据可以是SIFT或其它局部特征，
    具体实现代码如下：

    %采用GMM模型对数据data进行拟合，构建视觉词典  
    numFeatures = 5000 ;            %样本数  
    dimension = 2 ;                 %特征维数  
    data = rand(dimension,numFeatures) ; %这里随机生成一些数据，这里data可以是SIFT或其它局部特征  

    numClusters = 30 ;  %视觉词典大小  
    [means, covariances, priors] = vl_gmm(data, numClusters); %GMM拟合data数据分布，构建视觉词典  

    这里得到的means、covariances、priors分别为GMM的均值向量，协方差矩阵和先验概率，也就是GMM的参数。

    这里用GMM构建视觉词典也存在一个问题，这是GMM模型固有的问题，
    就是当GMM中的高斯函数个数，也就是聚类数，也就是numClusters，
    若与真实的聚类数不一致的话，
    GMM表现的不是很好（针对期望最大化EM方法估计参数），具体请参见GMM。

    接下来，我们创建另一组随机向量，这些向量用Fisher Vector和刚获得的GMM来编码，
    具体代码如下：

    numDataToBeEncoded = 1000;  
    dataToBeEncoded = rand(dimension,numDataToBeEncoded);    %2*1000维数据
    % 进行FV编码
    encoding = vl_fisher(datatoBeEncoded, means, covariances, priors);
### 传统视频行为分析算法总结
#### a.	特征提取方法
    方向梯度直方图 HOG   图像平面像素水平垂直误差，再求和成梯度幅值和梯度方向，划分梯度方向，按梯度大小加权统计
    光流直方图     HOF  需要梯度图和时间梯度图来计算像素水平和垂直速度，在求合成速度幅值和方向，按上面的方式统计 
    光流梯度直方图 MBH   在光流图上计算水平和垂直光流梯度，计算合成光流梯度幅值和方向，再统计
    轨迹特征      Trajectories， 匹配点按照光流速度得到坐标，获取坐标差值，按一条轨迹串联起来，正则化之后就是一个轨迹特征。

#### b.	特征归一化方法
    L2范数归一化  :
        Xi' = Xi/sqrt(X1^2 + ... + Xn^2)
    L1范数归一化后再对特征的每个维度开平方。
        L1范数归一化  : Xi' = Xi/(abs(X1) + ... + abs(Xn))
        开平方        ：Xi'' = sqrt(Xi')

#### c.	特征编码方法
    1) 视觉词袋BOVM模型
      1. 使用K_mean聚类算法对训练数据集特征集合进行聚类，得到特征单词字典；
      2. 使用矢量量化VQ算法 或者 同步正交匹配追踪SOMP算法 对 样本数据的特征 用词典进行编码。
      3. 使用SVM分类算法对编码后的特征向量进行分类.
    2) Fisher Vector 特征编码,高斯混合模型拟合中心点
      1. 使用高斯混合模型GMM算法提取训练集特征中的聚类信息，得到 K个高斯分布表示的特征单词字典；
      2. 使用这组组K个高斯分布的线性组合来逼近这些 测试集合的特征，也就是FV编码.
         Fisher vector本质上是用似然函数的梯度vector来表达一幅图像, 说白了就是数据拟合中对参数调优的过程。

         由于每一个特征是d维的，需要K个高斯分布的线性组合，有公式5，一个Fisher vector的维数为（2*d+1）*K-1维。
      3. 使用SVM分类算法对编码后的特征向量进行分类.

      Fisher Vector步骤总结：
        1.选择GMM中K的大小
        1.用训练图片集中所有的特征（或其子集）来求解GMM（可以用EM方法），得到各个参数；
        2.取待编码的一张图像，求得其特征集合；
        3.用GMM的先验参数以及这张图像的特征集合按照以上步骤求得其fv；
        4.在对训练集中所有图片进行2,3两步的处理后可以获得fishervector的训练集，然后可以用SVM或者其他分类器进行训练。

    3) 两种编码方式对比
        经过fisher vector的编码，大大提高了图像特征的维度，能够更好的用来描述图像。
        FisherVector相对于BOV的优势在于，BOV得到的是一个及其稀疏的向量，
        由于BOV只关注了关键词的数量信息，这是一个0阶的统计信息；
        FisherVector并不稀疏，同时，除了0阶信息，
        Fisher Vector还包含了1阶(期望)信息、2阶(方差信息)，
        因此FisherVector可以更加充分地表示一幅图片。
#### d. 视频分割 类似语言识别中的 间隔点检测
    动态时间规整DTW


## 3.2 深度学习方法
### 时空双流网络结构  Two Stream Network及衍生方法
#### 提出 
[“Two-Stream Convolutional Networks for Action Recognition in Videos”（2014NIPS）](https://arxiv.org/pdf/1406.2199.pdf)

[论文博客翻译](https://blog.csdn.net/liuxiao214/article/details/78377791)

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

#### key volume的自动识别
[A Key Volume Mining Deep Framework for Action Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_A_Key_Volume_CVPR_2016_paper.pdf)

    本文主要做的是key volume的自动识别。
    通常都是将一整段动作视频进行学习，而事实上这段视频中有一些帧与动作的关系并不大。
    因此进行关键帧的学习，再在关键帧上进行CNN模型的建立有助于提高模型效果。
    本文达到了93%的正确率吗，为目前最高。
    实验效果：UCF101-93.1%，HMDB51-63.3%

### 数据数据的提纯
    输入一方面指输入的数据类型和格式，也包括数据增强的相关操作。

    双流网络中，空间网络通道的输入格式通常为单RGB图像或者是多帧RGB堆叠。
    而空间网络一般是直接对ImageNet上经典的网络进行finetune。
    虽然近年来对motion信息的关注逐渐上升，指责行为识别过度依赖背景和外貌特征，
    而缺少对运动本身的建模，但是，事实上，运动既不是名词，
    也不应该是动词，而应该是动词+名词的形式，例如：play+basketball，也可以是play+football。
    所以，个人认为，虽然应该加大的时间信息的关注，但不可否认空间特征的重要作用。

#### 空间流上 改进 提取关键帧
    空间网络主要捕捉视频帧中重要的物体特征。
    目前大部分公开数据集其实可以仅仅依赖单图像帧就可以完成对视频的分类，
    而且往往不需要分割，那么，在这种情况下，
    空间网络的输入就存在着很大的冗余，并且可能引入额外的噪声。

    是否可以提取出视频中的关键帧来提升分类的水平呢？下面这篇论文就提出了一种提取关键帧的方法。

[A Key Volume Mining Deep Framework for Action Recognition 【CVPR2016】](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_A_Key_Volume_CVPR_2016_paper.pdf)
#### 提取关键帧 改进
    虽然上面的方法可以集成到一个网络中训练，
    但是思路是按照图像分类算法RCNN中需要分步先提出候选框，挑选出关键帧。
    既然挑选前需要输入整个视频，可不可以省略挑选这个步骤，
    直接在卷积/池化操作时，重点关注那些关键帧，而忽视那些冗余帧呢？
    去年就有人提出这样的解决方法。
[AdaScan: Adaptive Scan Pooling in Deep Convolutional Neural Networks for Human Action Recognition in Videos](https://arxiv.org/pdf/1611.08240.pdf)

    注：AdaScan的效果一般，关键帧的质量比上面的Key Volume Mining效果要差一点。不过模型整体比较简单。
#### 时间流   上输入的改进 光流信息
    输入方面，空间网络目前主要集中在关键帧的研究上。
    而对于temporal通道而言，则是更多人的关注焦点。
    首先，光流的提取需要消耗大量的计算力和时间（有论文中提到几乎占据整个训练时间的90%）；
    其次，光流包含的未必是最优的的运动特征。

[On the Integration of Optical Flow and Action Recognition](https://arxiv.org/pdf/1712.08416.pdf)

#### cnn网络自学习 光流提取 
    那么，光流这种运动特征可不可以由网络自己学呢？
[Hidden Two-Stream Convolutional Networks for Action Recognition](https://arxiv.org/pdf/1704.00389.pdf)

    该论文主要参考了flownet，即使用神经网络学习生成光流图，然后作为temporal网络的输入。
    该方法提升了光流的质量，而且模型大小也比flownet小很多。
    有论文证明，光流质量的提高，尤其是对于边缘微小运动光流的提升，对分类有关键作用。
    另一方面，该论文中也比较了其余的输入格式，如RGB diff。但效果没有光流好。

    目前，除了可以考虑尝试新的数据增强方法外，如何训练出替代光流的运动特征应该是接下来的发展趋势之一。


### 信息的融合
    这里连接主要是指双流网络中时空信息的交互。
    一种是单个网络内部各层之间的交互，如ResNet/Inception；
    一种是双流网络之间的交互，包括不同fusion方式的探索，
       目前值得考虑的是参照ResNet的结构，连接双流网络。
#### 基于 ResNet 的双流融合
    空间和时序网络的主体都是ResNet，
    增加了从Motion Stream到Spatial Stream的交互。论文还探索多种方式。
    
[Spatiotemporal Multiplier Networks for Video Action Recognition](http://openaccess.thecvf.com/content_cvpr_2017/papers/Feichtenhofer_Spatiotemporal_Multiplier_Networks_CVPR_2017_paper.pdf)

#### 金字塔 双流融合

[Spatiotemporal Pyramid Network for Video Action Recognition](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Spatiotemporal_Pyramid_Network_CVPR_2017_paper.pdf)

    行为识别的关键就在于如何很好的融合空间和时序上的特征。
    作者发现，传统双流网络虽然在最后有fusion的过程，但训练中确实单独训练的，
    最终结果的失误预测往往仅来源于某一网络，并且空间/时序网络各有所长。
    论文分析了错误分类的原因：
    空间网络在视频背景相似度高的时候容易失误，
    时序网络在long-term行为中因为snippets length的长度限制容易失误。
    那么能否通过交互，实现两个网络的互补呢？

    该论文重点在于STCB模块，详情请参阅论文。
    交互方面，在保留空间、时序流的同时，对时空信息进行了一次融合，最后三路融合，得出最后结果

#### 这两篇论文从pooling的层面提高了双流的交互能力

[Attentional Pooling for Action Recognition](https://papers.nips.cc/paper/6609-attentional-pooling-for-action-recognition.pdf)

[ActionVLAD: Learning spatio-temporal aggregation for action classification](http://openaccess.thecvf.com/content_cvpr_2017/papers/Girdhar_ActionVLAD_Learning_Spatio-Temporal_CVPR_2017_paper.pdf)


#### 基于ResNet的结构探索新的双流连接方式
[Deep Convolutional Neural Networks with Merge-and-Run Mappings](https://arxiv.org/pdf/1611.07718.pdf)

#### 
