# TSN 结构 
[Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/pdf/1608.00859.pdf)

[caffe code](https://github.com/yjxiong/temporal-segment-networks)

[TSN（Temporal Segment Networks）代码实验](https://blog.csdn.net/zhang_can/article/details/79704084)

[(TSN)实验及错误日志](https://blog.csdn.net/cheese_pop/article/details/79958090)

[tensorFlow PyTorch 版本](https://github.com/yjxiong/tsn-pytorch)

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
# 安装 下载项目代码，并编译
## 安装编译
    git clone --recursive https://github.com/yjxiong/temporal-segment-networks
    bash build_all.sh; 
    或者多GPU并行：
    MPI_PREFIX=<root path to openmpi installation> 
    bash build_all.sh MPI_ON 
    build_all.sh文件会下载opencv 2.4.13，
    denseflow(用来截取视频帧和光流)，
    并且编译caffe-action (双流网络结构)
    这里有一点值得注意的是需要先clone代码，再编译。
    如果是从网上download的代码直接编译的话，会因为缺少部分文件导致编译失败。 
    
## 获取视频帧和光流
    论文中使用的数据库是HMDB-51和UCF-101，可以到他们的数据库官网中下载，并解压。 
    获取视频帧和光流代码： 
    bash scripts/extract_optical_flow.sh SRC_FOLDER OUT_FOLDER NUM_WORKER 
    各参数含义如下： 
    - SRC_FOLDER 数据集路径 
    - OUT_FOLDER 提取的rgb帧和光流帧 
    - NUM_WORKER 使用的gpu数量 >= 1

## 下载预训练好的模型
    bash scripts/get_reference_models.sh 
    模型比较大，网络连接不通顺或者有精力的话，可以自己直接复制链接，从网页上download。      
## 测试
    UCF101 split1部分：

    rgb流测试 部分
    python tools/eval_net.py ucf101 1 rgb /data3/UCF-all-in-one/ucf_frame/ \ 
    models/ucf101/tsn_bn_inception_rgb_deploy.prototxt \
    models/ucf101_split_1_tsn_rgb_reference_bn_inception.caffemodel \
    --num_worker 4 --save_scores rgb_score

    flow流测试 部分
    python tools/eval_net.py ucf101 1 flow /data3/UCF-all-in-one/ucf_transed/ \
    models/ucf101/tsn_bn_inception_flow_deploy.prototxt \
    models/ucf101_split_1_tsn_flow_reference_bn_inception.caffemodel \
    --num_worker 4 --save_scores ucf101/flow_score

    融合 fusion
    python tools/eval_scores.py ucf101/rgb_score.npz ucf101/flow_score.npz --score_weights 1 1.5 


# TSN改进版本之一  加权融合
    改进的地方主要在于fusion部分，不同的片段的应该有不同的权重，而这部分由网络学习而得，最后由SVM分类得到结果。
[Deep Local Video Feature for Action Recognition 【CVPR2017】](https://arxiv.org/pdf/1701.07368.pdf)

# TSN改进版本二  时间推理
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
    
