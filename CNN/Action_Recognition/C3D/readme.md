# 3D卷积视频分类
[代码参考](https://github.com/Ewenwan/C3D)


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

[博客](https://blog.csdn.net/u014380165/article/details/78986416)

    改进ResNet内部连接中的卷积形式。然后，超深网络，一般人显然只能空有想法，望而却步。
