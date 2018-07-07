# 残差网络
[v2_tf](https://github.com/tensorflow/models/blob/master/research/adv_imagenet_models/inception_resnet_v2.py)

[何凯明深度残差学习 幻灯片](http://image-net.org/challenges/talks/ilsvrc2015_deep_residual_learning_kaiminghe.pdf)

[论文](https://arxiv.org/pdf/1512.03385.pdf)

[官方代码](https://github.com/Ewenwan/deep-residual-networks)

* ResNet 论文翻译
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
[中文版](http://noahsnail.com/2017/07/31/2017-7-31-ResNet%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
[中英文对照](http://noahsnail.com/2017/07/31/2017-7-31-ResNet%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)


# 核心思想
 
    结合不同卷积层的特征，添加直通通路，可以使得梯度传播的更远，网络可以更深
    f(x) + W*x
    f(x) 为 23x3的卷积 
    实际中，考虑计算的成本，对残差块做了计算C，即将2个3x3的卷积层替换为 1x1 + 3x3 + 1x1 。
    新结构中的中间3x3的卷积层首先在一个降维1x1卷积层下减少了计算，然后在另一个1x1的卷积层下做了还原，既保持了精度又减少了计算量。
# 核心模块
    ________________________________>
    |                                 +  f(x) + x
    x-----> 1x1 + 3x3 + 1x1 卷积 -----> 

# 网络模型
    残差网络 f(x) + W*x
    50个卷积层
    ResNet50
    2018/04/22
    1  64个输出  7*7卷积核 步长 2 224*224图像输入 大小减半（+BN + RELU + MaxPol）
    2  3个 3*3卷积核  64输出的残差模块 256/4 = 64     且第一个残差块的第一个卷积步长为1
    3  4个 3*3卷积核  128输出的残差模块 512/4 = 128   且第一个残差块的第一个卷积步长为2      
    4  6个 3*3卷积核  256输出的残差模块 1024/4 = 256  且第一个残差块的第一个卷积步长为2  
    5  3个  3*3卷积核 512输出的残差模块 2048/4 = 512  且第一个残差块的第一个卷积步长为2  
    6  均值池化 
    7  全连接层 输出 1000  类
    8  softmax分类 预测类别输出
    实际中，考虑计算的成本，对残差块做了计算优化，即将2个3x3的卷积层替换为 1x1 + 3x3 + 1x1 。
    新结构中的中间3x3的卷积层首先在一个降维1x1卷积层下减少了计算，然后在另一个1x1的卷积层下做了还原，既保持了精度又减少了计算量。

# ResNeXt 
    ResNeXt是ResNet的加强版，将ResNet原本简单的“plain版残差结构”替换成了“Inception版残差结构”
    F(x) 由一系列的 卷积层通道 组合而成
[博客参考](https://blog.csdn.net/jningwei/article/details/80059533)

[代码参考](https://github.com/facebookresearch/ResNeXt)

# DenseNet 密集网络
## 每一层的输入来自于前面所有层的输出

[论文](https://arxiv.org/pdf/1608.06993.pdf)

[代码](https://github.com/liuzhuang13/DenseNet)

[tf实现](https://github.com/LaurentMazare/deep-models/tree/master/densenet)

![DenseNet 密集网络](https://img-blog.csdn.net/20171208164855253?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdHV6aXhpbmk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
