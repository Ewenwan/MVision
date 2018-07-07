# DenseNet 密集网络
## 每一层的输入来自于前面所有层的输出

[论文](https://arxiv.org/pdf/1608.06993.pdf)

[代码](https://github.com/liuzhuang13/DenseNet)

[tf实现](https://github.com/LaurentMazare/deep-models/tree/master/densenet)

[MXNet版本代码（有ImageNet预训练模型）](https://github.com/miraclewkf/DenseNet)

[Caffe代码](https://github.com/shicai/DenseNet-Caffe)

[解读](https://www.leiphone.com/news/201708/0MNOwwfvWiAu43WO.html)

    受 Highway、ResNet 等算法思路的启发，提出一种跨层的连接网络，思路非常简单，直接上图：
![](https://img-blog.csdn.net/20170816220927962)

  
## 算法思路

       作者这个提法比较大胆，每个层的 input 包括之前所有层的信息，
       通过将前面N多个层的 Feature 组合起来，形成对特征更丰富的描述和判别。
       从思想上来讲，是比较容易接受的，看一个完整的网络结构图：
![](https://img-blog.csdn.net/20170816220933272)
  
  
       这个网络包含3个 Dense Block，中间通过 Convolution 和 Pooling 连接。

       但是明显这将带来很大的计算量，我们来看作者是怎么处理的？

       由于每个Layer的输入会比较多，因此可以减少每一层的 Channel 数量，Feature 利用率比较高，
       整体算下来，同样的连接数量，会比 ResNet 的 Feature 更少，通过实验对比，用一半的计算量达到了 ResNet 的效果。
