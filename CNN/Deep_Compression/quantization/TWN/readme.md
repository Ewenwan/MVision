# TWN 三值系数网络
    权值三值化的核心：
        首先，认为多权值相对比于二值化具有更好的网络泛化能力。
        其次，认为权值的分布接近于一个正态分布和一个均匀分布的组合。
        最后，使用一个 scale 参数去最小化三值化前的权值和三值化之后的权值的 L2 距离。   
        
[caffe-代码](https://github.com/Ewenwan/caffe-twns)

[论文 Ternary weight networks](https://arxiv.org/pdf/1605.04711.pdf)

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
    
[this blog](http://blog.csdn.net/xjtu_noc_wei/article/details/52862282) is useful for implementation

## Experimental Results

Three data sets are used in this paper, including MNIST, CIFAR-10, ImageNet. To different data sets, the authors conducted experiments using LeNet-5 (32-C5 + MP2 + 64-C5 + MP2 + 512FC + SVM), VGG-inspired network (2$$\times$$(128-C3) + MP2 + 2$$\times$$(256-C3) + MP2 + 2$$\times$$(512-C3) + MP2 + 1024-FC + Softmax), ResNet-18, respectively. Network architecture and parameters setting for different data sets are shown as follows:

|                                          | MNIST   | CIFAR-10 | ImageNet             |
| ---------------------------------------- | ------- | -------- | -------------------- |
| network architecture                     | LeNet-5 | VGG-7    | ResNet-18 (B)        |
| weight decay                             | 1e-4    | 1e-4     | 1e-4                 |
| mini-batch size of BN                    | 50      | 100      | 64($$\times$$4 GPUs) |
| initial learning rate                    | 0.01    | 0.1      | 0.1                  |
| learning rate decay (divided by 10) epochs | 15, 25  | 80, 120  | 30, 40, 50           |
| momentum                                 | 0.9     | 0.9      | 0.9                  |

Comparison of the proposed method and the previous methods are shown as follows:

|          Method           | MINIST | CIFAR-10 | ImageNet Top1 (ResNet-18 / ResNet-18B) | ImageNet Top5 (ResNet-18 / ResNet-18B) |
| :-----------------------: | :----: | :------: | :------------------------------------: | :------------------------------------: |
|            TWN            | 99.35  |  92.56   |              61.8 / 65.3               |              84.2 / 86.2               |
|           BPWN            | 99.05  |  90.18   |              57.5 / 61.6               |              81.2 / 83.9               |
|   FPWN (full precision)   | 99.41  |  92.88   |              65.4 / 67.6               |              86.76 / 88.0              |
|      Binary Connect       | 98.82  |  91.73   |                   -                    |                   -                    |
| Binarized Neural Networks |  88.6  |  89.85   |                   -                    |                   -                    |
|  Binary Weight Networks   |   -    |    -     |                  60.8                  |                  83.0                  |
|         XNOR-Net          |   -    |    -     |                  51.2                  |                  73.2                  |

