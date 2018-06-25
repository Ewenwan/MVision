# BNN 二值化网络

![](https://img-blog.csdn.net/20170214003827832?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGFuZ3dlaTIwMTQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

    二值化神经网络，是指在浮点型神经网络的基础上，
    将其权重矩阵中权重值(线段上) 和 各个 激活函数值(圆圈内) 同时进行二值化得到的神经网络。
        1. 一个是存储量减少，一个权重使用 1bit 就可以，而原来的浮点数需要32bits。
        2. 运算量减少， 原先浮点数的乘法运算，可以变成 二进制位的异或运算。
        

# BNN的 激活函数值 和 权重参数 都被二值化了, 前向传播是使用二值，反向传播时使用全精度梯度。 
    
[ Keras 的实现 实现了梯度的 straight-through estimator](https://github.com/Ewenwan/nn_playground/tree/master/binarynet)

[代码注解 theano 版本 采用确定性（deterministic）的二值化方式](https://github.com/Ewenwan/BinaryNet)

[torch版本 基于概率随机随机化（stochastic）的二值化, 对BN也离散化](https://github.com/Ewenwan/BinaryNet-1)

[论文 Binarized Neural Networks BNN](https://arxiv.org/pdf/1602.02830.pdf)
   


   
## **二值化方法**

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
      
## **训练二值化网络**

    提出的解决方案是：权值和梯度在训练过程中保持全精度（full precison），
    也即，训练过程中，权重依然为浮点数，
    训练完成后，再将权值二值化，以用于 inference。

    在训练过程中，权值为 32 位的浮点数，取值值限制在 [-1, 1] 之间，以保持网络的稳定性,
    为此，训练过程中，每次权值更新后，需要对权值 W 的大小进行检查，W=max(min(1,W),−1)。

    前向运算时，我们首先得到二值化的权值：Wkb=sign(Wk),k=1,⋯,n 
    然后，用 Wkb 代替 Wk：

    xk=σ(BN(Wkb * xk−1)=sign(BN(Wkb * xk−1))

    其中，BN(⋅) 为 Batch Normalization 操作。


## **前向传播时**

**对权重值W 和 激活函数值a 进行二值化**
    
    Wk = Binary(Wb)   // 权重二值化
    Sk = ak-1 * Wb    // 计算神经元输出
    ak = BN(Sk, Ck)   // BN 方式进行激活
    ak = Binary(ak)   // 激活函数值 二值化

![](https://img-blog.csdn.net/20170214010139607?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGFuZ3dlaTIwMTQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## **在反传过程时**

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

## **求各层梯度方式如下：**

![](https://img-blog.csdn.net/20170214005928789?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGFuZ3dlaTIwMTQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
    
## **梯度更新方式如下：**

![](https://img-blog.csdn.net/20170214010005900?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGFuZ3dlaTIwMTQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
    
