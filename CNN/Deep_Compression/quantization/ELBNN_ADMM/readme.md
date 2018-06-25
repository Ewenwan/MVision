# 约束低比特(3比特)量化 Extremely Low Bit Neural Networks

[论文 Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM](https://arxiv.org/pdf/1707.09870.pdf)

[翻译](https://www.jiqizhixin.com/articles/2018-01-22-6)

[解析2](https://www.jianshu.com/p/c34ec77dae9e)

[ADMM 算法理解 对原函数不好求解，转而求解它的对偶函数，基于对对偶函数的优化，从来解出原问题](https://blog.csdn.net/danyhgc/article/details/76014478)

[ADMM 算法实现](http://web.stanford.edu/~boyd/admm.html)

![](http://file.elecfans.com/web1/M00/55/79/pIYBAFssV_WAdFmiAACFxVTKLmQ760.png)

    上图展示了阿里巴巴冷聪等人做的通过ADMM算法求解binary约束的低比特量化工作。
    从凸优化的角度，在第一个优化公式中，f(w)是网络的损失函数，
    后面会加入一项W在集合C上的loss来转化为一个优化问题。
    这个集合C取值只有正负1，如果W在满足约束C的时候，它的loss就是0；
    W在不满足约束C的时候它的loss就是正无穷。
    为了方便求解还引进了一个增广变量，保证W是等于G的，
    这样的话就可以用ADMM的方法去求解。
    
    提出一种基于低比特表示技术的神经网络压缩和加速算法。
    我们将神经网络的权重表示成离散值，
    并且离散值的形式为 2 的幂次方的形式，比如 {-4，-2，-1，0，1，2，4}。
    这样原始 32 比特的浮点型权重可以被压缩成 1-3 比特的整形权重，
    同时，原始的浮点数乘法操作可以被定点数的移位操作所替代。
    在现代处理器中，定点移位操作的速度和能耗是远远优于浮点数乘法操作的。
    
    {-1，0，1 }, 三值网络，存储只需要2bit，极大地压缩存储空间，
    同时也可以避免乘法运算，只是符号位的变化和加减操作，从而提升计算速度。
![](https://upload-images.jianshu.io/upload_images/2509688-eab1154e07f49554.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/533)
    
    {-2，-1，0，1，2}, 五值网络 和 {-4，-2，-1，0，1，2，4} 七值网络，
    需要3bit存储
    
    首先，我们将离散值权重的神经网络训练定义成一个离散约束优化问题。
    以三值网络为例，其目标函数可以表示为：
![](https://image.jiqizhixin.com/uploads/editor/e02461c9-1369-4f22-ab30-c12dfec03db5/3718302.png)

    在约束条件中引入一个 scale（尺度）参数。
    对于三值网络，我们将约束条件写成 {-a, 0, a}, a>0.
    
    这样做并不会增加计算代价，
    因为在卷积或者全连接层的计算过程中可以先和三值权重 {-1, 0, 1} 进行矩阵操作，
    然后对结果进行一个标量 scale。
    从优化的角度看，增加这个 scale 参数可以大大增加约束空间的大小，
    
    这有利于算法的收敛。如下图所示：
![](https://image.jiqizhixin.com/uploads/editor/fd9ec9ac-50dd-4867-b7bb-b80bafd16c51/5179603.jpg)
    
    对于三值网络而言，scale 参数可以将约束空间从离散的 9 个点扩增到 4 条直线。

    为了求解上述约束优化问题，我们引入 ADMM 算法。
    在此之前，我们需要对目标函数的形式做一个等价变换(对偶变换)。
![](https://image.jiqizhixin.com/uploads/editor/620d2e54-5d9b-49ef-8d11-f3b29ca39794/6118804.png)

![](https://upload-images.jianshu.io/upload_images/2509688-09b68d52c24fac35.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/700)

    其中 Ic 为指示函数，
    如果 G 符合约束条件，则 Ic(G)=0，
    否则 Ic(G) 为无穷大。
    该目标函数的增广拉格朗日形式为( 将条件 引入到 目标函数):
![](https://image.jiqizhixin.com/uploads/editor/f6b5da9e-e441-489e-a53c-da61ad8f4ca9/6905205.png)
    
        ADMM 算法将上述问题分成三个子问题进行求解，即：
![](https://image.jiqizhixin.com/uploads/editor/b47c80ae-2399-48c9-a94e-0073b7a8c716/7818706.jpg)

    与其它算法不同的是，我们在实数空间和离散空间分别求解，
    然后通过拉格朗日乘子的更新将两组解联系起来。
    第一个子问题需要找到一个网络权重最小化
![](https://image.jiqizhixin.com/uploads/editor/9a21f213-5dc6-4bf5-b751-4095fa03f6c1/8625907.png)

    在实验中我们发现使用常规的梯度下降算法求解这个问题收敛速度很慢。
    在这里我们使用 Extra-gradient 算法来对这个问题进行求解。
    Extra-gradient 算法包含两个基本步骤，分别是：
![](https://image.jiqizhixin.com/uploads/editor/13c91f04-9188-462c-9b1a-f46816b5af96/9496608.png)

    第二个子问题在离散空间中进行优化。通过简单的数学变换第二个子问题可以写成：
![](https://image.jiqizhixin.com/uploads/editor/26ce6576-e2e9-4b19-965a-a8962d0e79bb/0162109.png)

    该问题可以通过迭代优化的方法进行求解。当 a 或 Q 固定时，很容易就可以获得 Q 和 a 的解析解。
    
    除上述三值网络外，还有以下几种常用的参数空间：
![](https://upload-images.jianshu.io/upload_images/2509688-d3699da636ddfe3b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/399)

    参数空间中加入2、4、8等值后，仍然不需要乘法运算，只需进行移位操作。
    因此，通过这种方法将神经网络中的乘法操作全部替换为移位和加操作。
    
    
    
