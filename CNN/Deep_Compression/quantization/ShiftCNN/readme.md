# ShiftCNN 二进制位量化网络 哈希函数的味道啊  仅权重量化
[ShiftCNN](http://cn.arxiv.org/pdf/1706.02393v1)

[代码 caffe-quant-shiftcnn ](https://github.com/Ewenwan/caffe-quant-shiftcnn)

[博客](https://blog.csdn.net/shuzfan/article/details/77856900)

    整个对参数进行量化的流程如下：
![](https://img-blog.csdn.net/20170905204744197?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2h1emZhbg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


    一个利用低精度和量化技术实现的神经网络压缩与加速方案。
    
>最优量化

    量化可以看作用离散码本描述样本分布。 
    优化目标(最大概率准则)和优化方法(L1和L2正则化)通常导致了神经网络参数呈现中心对称的非均匀分布。
    因此，一个最佳的量化码本应当是一个非均匀分布的码本。 
    这也是为什么BinaryNet(-1,+1)、ternary quantization(-1,0,+1)这种策略性能不足的一个重要原因。
    
    需要注意的是，
    量化之前需要对参数进行范围归一化，
    即除以最大值的绝对值，这样保证参数的绝对值都小于1。
    该量化方法具有码本小、量化简单、量化误差小的优点。
    
>量化

    ShiftCNN所采用是一种相当巧妙的类似于残差量化的方法。

    完整的码本包含 N 个子码本。 
    每个码本包含 M=2^B−1 个码字，即每一个码字可以用 B bit 表示。 
    每个码本定义如下：

     Cn=0, ±2^−n+1, ±2^−n, …, ±2^−n−⌊M/2⌋+2
    假设 N=2，B=4，则码本为

    C1=0, ±2^−1, ±2^−2, ±2^−3, ±2^−4, ±2^−5, ±2^−6
    C2=0, ±2^−2, ±2^−3, ±2^−4, ±2^−5, ±2^−6, ±2^−7
    
    于是，每一个权重都可以使用 N*B bit 的索引通过下式求和计算得到：
    wi' = sum(Cn[id(n)])

>卷积计算

    卷积计算的复杂度主要来自乘法计算。
    ShiftCNN采用简单的移位和加法来实现乘法，从而减少计算量。
    比如计算 y=wx, 而 w 通过量化已经被我们表示成了,
    类似于 2^−1 + 2^−2 + 2^−3 这种形式，
    于是 y = x>>1 + x>>2 + x>>3 

![](https://img-blog.csdn.net/20170905211321692?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2h1emZhbg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

    其中累加单元如下：(主要是逻辑控制与求和):
![](https://img-blog.csdn.net/20170905211836994?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2h1emZhbg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

    实验
    ShiftCNN量化后无需再训练。 
    个人觉得再训练应该会更好一点。
    再训练-再量化-再训练-再量化
