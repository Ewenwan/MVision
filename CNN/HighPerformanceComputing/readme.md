# HighPerformanceComputing 
      高性能计算(High performance computing， 缩写HPC) 
      指通常使用很多处理器（作为单个机器的一部分）
      或者某一集群中组织的几台计算机（作为单个计 算资源操作）的计算系统和环境。
      有许多类型的HPC 系统，其范围从标准计算机的大型集群，到高度专用的硬件。
      大多数基于集群的HPC系统使用高性能网络互连，比如那些来自 InfiniBand 或 Myrinet 的网络互连。
      基本的网络拓扑和组织可以使用一个简单的总线拓扑，
      在性能很高的环境中，网状网络系统在主机之间提供较短的潜伏期，
      所以可改善总体网络性能和传输速率。
      
[浮点运算和代码优化, 并行计算, Optimizer软件](http://antkillerfarm.github.io/ai/2015/10/12/float.html)

# 相关 库
      0、小米 mace
[代码](https://github.com/Ewenwan/mace)
      
      Mobile AI Compute Engine (MACE) 是一个专为移动端异构计算平台优化的神经网络计算框架。

      1、OpenVINO  intel cpu 核显 优化加速
      Intel推出OpenVINO工具包，将计算机视觉带到物联网终端
      OpenVINO（开放的视觉推理和神经网络优化）工具包
      使开发人员能够在云上（如TensorFlow，MXNet和Caffe等流行款框架）构建和训练人工智能模型，
      并将其部署到各种产品中。
      Windows*
      Linux* (supports Ubuntu*, CentOS*, and Yocto Project*)
      Linux for FPGA 
[英特尔推深度学习加速工具包 OpenVINO](https://github.com/Ewenwan/dldt)

      
      2、腾讯NCNN框架入门到应用

[代码](https://github.com/Ewenwan/ncnn)
     
     3、FeatherCNN
[代码](https://github.com/Ewenwan/FeatherCNN)

     4、Tengine 高性能神经网络推理引擎
[代码](https://github.com/Ewenwan/Tengine)

      5、百度MDL
[代码](https://github.com/Ewenwan/paddle-mobile)

      6、九言科技 绝影（Prestissimo）
[代码](https://github.com/Ewenwan/In-Prestissimo)

      7、Google量化方法 r=S(q-Z)  tflite  TensorFlow Lite  
[代码](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite)
      
      8、英伟达 TensorRT ， NVIDIA TensorRT是一种高性能神经网络推理(Inference)引擎
[代码](https://github.com/Ewenwan/TensorRT_Tutorial)

[英伟达 CUDA 和 TensorRT 代码实验](https://github.com/Ewenwan/CUDA_Test)

      9、FaceBOOK caffe2 pytorch QNNPACK  uint8量化
[QNNPACK uint8量化 ](https://github.com/Ewenwan/QNNPACK)


[深度学习框架的并行优化方法小结](https://github.com/DragonFive/myblog/blob/master/source/_posts/mpi_parallel.md)




## 卷积计算优化
    目前，卷积的计算大多采用间接计算的方式，主要有以下三种实现方式：

    1、im2col + GEMM。
       caffe等很多框架中都使用了这种计算方式，
       原因是将问题转化为矩阵乘法后可以方便的使用很多矩阵运算库（如MKL、openblas、Eigen等）。
[openblas](https://www.leiphone.com/news/201704/Puevv3ZWxn0heoEv.html)
       
[GEMM 普通矩阵乘法（General Matrix Multiplication）](https://github.com/flame/how-to-optimize-gemm/wiki)
       
       
    2、FFT变换。 
       时域卷积等于频域相乘，因此可将问题转化为简单的乘法问题。
    3、Winograd。 
       这种不太熟悉，据说在GPU上效率更高。 
       NNPACK就是FFT和Winograd方法的结合。
       
    上面三种方法执行效率都还不错，但对内存占用比较高，因为需要存储中间结果或者临时辅助变量。


    1、Strassen 算法:
    分析 CNN 的线性代数特性，增加加法减少乘法，
    这样降低了卷积运算的计算的复杂度(o(n^3) -> o(n^2.81))，
    但是这种方法不适合在硬件里面使用，这里就不做详细的介绍了。
    
    2、 MEC：
    一种内存利用率高且速度较快的卷积计算方法
[MEC: Memory-efficient Convolution for Deep Neural Network 论文](http://cn.arxiv.org/pdf/1706.06873v1)


[快速矩阵乘法 分块矩阵乘法 Strassen算法 Coppersmith-Winograd算法](http://hongbomin.com/2016/07/19/fast-matrix-multiplication/)

[博客解析](https://blog.csdn.net/shuzfan/article/details/77427979)
    
## openblas GEMM 矩阵乘法优化

![](https://static.leiphone.com/uploads/new/article/740_740/201704/58f08bf33fabd.png?imageMogr2/format/jpg/quality/90)

最原始3个for循环 (矩阵比较小的时候，速度还能快一些，当矩阵大了的时候，一定会跌下去,cache缓存问题):

![](https://static.leiphone.com/uploads/new/article/740_740/201704/58f08d87a8397.png?imageMogr2/format/jpg/quality/90)

矩阵分块，块复用，减少仿存，相当于减少内存访问：

![](https://static.leiphone.com/uploads/new/article/740_740/201704/58f08dd7b16d4.png?imageMogr2/format/jpg/quality/90)

![](https://static.leiphone.com/uploads/new/article/740_740/201704/58f08e08680b9.png?imageMogr2/format/jpg/quality/90)

操作寄存器，不是操作内存：

我可以申请一堆C 00，01这样的寄存器变量，在C语言中是register double，还有矩阵A的部分，也用寄存器变量。

当然B还是之前的方式，最后再写回C里面。

只是我们引入了寄存器变量，让更多的数据保存到寄存器里，而不是放到cache缓存里，来减轻cache的压力.

![](https://static.leiphone.com/uploads/new/article/740_740/201704/58f08e24d0ee0.png?imageMogr2/format/jpg/quality/90)


B矩阵仿存，使用指针访问，

一开始先把对应的指针位置指好，每次计算的时候只要指针连续移动就好，而不是每次读一个位置重新算一遍，这样速度就会快一些。

![](https://static.leiphone.com/uploads/new/article/740_740/201704/58f08ea8442c1.png?imageMogr2/format/jpg/quality/90)

最里层循环展开：

在最里层循环，是不是可以展开成4次，在做这个的时候，我们可以降低整个循环这部分的开销，而且让它流水的情况更好。

![](https://static.leiphone.com/uploads/new/article/740_740/201704/58f08f08cf086.png?imageMogr2/format/jpg/quality/90)

通过使用寄存器变量，使用了指针，在做了一定的底层循环展开之后，达到了红色线的性能:

![](https://static.leiphone.com/uploads/new/article/740_740/201704/58f08f44ae0fa.png?imageMogr2/format/jpg/quality/90)

之后可以使用更大的分块，在进行寄存器，指针，展开优化。
      
