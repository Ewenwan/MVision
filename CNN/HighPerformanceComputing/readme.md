# HighPerformanceComputing 
      高性能计算(High performance computing， 缩写HPC) 
      指通常使用很多处理器（作为单个机器的一部分）
      或者某一集群中组织的几台计算机（作为单个计 算资源操作）的计算系统和环境。
      有许多类型的HPC 系统，其范围从标准计算机的大型集群，到高度专用的硬件。
      大多数基于集群的HPC系统使用高性能网络互连，比如那些来自 InfiniBand 或 Myrinet 的网络互连。
      基本的网络拓扑和组织可以使用一个简单的总线拓扑，
      在性能很高的环境中，网状网络系统在主机之间提供较短的潜伏期，
      所以可改善总体网络性能和传输速率。
[让深度学习更高效运行的两个视角 | 计算量和访存](https://zhuanlan.zhihu.com/p/33693725)
      
[海思NNIE之Mobilefacenet量化部署](https://github.com/Ewenwan/nniefacelib)

[斯坦福大学Fall 2018课程-机器学习硬件加速器 cs217](https://cs217.stanford.edu/)
      
[浮点运算和代码优化, 并行计算, Optimizer软件](http://antkillerfarm.github.io/ai/2015/10/12/float.html)

[第十七章 模型压缩及移动端部署](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch17_%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9%E3%80%81%E5%8A%A0%E9%80%9F%E5%8F%8A%E7%A7%BB%E5%8A%A8%E7%AB%AF%E9%83%A8%E7%BD%B2/%E7%AC%AC%E5%8D%81%E4%B8%83%E7%AB%A0_%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9%E3%80%81%E5%8A%A0%E9%80%9F%E5%8F%8A%E7%A7%BB%E5%8A%A8%E7%AB%AF%E9%83%A8%E7%BD%B2.md)

# 相关 库
      0、小米 mace
[代码](https://github.com/Ewenwan/mace)
      
      Mobile AI Compute Engine (MACE) 是一个专为移动端异构计算平台优化的神经网络计算框架。

mace是基于opencl开发的，mace框架出来得比较早，当然没有比arm的computelibrary早。很多框架的GPU推理实现都或多或少的参考了computeLibrary。

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

腾讯的ncnn：使用vulkan，支持跨平台ios，android。不过ios需要通过第三方的SDK才能使用vulkan。苹果自己开发了一套metal的gpu编程API。以后ios上什么opencl，opengles,vulkan都不再是官方原生支持的GPU编程api了。

     
     3、FeatherCNN
[代码](https://github.com/Ewenwan/FeatherCNN)

     4、Tengine 高性能神经网络推理引擎
[代码](https://github.com/Ewenwan/Tengine)

      5、百度MDL
[代码](https://github.com/Ewenwan/paddle-mobile)

百度的paddle-lite：使用vulkan开发安卓版本的GPU推理，使用metal开发IOS版本的GPU推理

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


      10、阿里的mnn 

使用opencl，opengles，vulkan，metal四种GPU编程API开发了这个推理框架。据说很多公司开始把mnn纳入到公司内部的推理框架进行二次开发，估计更全面的GPU编程API支持是其一个最大优势。

      11、谷歌的tflite：

使用opengles的compute shader实现了安卓版本的GPU推理，对于IOS版本则是使用metal开发。

      12、arm中国的tengine：

tengine使用的是arm compute library框架作为底层GPU实现，据了解tengine在cpu端的优化下了很大功夫，当然作为ARM旗下的推理框架，自然对arm的架构和ISA指令更加了解。

arm compute library：这个框架是使用opencl和opengles来实现GPU推理的。该框架做得比较早。是armnn的底层推理实现。因为arm独特的ip授权模式，armnn是为了让半导体公司能直接打通安卓的android-nn框架。

13、 闭源的高通SNPE。

snpe是高通开发的一个推理框架，支持GPU推理，之前尝试分析过，一些调试数据看，内部必然存在opencl实现。  


当然，这些框架为了兼容性，都实现了CPU的推理功能。毕竟cpu推理兼容性更好，特别是现阶段几乎所有的手机端都是采用ARM的cpu。因此使用cpu的推理方案兼容性会更好。

# 背景 

Roofline Model。

这个Model是指计算机上的一个应用，它占用了两类最主要的资源：算术逻辑单元的计算资源，存储器的带宽资源。这里的计算资源以FLOPS来表示；带宽资源以byte/s表示。

Roofline model是说什么呢？横轴是Operational Intensity，就是计算的密度，单位是FLOPS/byte；纵轴是performance，也就是性能，单位是FLOPS。

图中有一条折线，这个折线开始的时候是随着计算密度的增加而增加，最终会稳定在一个固定的performance上。这个意思是：当这个应用程序的计算密度大于一定值之后，将会变成一个受算术逻辑单元的计算量所限制的程序；而这个计算密度如果小于一定值，将会变成一个受存储器带宽所限制的程序。

这里折线的拐点非常重要。这个拐点跟硬件很相关，它实际上表示的是硬件的理论计算能力和它的内存带宽之间的一个比值。

举两个具体的例子，第一个是矩阵乘矩阵，矩阵C等于A乘B，而A跟B分别是一千乘一千的矩阵。假设存储和计算都是用float 32位来表示，这样一个计算将会做1000乘1000乘1000的浮点乘加，也就是2G FLOPS的运算。我们要读取A和B，然后计算出来C，把它写回去，最少的存储器访问就是三个矩阵的大小，也就是12个MB。

另外一个是矩阵乘向量，也就是矩阵A乘向量B，等于向量C，这时候维度还是1000的情况下，它的计算量就是1000乘1000的浮点乘加，也就是2M。而存储器访问的话最少大约是1000乘于1000个浮点数，也就是4MB。

可以明显地看到上面乘矩阵的操作，它的计算量是2G，访存量是12M，那么它的这个计算量除以访存量，也就是刚刚提到的计算密度，大概是200左右。下面这个矩阵和向量中，它的计算量是2M，访存量是4M，那它的计算量除以访存量大约就只有0.5，显然这两个就是非常不同的程序。

上面矩阵乘矩阵，是一个典型的受计算量约束的程序；而下面矩阵乘向量则是一个典型的受存储器带宽所约束的程序。

小模型部署在这些硬件上，通常都是被存储带宽所限制住了，而不是被计算量所限制住。



## 卷积计算优化
    目前，卷积的计算大多采用间接计算的方式，主要有以下三种实现方式：

    1、im2col + GEMM。
       caffe等很多框架中都使用了这种计算方式，
       原因是将问题转化为矩阵乘法后可以方便的使用很多矩阵运算库（如MKL、openblas、Eigen等）。
[openblas](https://www.leiphone.com/news/201704/Puevv3ZWxn0heoEv.html)
       
[GEMM 普通矩阵乘法（General Matrix Multiplication）多种优化](https://github.com/flame/how-to-optimize-gemm/wiki)
     
       
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

BLAS是 Basic Linear Algebra Subprograms （基本线性代数子程序）的首字母缩写，主要用来做基础的矩阵计算，或者是向量计算。它分为三级：

      BLAS 1级，主要做向量与向量间的dot或乘加运算，对应元素的计算；
      BLAS 2级，主要做矩阵和向量，就类似PPT中蓝色部分所示，矩阵A*向量x， 得到一个向量y。除此之外，可能还会有对称的矩阵变形；
      BLAS 3级，主要是矩阵和矩阵的计算，最典型的是A矩阵*B矩阵，得到一个C矩阵。由矩阵的宽、高，得到一个m*n的C矩阵。


最原始3个for循环 (矩阵比较小的时候，速度还能快一些，当矩阵大了的时候，一定会跌下去,cache缓存问题):

![](https://static.leiphone.com/uploads/new/article/740_740/201704/58f08d87a8397.png?imageMogr2/format/jpg/quality/90)

矩阵分块，块复用，减少仿存，相当于减少内存访问，提高Cache利用率：

![](https://static.leiphone.com/uploads/new/article/740_740/201704/58f08dd7b16d4.png?imageMogr2/format/jpg/quality/90)

![](https://static.leiphone.com/uploads/new/article/740_740/201704/58f08e08680b9.png?imageMogr2/format/jpg/quality/90)

核心汇编优化：

* 寄存器分块
* SIMD指令
* 指令流水线优化，循环展开，重排，预取


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
      
