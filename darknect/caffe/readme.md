[yolo_darknet 转 caffe](https://github.com/Ewenwan/MVision/blob/master/darknect/caffe/yolo_darknet_to_caffe.md)

# caffe使用
[caffe 安装](https://github.com/Ewenwan/MVision/blob/master/darknect/caffe/caffe_%E5%AE%89%E8%A3%85.md)

[Caffe代码解析](http://alanse7en.github.io/caffedai-ma-jie-xi-2/)

[caffe网络模型结构在线可视化](http://ethereon.github.io/netscope/#/editor)

[CAFFE使用 源码分析等](https://blog.csdn.net/fangjin_kl/article/list/3)

[caffe 模型配置文件 prototxt 详解](https://github.com/Ewenwan/MVision/blob/master/darknect/caffe/caffe_%E7%AE%80%E4%BB%8B_%E4%BD%BF%E7%94%A8.md)

[caffe.proto 系统变量层类型参数配置文件](https://github.com/Ewenwan/MVision/blob/master/darknect/caffe/caffe.proto%E7%AE%80%E4%BB%8B.md)

[ caffe_简介_使用.md](https://github.com/Ewenwan/MVision/blob/master/darknect/caffe/caffe_%E7%AE%80%E4%BB%8B_%E4%BD%BF%E7%94%A8.md)

[Caffe使用教程_c++接口](https://github.com/Ewenwan/MVision/blob/master/darknect/caffe/Caffe%E4%BD%BF%E7%94%A8%E6%95%99%E7%A8%8B_c%2B%2B%E6%8E%A5%E5%8F%A3.md)

[caffe 模型搜集](https://github.com/SnailTyan/caffe-model-zoo)
![screenshot](https://user-images.githubusercontent.com/21311442/33640664-cbcbeff2-da6c-11e7-97c8-1ad8d7fdf4c0.png)

[caffe详解](https://yufeigan.github.io/2014/12/09/Caffe%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B03-Layer%E7%9A%84%E7%9B%B8%E5%85%B3%E5%AD%A6%E4%B9%A0/)

# 主要类对象
      caffe大致可以分为三层结构blob，layer，net。
      数据的保存，交换以及操作都是以blob的形式进行的，
      layer是模型和计算的基础，
      net整和并连接layer,
      solver则是模型的优化求解。
## 数据Blob 是Caffe的基本数据结构, 4维的数组(Num, Channels, Height, Width) 
      设Blob数据维度为 number N x channel K x height H x width W，数据批次数量，通道数量，高宽尺寸
      Blob是row-major 行优先 保存的，因此在(n, k, h, w)位置的值，
      实际物理位置为((n * K + k) * H + h) * W + w，其中Number/N是batch size。
      
      lob同时保存了data和diff(梯度)，访问data或diff有两种方法:
            1. const Dtype* cpu_data() const; //不修改值 指针指向常量 / gpu_data() 
            2. Dtype* mutable_cpu_data();     //修改值               / mutable_gpu_data()
      Blob会使用SyncedMem(分配内存和释放内存类) 自动决定什么时候去copy data以提高运行效率，
      通常情况是仅当gnu或cpu修改后有copy操作，文档里面给了一个例子：     
            
```c
using caffe::Blob; // 作为数据传输的媒介，无论是网络权重参数，还是输入数据，都是转化为Blob数据结构来存储
// 可以把Blob看成一个有4维的结构体（包含数据和梯度），而实际上，它们只是一维的指针而已，其4维结构通过shape属性得以计算出来。
// shared_ptr<SyncedMemory> data_ //数据
// shared_ptr<SyncedMemory> diff_ //梯度
// void Blob<Dtype>::Reshape(const int num, const int channels, const int height,const int width)
// 在更高一级的Layer中Blob用下面的形式表示学习到的参数：
// vector<shared_ptr<Blob<Dtype> > > blobs_;
// 这里使用的是一个Blob的容器是因为某些Layer包含多组学习参数，比如多个卷积核的卷积层。
// 以及Layer所传递的数据形式，后面还会涉及到这里：
// vector<Blob<Dtype>*> &bottom;
// vector<Blob<Dtype>*> *top
```

## 各种层实现 卷积 池化 
      在Layer中 输入数据input data用bottom表示，
                输出数据 output data用top表示。
      每一层定义了三种操作:
          1. setup（Layer初始化）, 
          2. forward（正向传导，根据input计算output）, 
          3. backward（反向传导计算，根据output计算input的梯度）。
          forward和backward有GPU(大部分)和CPU两个版本的实现。
                
```c
using caffe::Layer;// 作为网络的基础单元，神经网络中层与层间的数据节点、前后传递都在该数据结构中被实现，
// 层类种类丰富，比如常用的卷积层、全连接层、pooling层等等，大大地增加了网络的多样性.
// NeuronLayer类 定义于neuron_layers.hpp中, 比如Dropout运算，激活函数ReLu，Sigmoid等.
// LossLayer类 定义于loss_layers.hpp中，其派生类会产生loss，只有这些层能够产生loss。
// 数据层 定义于data_layer.hpp中，作为网络的最底层，主要实现数据格式的转换。
// 特征表达层（我自己分的类）定义于vision_layers.hpp, 包含卷积操作，Pooling操作.
// 网络连接层和激活函数（我自己分的类）定义于common_layers.hpp，包括了常用的 全连接层 InnerProductLayer 类。

// 在Layer内部，数据主要有两种传递方式，正向传导（Forward）和反向传导（Backward）。Forward和Backward有CPU和GPU（部分有）两种实现。
// virtual void Forward(const vector<Blob<Dtype>*> &bottom, vector<Blob<Dtype>*> *top) = 0;
// virtual void Backward(const vector<Blob<Dtype>*> &top, const vector<bool> &propagate_down,  vector<Blob<Dtype>*> *bottom) = 0;
// Layer类派生出来的层类通过这实现这两个虚函数，产生了各式各样功能的层类。
// Forward是从根据bottom计算top的过程，Backward则相反（根据top计算bottom）。
```

## 网络Net 由各种层Layer组成的 无回路有向图DAG
      Layer之间的连接由一个文本文件描述。
      网络模型初始化Net::Init()会产生blob和layer并调用　Layer::SetUp。
      在此过程中Net会报告初始化进程(大量网络载入信息)。这里的初始化与设备无关。
      在初始化之后通过Caffe::set_mode()设置Caffe::mode()来选择运行平台CPU或GPU，结果是相同的。
```c
using caffe::Net;// 作为网络的整体骨架，决定了网络中的层次数目以及各个层的类别等信息
// Net用容器的形式将多个Layer有序地放在一起，其自身实现的功能主要是对逐层Layer进行初始化，
// 以及提供Update( )的接口（更新网络参数），本身不能对参数进行有效地学习过程。
// vector<shared_ptr<Layer<Dtype> > > layers_;
// 同样Net也有它自己的 前向传播 和反向传播
// vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>* > & bottom, Dtype* loss = NULL); // 前传得到　loss
// void Net<Dtype>::Backward();// 反传播得到各个层参数的梯度
// 他们是对整个网络的前向和方向传导，各调用一次就可以计算出网络的loss了。
```

## 求解器Solver 各种数值优化的方法
```c
using caffe::Solver;// 作为网络的求解策略，涉及到求解优化问题的策略选择以及参数确定方面，修改这个模块的话一般都会是研究DL的优化求解的方向。
// 包含一个Net的指针，主要是实现了训练模型参数所采用的优化算法，它所派生的类就可以对整个网络进行训练了。
// shared_ptr<Net<Dtype> > net_;
// 不同的模型训练方法通过重载函数ComputeUpdateValue( )实现计算update参数的核心功能.
// virtual void ComputeUpdateValue() = 0;
// 进行整个网络训练过程（也就是你运行Caffe训练某个模型）的时候，实际上是在运行caffe.cpp中的train( )函数，
// 而这个函数实际上是实例化一个Solver对象，初始化后调用了Solver中的Solve( )方法。
// 而这个Solve( )函数主要就是在迭代运行下面这两个函数，就是刚才介绍的哪几个函数。
// ComputeUpdateValue();
// net_->Update();
```

# 目录结构
## caffe文件夹下主要文件： 这表示文件夹

    data    用于存放下载的训练数据
    docs    帮助文档
    example 一些代码样例
    matlab  MATLAB接口文件
    python  Python接口文件
    model   一些配置好的模型参数
    scripts 一些文档和数据用到的脚本

## 下面是核心代码文件夹：

    tools   保存的源码是用于生成二进制处理程序的，caffe在训练时实际是直接调用这些二进制文件。
    include Caffe的实现代码的头文件
    src     实现Caffe的源文件
    后面的学习主要围绕后面两个文件目录（include和src）下的代码展开
# 源码结构
      由于include和src两个目录在层次上基本一一对应因此主要分析src即可了解文件结构。
      这里顺便提到一个有意思的东西，我是在Sublime上面利用SublimeClang插件分析代码的（顺便推荐下这插件，值得花点时间装）。
      Sublime tx2 SublimeClang 参考配置: https://www.cnblogs.com/wxquare/p/4751297.html
      在配置的时候发现会有错误提示找不到”caffe/proto/caffe.pb.h”，去看了下果然没有，但编译的时候没有报错，说明是生成过后又删除了，
      查看Makefile文件后发现这里用了proto编译的，所以在”src/caffe/proto”下面用CMakeLists文件就可以编译出来了。
[Google Protocol Buffer 的使用和原理 ](https://www.ibm.com/developerworks/cn/linux/l-cn-gpb/)

## src
### gtest 
      google test一个用于测试的库,
      你make runtest时看见的很多绿色RUN OK就是它，这个与caffe的学习无关，不过是个有用的库
### caffe 关键的代码都在这里了
      1. test 用gtest测试caffe的代码
      2. util 数据转换时用的一些代码。
         caffe速度快，很大程度得益于内存设计上的优化（blob数据结构采用proto）和对卷积的优化（部分与im2col相关）.
      3. proto  即所谓的“Protobuf”，全称“Google Protocol Buffer”，是一种数据存储格式，帮助caffe提速。
      4. layers 深度神经网络中的基本结构就是一层层互不相同的网络了，
      5. 这个文件夹下的源文件以及目前位置“src/caffe”中包含的我还没有提到的所有.cpp文件,
         就是caffe的核心目录下的核心代码了。
### caffe核心代码
      1. /layers         此文件夹下面的代码全部至少继承了类Layer;
      2. blob[.cpp .h]   基本的数据结构Blob类;
      3. common[.cpp .h] 定义Caffe类;
      4. data_transformer[.cpp .h] 输入数据的基本操作类DataTransformer;
      5. internal_thread[.cpp .h]  使用boost::thread线程库;
      6. layer_factory.cpp layer.h 层类Layer ;
      7. net[.cpp .h]              网络结构类Net ;
      8. solver[.cpp .h]           优化方法类Solver  ;
      9. syncedmem[.cpp .h]        分配内存和释放内存类CaffeMallocHost，用于同步GPU，CPU数据
      
# 层Layer
Layer是所有层的基类，在Layer的基础上衍生出来的有5种Layers：

      1. 数据层 data_layer
      2. 神经元层 neuron_layer
      3. 损失函数层 loss_layer
      4. 网络连接层和激活函数 common_layer 替换成各种层名字的文件
      5. 特征表达层 vision_layer   也展开成各自层对应的文件 
      ... 新版本caffe 把每种层类型都写了一个头文件和源文件

它们都有对应的[.hpp .cpp]文件声明和实现了各个类的接口。

      在Layer中 输入数据input data用bottom表示，
                输出数据 output data用top表示。
      每一层军定义了三种操作:
          1. setup（Layer初始化）, 
          2. forward（正向传导，根据input计算output）, 
          3. backward（反向传导计算，根据output计算input的梯度）。
          forward和backward有GPU(大部分)和CPU两个版本的实现。
      还有其他一些特有的操作
# 依赖库介绍
## 1. glog,  google出的一个C++轻量级日志库
[博文介绍](http://www.cnblogs.com/tianyajuanke/archive/2013/02/22/2921850.html)

      代码中充斥这类似 
      HECK_EQ(数字相等检查)、 CHECK_NE(不相等)、CHECK_LE(小于等于)、CHECK_LT(小于)、CHECK_GE(大于等于)、CHECK_GT(大于)
      CHECK_NOTNULL(指针非空检查)、CHECK_STRNE()
      CHECK_STREQ(字符串相等检查)、
      CHECK_DOUBLE_EQ(浮点数相等检查)、
      CHECK_GT(大于检查)
      函数,
      这就是glog里面的 (CHECK 宏) ，类似ASSERT()的断言。
      当通过该宏指定的条件不成立的时候，程序会中止，并且记录对应的日志信息。
      
      还有打印日志函数 LOG(INFO),日志输出到 stderr(标准错误输出)
### glog捕捉 程序段错误信息 在 caffe::GlobalInit(&argc, &argv); 有用到
      
```c 
捕捉 段错误 核心已转储 信息 core dumped  方便调试错误

// 通过 google::InstallFailureSignalHandler(); 即可注册，将 core dumped 信息输出到 stderr，如：

#include <glog/logging.h>
#include <string>
#include <fstream>

//将信息输出到单独的文件和 LOG(ERROR)
void SignalHandle(const char* data, int size)
{
    std::ofstream fs("glog_dump.log",std::ios::app);
    std::string str = std::string(data,size);
    fs<<str;
    fs.close();
    LOG(ERROR)<<str;
}

class GLogHelper
{
public:
    GLogHelper(char* program)
    {
        google::InitGoogleLogging(program);// 传入可执行文件吗
        FLAGS_colorlogtostderr=true;
        google::InstallFailureSignalHandler();
        //默认捕捉 SIGSEGV 信号信息输出会输出到 stderr，可以通过下面的方法自定义输出方式：
        google::InstallFailureWriter(&SignalHandle);
    }
    ~GLogHelper()
    {
        google::ShutdownGoogleLogging();
    }
};

void fun()
{
    int* pi = new int;// 在堆中申请一个变量，由在栈中的指针pi指针指向
    delete pi;//删除对应堆中的内存
    pi = 0;   // 指针置空
    int j = *pi;// 对一个空指针进行解引用，报段错误
}

int main(int argc,char* argv[])
{
    GLogHelper gh(argv[0]);//这个可以捕捉core dump的详细信息，可以定位到在fun()函数处出错
    fun();
}
```
## 2.gflags ,是google的一个开源的处理命令行参数的库
[参考](https://blog.csdn.net/lezardfu/article/details/23753741)

      首先需要 使用 gflags的宏：DEFINE_xxxxx(变量名，默认值，help-string)  定义命令行参数
```c
// 求解器prototxt文件名
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
```
      1. 首先需要include "gflags.h"（废话，-_-b）
            #include <gflags/gflags.h>

      2. 将需要的命令行参数使用gflags的宏：DEFINE_xxxxx(变量名，默认值，help-string) 定义在文件当中，注意全局域(放在文件最前面的部分)。
      gflags支持以下类型：

          DEFINE_bool: boolean         布尔量
          DEFINE_int32: 32-bit integer 32位整形
          DEFINE_int64: 64-bit integer 64位整形
          DEFINE_uint64: unsigned 64-bit integer 64位无符号整形
          DEFINE_double: double        64位浮点型
          DEFINE_string: C++ string    字符串string
```c
// 求解器prototxt文件名
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
```
      3. 在main函数中加入：（一般是放在main函数的头几行，越早了解用户的需求越好么^_^）
            google::ParseCommandLineFlags(&argc, &argv, true);
            argc　参数计数 counter
            argv  参数列表 vector 字符串列表向量
            第三个参数，如果设为true，则该函数处理完成后，argv中只保留argv[0]，argc会被设置为1。
                       如果为false，则argv和argc会被保留，但是注意函数会调整argv中的顺序
```c
caffe::GlobalInit(&argc, &argv);
// 原函数
void GlobalInit(int* pargc, char*** pargv) {
  // Google flags.   namespace gflags = google;
  ::gflags::ParseCommandLineFlags(pargc, pargv, true);
  // Google logging.   glog中的日志等级
  ::google::InitGoogleLogging(*(pargv)[0]);
  // Provide a backtrace on segfault.
  ::google::InstallFailureSignalHandler();//  glog的 异常捕获(段错误)处理
}
```
      4. 这样，在后续代码中可以使用FLAGS_变量名访问对应的命令行参数了

            printf("%s", FLAGS_mystr);

      5. 最后，编译成可执行文件之后，用户可以使用：executable --参数1=值1 --参数2=值2 ... 来为这些命令行参数赋值。

            ./mycmd --var1="test" --var2=3.141592654 --var3=32767 --mybool1=true --mybool2 --nomybool3
### gflags进阶使用
      1. 在其他文件中使用定义的flags变量：
            有些时候需要在main之外的文件使用定义的flags变量，
            这时候可以使用宏定义DECLARE_xxx(变量名)声明一下（就和c++中全局变量的使用是一样的，extern一下一样）
            
      2. 定制你自己的help信息与version信息：(gflags里面已经定义了-h和--version，你可以通过以下方式定制它们的内容)
            version信息：使用google::SetVersionString设定，使用google::VersionString访问
            help信息：使用google::SetUsageMessage设定，使用google::ProgramUsage访问
            注意：google::SetUsageMessage和google::SetVersionString必须在google::ParseCommandLineFlags之前执行
            
## gtest Google的开源C++单元测试框架 
      gtest Google的开源C++单元测试框架，caffe里面test代码中大量用到，网上教程也是一大堆。
[参考](http://www.cnblogs.com/coderzh/archive/2009/04/06/1426755.html)

## 关于CPU加速
      Caffe推荐的BLAS（Basic Linear Algebra Subprograms）有三个选择ATLAS，Intel MKL，OpenBLAS。
      其中ATLAS是caffe是默认选择开源免费，如果没有安装CUDA的不太推荐使用，因为CPU多线程的支持不太好；
      Intel MKL是商业库要收费，我没有试过但caffe的作者安装的是这个库，估计效果应该是最好的；
      OpenBLAS开源免费，支持CPU多线程，我安装的就是这个。
      
      1. 安装ATLAS 
            sudo apt-get install libatlas-base-dev
            
      2. 安装OpenBLAS
            sudo apt-get install libopenblas-base
## BLAS与boost::thread加速
## BLAS 
http://www.netlib.org/blas/blasqr.pdf 
      
      
      BLAS (Basic Linear Algebra Subprograms, 基础线性代数子程序库),
      是一个应用程序接口（API）标准，说的简单点就是向量、矩阵之间乘加这些运算。
      BLAS虽然本身就有实现但 效率不高，因此有大量的开源或商业项目对BLAS进行优化,
      比如OpenBLAS（开源），Intel MKL（收费），ATLAS（开源）。
      我用的是OpenBLAS这个库。
## OpenBLAS
[参考](https://www.leiphone.com/news/201704/Puevv3ZWxn0heoEv.html)

      OpenBLAS是C语言实现的，这个库安装比较简单，如上面，唯一的一个问题是使用方法。
      前面介绍BLAS提供了接口，文档在这里 http://www.netlib.org/blas/blasqr.pdf 
      这个文档中：
         BLAS 1级 ,Level 1: 是vector与vector的操作， 向量与向量, 主要做向量与向量间的dot或乘加运算，对应元素的计算
         BLAS 2级, Level 2: 是vector与matrix的操作， 向量与矩阵, 就类似下图中蓝色部分所示，矩阵A*向量x， 得到一个向量y。
         BLAS 3级，Level 3: 是matrix与matrix的操作,  矩阵与矩阵, 最典型的是A矩阵*B矩阵，得到一个C矩阵。由矩阵的宽、高，得到一个m*n的C矩阵。
      每个函数的开头有一个x表示精度比如替换成 s表示float类型(实数)，d表示double类型，c表示复数。
      
      其实虽然函数很多但其实使用方法大同小异，BLAS之所以分的这么细（区分到对称矩阵，三角矩阵）是为了方便针对不同的情况做优化。
      所以其实搞清楚最关键的矩阵与矩阵的运算就已经理解了一大半。
![](https://static.leiphone.com/uploads/new/article/740_740/201704/58f08bf33fabd.png?imageMogr2/format/jpg/quality/90)

![](https://static.leiphone.com/uploads/new/article/740_740/201704/58f08c200cf65.png?imageMogr2/format/jpg/quality/90)
      
      基于矩阵类学习的深度学习，有90%或者更多的时间是通过BLAS来操作的。
      当然，随着新的算法出现，卷积层对3*3的卷积核有专门的算法，
      或者用FFT类类算法也可以做，但是在通用上，展矩阵来做也非常广泛。
      
      针对开源的而言，有如下几种，之前比较知名的是GoToBLAS，和OpenBLAS有很深的渊源，但是在2010年终止开发了.
![](https://static.leiphone.com/uploads/new/article/740_740/201704/58f08c6b9c8ff.png?imageMogr2/format/jpg/quality/90)

### 矩阵相乘  
      https://github.com/xianyi/OpenBLAS/wiki/User-Manual
      以dgemm为例，全称为double-precision generic matrix-matrix muliplication，就是矩阵相乘，
      在OpenBLAS中的声明是:  
      A * B = C
      [M K] * [K N] = [M N]
```c
cblas_dgemm(const enum CBLAS_ORDER Order,         // 指定矩阵的存储方式如RowMajor 行优先/列优先
            const enum CBLAS_TRANSPOSE TransA,    // A运算后是否转置
            const enum CBLAS_TRANSPOSE TransB,    // B运算后是否转置
            const blasint M,                      // A的行数
            const blasint N,                      // B的列数
            const blasint K,                      // A的列数
            const double alpha,                   // 公式中的alpha
            const double *A,                      // A
            const blasint lda,                    // A一行的存储间隔
            const double *B,                      // B
            const blasint ldb,                    // B一行的存储间隔
            const double beta,                    // 公式中的beta
            double *C,                            // C
            const blasint ldc)                    // C一行的存储间隔
```
因为这里的A、B、C矩阵都是以一维数组的形式存储所以需要告诉函数他们一行的存储间隔就是lda、ldb、ldc它们。      
      
      示例：
```c
#include <cblas.h>
#include <stdio.h>

void main()
{
  int i=0;
  double A[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};// 3*2   
  double B[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};// 2*3
  double C[9] = {.5,.5,.5,.5,.5,.5,.5,.5,.5};// 3*3
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,3,3,2,1,A, 3, B, 3,2,C,3);

  for(i=0; i<9; i++)
    printf("%lf ", C[i]);
  printf("\n");
}
编译:
gcc -o test_cblas_open test_cblas_dgemm.c -I /your_path/OpenBLAS/include/ -L/your_path/OpenBLAS/lib -lopenblas -lpthread -lgfortran
```
## Boost 相当强大的C++拓展库
      用百度百科的话说，Boost库是一个经过千锤百炼、可移植、
      提供源代码的C++库,作为标准库的后备,是C++标准化进程的发动机之一。
      实际感受就是一个相当强大的C++拓展库，很多C++标准库里面没有的功能得以实现。
      最近就用到了ublas，thread，date_time这三个模块。这里做一些简要的介绍。
[参考](https://yufeigan.github.io/2015/01/02/Caffe%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B05-BLAS%E4%B8%8Eboost-thread%E5%8A%A0%E9%80%9F/)

## ublas boost的BLAS实现 矩阵运算
      调用位置boost::numeric::ublas,虽然速度一般，
      但用起来非常方便可以直接用 + - 运算符号操作，还可以直接用 << >> 等标准输入输出流。
## date_time 计时
      在我计算多线程运行时间的时候发现标准C++提供的std::clock_t，
      对于多CPU跑线程的情况会把几个CPU的时间加在一起，
      所以采用了 boost::posix_time::ptime 这种类型计数。
      解决了计时不准确的问题。
## thread 线程库 是一个跨平台的线程封装库
      Boost提供的thread虽然功能据说不是非常强大，
      但是由于使用C++的思想重新设计，使用起来相对比较方便。
      网上文档非常多，比如：
      
[1. C++ Boost Thread 编程指南](http://www.cnblogs.com/chengmin/archive/2011/12/29/2306416.html)

[2. Boost::Thread使用示例](https://blog.csdn.net/zhuxiaoyang2000/article/details/6588031)
### 1 创建线程
```c
#include <boost/thread/thread.hpp>
#include <iostream>
 
void hello()
{
	std::cout<<"Hello world, I'm a thread!"<<std::endl;
}
 
int main()
{
	boost::thread thrd(&hello);// 传入需要线程执行的函数的指针
	thrd.join();
 
	system("pause");
	return 0;
}
```
### 2 避免不同线程同时访问共享区域 互斥体（mutex，mutual exclusion的缩写）   
      当一个线程想要访问共享区时，首先要做的就是锁住（lock）互斥体。
      如果其他的线程已经锁住了互斥体，那么就必须先等那个线程将互斥体解锁，
      这样就保证了同一时刻只有一个线程能访问共享区域。
      
      Boost线程库支持两大类互斥体，包括简单互斥体（simple mutex)和递归互斥体（recursive mutex)。
      如果同一个线程对互斥体上了两次锁，就会发生死锁（deadlock），也就是说所有的等待解锁的线程将一直等下去。
      
      有了递归互斥体，单个线程就可以对互斥体多次上锁，
      当然也必须解锁同样次数来保证其他线程可以对这个互斥体上锁。
      
      
      一个线程可以有三种方法来对一个互斥体加锁：
          一直等到没有其他线程对互斥体加锁。
          如果有其他互斥体已经对互斥体加锁就立即返回。
          一直等到没有其他线程互斥体加锁，直到超时。
          
      Boost线程库提供了6中互斥体类型，下面是按照效率进行排序：
            boost::mutex,
            boost::try_mutex,
            boost::timed_mutex,
            boost::recursive_mutex,
            boost::recursive_try_mutex,
            boost::recursive_timed_mutex
```c
    #include <boost/thread/thread.hpp>
    #include <boost/thread/mutex.hpp>
    #include <iostream>
     
    boost::mutex io_mutex;// 互斥体
     
    struct count
    {
    	count(int id) : id(id) {}
     
    	void operator()()
    	{
    		for(int i = 0; i < 10; ++i)
    		{
    			boost::mutex::scoped_lock lock(io_mutex);// 对互斥体上锁
    			std::cout<<id<<": "<<i<<std::endl;
    		}
    	}
     
    	int id;
    };
     
    int main()
    {
    	boost::thread thrd1(count(1));
    	boost::thread thrd2(count(2));
    	thrd1.join();
    	thrd2.join();
     
    	system("pause");
    	return 0;
    }
```

只让一个线程输出信息到屏幕：
```c
boost::mutex io_mutex;// 互斥体
void foo(){
    {
        boost::mutex::scoped_lock lock(io_mutex);//上锁
        std::cout << "something output!" << std::endl;
    }
    // something to do!
}

// 用这种方法多个函数在对统一个数据操作的时候就不会有冲突了。
```
## 线程的并行化 boost::thread_group
```c
boost::thread_group group;
for (int i = 0; i < 15; ++i)
    group.create_thread(aFunctionToExecute);
group.join_all();// 当执行join_all()的时候才是真正的并行了程序。
```
## 成员函数没有实例但又要传参的方法

      编译时候错误：error: reference to non-static member function must be called; did you mean to call it with no arguments?
      查了Google发现是因为我定义的类中的成员函数用group.create_thread()中调用了没有实例化的成员函数解决方法是使用std::bind
      
      group.create_thread(boost::bind(&myfunction, this)); 


# prototxt
## 文件的可视化
      1.使用在线工具netscope。

      https://ethereon.github.io/netscope/quickstart.html

      2.使用自带draw_net.py脚本。

      参考：

      http://www.cnblogs.com/zjutzz/p/5955218.html

      caffe绘制网络结构图

      http://yanglei.me/gen_proto/

      另一个在线工具。
      
## caffe 模型配置文件 prototxt 详解
 
[博客参考](https://blog.csdn.net/maweifei/article/details/72848185?locationNum=15&fps=1)

![](https://img-blog.csdn.net/20160327122151958?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

[caffe 模型配置文件 prototxt 详解](https://github.com/Ewenwan/MVision/blob/master/darknect/caffe/caffe_%E7%AE%80%E4%BB%8B_%E4%BD%BF%E7%94%A8.md)


# caffe 中 BatchNorm层  要和  Scale 层一起用才有 批规范化的效果
[参考 ](https://blog.csdn.net/Suan2014/article/details/79623495)
> 批规范化：

      1) 输入归一化 x_norm = (x-u)/std, 其中u和std是个累计计算的均值和方差。
      2）y=alpha×x_norm + beta，对归一化后的x进行比例缩放和位移。其中alpha  和beta是通过迭代学习的。
      caffe中的bn层其实只做了第一件事； 
      scale 层做了第二件事；
      scale层里为什么要设置bias_term=True，这个偏置就对应2）件事里的beta。
