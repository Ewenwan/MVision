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
## glog google出的一个C++轻量级日志库
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
        google::InitGoogleLogging(program);
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

      
# prototxt文件的可视化

      1.使用在线工具netscope。

      https://ethereon.github.io/netscope/quickstart.html

      2.使用自带draw_net.py脚本。

      参考：

      http://www.cnblogs.com/zjutzz/p/5955218.html

      caffe绘制网络结构图

      http://yanglei.me/gen_proto/

      另一个在线工具。

# caffe 中 BatchNorm层  要和  Scale 层一起用才有 批规范化的效果
[参考 ](https://blog.csdn.net/Suan2014/article/details/79623495)
> 批规范化：

      1) 输入归一化 x_norm = (x-u)/std, 其中u和std是个累计计算的均值和方差。
      2）y=alpha×x_norm + beta，对归一化后的x进行比例缩放和位移。其中alpha  和beta是通过迭代学习的。
      caffe中的bn层其实只做了第一件事； 
      scale 层做了第二件事；
      scale层里为什么要设置bias_term=True，这个偏置就对应2）件事里的beta。

# 1. yolo 模型转换到 caffe下
      1.1  yolov1的caffe实现
[caffe-yolo v1 python](https://github.com/xingwangsfu/caffe-yolo)

[caffe-yolo v1  c++](https://github.com/yeahkun/caffe-yolo)

      1.2. yolov2新添了route、reorg(passtrough层)、region层(最后输入解码)，好在github上有人已经实现移植。
[移植yolo2到caffe框架](https://github.com/hustzxd/z1)

      region_layer.cpp
      region_layer.cu
      region_layer.hpp

      reorg_layer.cpp
      reorg_layer.cu
      reorg_layer.hpp

      util/ math_functions.hpp  需要修改


[caffe-yolov2](https://github.com/gklz1982/caffe-yolov2)

## 上面 的两个 caffe 的实现下载后需要修改 cudnn.hpp文件 和CMAKE文件
      \include\caffe\util\cudnn.hpp
      Makefile.config
      修改后的文件 见 文件夹

## 1.2 三个文件的作用
      1. create_yolo_prototxt.py ：  
            用来将原来的yolo的cfg文件 转成 caffe的prototxt文件，这是模型的配置文件，是描述模型的结构。
      2. create_yolo_caffemodel.py ：
            用来将yolo的weights文件转成caffe的caffemodel文件， 这是模型的参数，里面包含了各层的参数。
      3. yolo_detect.py ：这个Python程序里import了caffe，caffe的python库。
            运行这个python程序需要指定用上两个python程序转好的prototxt文件和caffemodel文件，用于初始化caffe的网络。
            并在输入的图像上得到检测结果。
            python里能够import caffe 
            你需要在caffe文件夹下make pycaffe，并设置PYTHONPATH环境变量。

### 1.2.1 yolo的cfg文件 转成 caffe的prototxt
    python create_yolo_prototxt.py yolov1_test.prototxt  yolov1.cfg
### 1.2.2 yolo的weights文件转成caffe的caffemodel
    python create_yolo_caffemodel.py -m yolov1_test.prototxt -w yolov1.weights -o yolov1.caffemodel
    python yolo_weight_to_caffemodel_v1.py -m yolov1_caffe_test.prototxt -w yolov1.weights -o yolov1_caffe.caffemodel
### 1.2.3 检测 
    python yolo_main.py -m model_filename -w weight_filename -i image_filename   
    python yolov1_caffe_main.py -m yolov1_caffe_test.prototxt -w yolov1.caffemodel -i dog.jpg

# 2. caffe 模型配置文件 prototxt 详解
[博客参考](https://blog.csdn.net/maweifei/article/details/72848185?locationNum=15&fps=1)

![](https://img-blog.csdn.net/20160327122151958?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

[caffe 模型配置文件 prototxt 详解](https://github.com/Ewenwan/MVision/blob/master/darknect/caffe/caffe_%E7%AE%80%E4%BB%8B_%E4%BD%BF%E7%94%A8.md)

[caffe 版本 yolo 过程记录](https://blog.csdn.net/u012235274/article/details/52120152)

[caffe-yolo 训练](https://blog.csdn.net/u012235274/article/details/52399425)

[YOLO算法的Caffe实现](https://blog.csdn.net/u014380165/article/details/72553074)
