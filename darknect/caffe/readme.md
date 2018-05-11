# 1. yolo 模型转换到 caffe下
      1.1  yolov1的caffe实现
[caffe-yolo v1 python](https://github.com/xingwangsfu/caffe-yolo)

[caffe-yolo v1  c++](https://github.com/yeahkun/caffe-yolo)

      1.2. yolov2新添了route、reorg、region层，好在github上有人已经实现移植。
[移植yolo2到caffe框架](https://github.com/hustzxd/z1)

[caffe-yolov2](https://github.com/gklz1982/caffe-yolov2)


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
    python create_yolo_prototxt.py
### 1.2.2 yolo的weights文件转成caffe的caffemodel
    python create_yolo_caffemodel.py -m yolo_train_val.prototxt -w yolo.weights -o yolo.caffemodel
    
    
# 2. caffe 模型配置文件 prototxt 详解
![](https://img-blog.csdn.net/20160327122151958?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

      每个模型由多个 层 构成
      layer {{{{
        name: "{}" #层名字，可随意取名
        type: "{}" #层类型 数据层Data 卷积层Convolution 池化层Pooling 非线性变换、内积运算，以及数据加载、归一化和损失计算 等
        bottom: "{}"# 层入口 输入
        top: "{}"{{}}# 层出口 输出  可以有多个 bottom 和 top 表示有多条数据通道
      }}}}

## 2.1 相关头文件

    然后我们从头文件看看：
    Caffe中与Layer相关的头文件有7个，

### 2.1.1 基类层 layer.hpp: 

>  layer.hpp`头文件里，包含了这几个头文件：

      #include "caffe/blob.hpp"
      #include "caffe/common.hpp"
      #include "caffe/proto/caffe.pb.h"
      #include "caffe/util/device_alternate.hpp"

> 父类Layer，定义所有layer的基本接口。

      1. layer中有这三个主要参数：
         LayerParameter layer_param_;      
              // 这个是protobuf文件中存储的layer参数
         vector<share_ptr<Blob<Dtype>>> blobs_;
              // 这个存储的是layer的参数，在程序中用的，数据流
         vector<bool> param_propagate_down_; 
              // 这个bool表示是否计算各个blob参数的diff，即传播误差

      2. 其三个主要接口：
        virtual void SetUp(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top)
         // 需要根据实际的参数设置进行实现，对各种类型的参数初始化；
        inline Dtype Forward(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top);
         // 前向计算
        inline void Backward(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const <Blob<Dtype>*>* bottom);
         // 反向传播更新

         输入统一都是bottom，输出为top。
         其中Backward里面有个propagate_down参数，用来表示该Layer是否反向传播参数。
         Forward_cpu、Forward_gpu和Backward_cpu、Backward_gpu，这些接口都是virtual，

### 2.1.2 数据层 data_layers.hpp: 
> **data_layers.hpp这个头文件包含了这几个头文件**

      #include "boost/scoped_ptr.hpp" // 是一个类似于auto_ptr的智能指针
        // scoped_ptr(局部指针)名字的含义：这个智能指针只能在作用域里使用，不希望被转让
        // 把拷贝构造函数和赋值操作都声明为私有的，禁止对智能指针的复制操作，保证了被它管理的指针不能被转让所有权。
        // private:
        // scoped_ptr(scoped_ptr const &);
        // scoped_ptr & operator=(scoped_ptr const &);
      #include "hdf5.h" 
      #include "leveldb/db.h"//数据类型 数据库文件
      #include "lmdb.h"

      #include "caffe/blob.hpp"
      #include "caffe/common.hpp"
      #include "caffe/filler.hpp"// 在网络初始化时，根据layer的定义进行初始参数的填充，
            // type == "constant"  ConstantFiller<Dtype>(param);
            // type == "gaussian"  GaussianFiller<Dtype>(param);
            // type == "positive_unitball"   PositiveUnitballFiller<Dtype>(param);
            // type == "uniform"   UniformFiller<Dtype>(param);
            // type == "xavier"    XavierFiller<Dtype>(param);
            // 
      #include "caffe/internal_thread.hpp"// 里面封装了pthread函数，
        // 继承的子类可以得到一个单独的线程，主要作用是在计算当前的一批数据时，在后台获取新一批的数据。
      #include "caffe/layer.hpp"
      #include "caffe/proto/caffe.pb.h"

      继承自父类Layer，定义与输入数据操作相关的子Layer，
      例如DataLayer，HDF5DataLayer和ImageDataLayer等。

      data_layer作为原始数据的输入层，处于整个网络的最底层，
      它可以从数据库leveldb、lmdb中读取数据，也可以直接从内存中读取，还可以从hdf5，甚至是原始的图像读入数据。

> LevelDB是Google公司搞的一个高性能的key/value存储库，调用简单，数据是被Snappy压缩，据说效率很多，可以减少磁盘I/O.

> LMDB（Lightning Memory-Mapped Database），是个和levelDB类似的key/value存储库，但效果似乎更好些，其首页上写道“ultra-fast，ultra-compact”，

> HDF（Hierarchical Data Format）是一种为存储和处理大容量科学数据而设计的文件格式及相应的库文件，

      当前最流行的版本是HDF5,其文件包含两种基本数据对象：
            群组（group）：类似文件夹，可以包含多个数据集或下级群组；
            数据集（dataset）：数据内容，可以是多维数组，也可以是更复杂的数据类型。

### 2.1.3 vision_layers.hpp: 
      继承自父类Layer，定义与特征表达相关的子Layer，
      例如 卷积层ConvolutionLayer，池化层PoolingLayer和 LRNLayer等。

      vision_layer主要是图像卷积的操作，像convolusion、pooling、LRN都在里面，
      按官方文档的说法，是可以输出图像的，这个要看具体实现代码了。
      里面有个im2col的实现，看caffe作者的解释，主要是为了加速卷积的。

### 2.1.4 neuron_layers.hpp: 
      继承自父类Layer，定义与非线性变换相关的子Layer，神经元激活层，
      例如ReLULayer，TanHLayer和SigmoidLayer等。

      输入了data后，就要计算了，比如常见的sigmoid、tanh等等，
      这些都计算操作被抽象成了neuron_layers.hpp里面的类NeuronLayer，
      这个层只负责具体的计算，因此明确定义了输入
      ExactNumBottomBlobs()和ExactNumTopBlobs()都是常量1,即输入一个blob，输出一个blob。

### 2.1.5 common_layers.hpp: 继承自父类Layer，定义与中间结果数据变形、逐元素操作相关的子Layer，
      例如 通道合并ConcatLayer，点乘 InnerProductLayer和 SoftmaxLayer等 softmax归一化。
      NeruonLayer仅仅负责简单的一对一计算，
      而剩下的那些复杂的计算则通通放在了common_layers.hpp中。像
      ArgMaxLayer、ConcatLayer、FlattenLayer、SoftmaxLayer、SplitLayer和SliceLayer等
      各种对blob增减修改的操作。

### 2.1.6 loss_layers.hpp: 
      继承自父类Layer，定义与输出误差计算相关的子Layer，
      例如 欧几里得距离损失 EuclideanLossLayer，SoftmaxWithLossLayer 和 HingeLossLayer等。
      data layer和common layer都是中间计算层，
      虽然会涉及到反向传播，但反向传播的源头来自于loss_layer，即网络的最终端。

      这一层因为要计算误差，所以输入都是2个blob，输出1个blob。

### 2.1.7 layer_factory.hpp: 

      Layer工厂模式类，
      负责维护现有可用layer和相应layer构造方法的映射表。



## 2.2 数据层 Data

    数据通过数据层进入Caffe，数据层在整个网络的底部。
    数据可以来自高效的数据库（LevelDB 或者 LMDB），直接来自内存。
    如果不追求高效性，可以以HDF5或者一般图像的格式从硬盘读取数据。
    
   一些基本的操作，如：mean subtraction, 
                     scaling, 
                     random cropping, and 
                     mirroring 均可以直接在数据层上进行指定。

    type: "Data"
    数据格式一般有 LevelDB和 LMDB
    数据层 一般无 bottom: ,会有多个 top: 
    例如：
    top: "data"     数据 x
    top: "label"    标签 y   对应的是分类模型 监督学习
    incude{
       phase:TRAIN   一般训练和测试时是不一样的，这里表示训练阶段的层，如果没有include标签，表示即在训练阶段又在测试阶段
    }

### 2.2.1 数据库格式数据  Database
      类型：Data

      必须参数：
            source: 包含数据的目录名称
            batch_size: 一次处理 的 输入的数量，过大内存不够
      可选参数：
            rand_skip: 在开始的时候从输入中跳过这个数值，这在异步随机梯度下降（SGD）的时候非常有用
            backend [default LEVELDB]: 选择使用 LEVELDB 或者 LMDB
### 2.2.2 直接来自内存  In-Memory
      类型: MemoryData
      必需参数：
      batch_size, channels, height, width: 指定从内存读取数据的大小
      MemoryData层直接从内存中读取数据，而不是拷贝过来。
      因此，要使用它的话，你必须调用  
      MemoryDataLayer::Reset (from C++)
      或者Net.set_input_arrays (from Python)以此指定一块连续的数据（通常是一个四维张量）。
### 2.2.3 HDF5 Input
      类型: HDF5Data
      必要参数：
            source: 需要读取的文件名
            batch_size：一次处理的输入的数量

### 2.2.4 HDF5 Output
      类型: HDF5Output
      必要参数：
      file_name: 输出的文件名
      HDF5的作用和这节中的其他的层不一样，它是把输入的blobs写到硬盘
      
### 2.2.5 来自图像文件 Images
      类型: ImageData
      必要参数：
            source: text文件的名字，每一行给出一张图片的文件名和label
            batch_size: 一个batch中图片的数量

      可选参数：
            rand_skip：在开始的时候从输入中跳过这个数值，这在异步随机梯度下降（SGD）的时候非常有用
            shuffle [default false]
            new_height, new_width: 把所有的图像resize到这个大小

## 2.3 激励层（neuron_layers） 激活层
      一般来说，激励层是element-wise的操作，输入和输出的大小相同，一般情况下就是一个非线性函数。

      数据输入输出维度不变
      输入：
            n×c×h×w
      输出：
            n×c×h×w
### 2.3.1 ReLU / Rectified-Linear and Leaky-ReLU 最小阈值激活
      标准， f(x) = max(0,x) ，当x > 0时输出x，但x <= 0时输出negative_slope 阈值
      Leaky-ReLU   max(0.1x,x)

      定义：
      layer {
       name: "relu1"
       type: "ReLU"
       bottom: "conv1"
       top: "conv1"
      }

### 2.3.2 Sigmoid    负指数导数激活
      标准   f(x) = 1/(1+exp(-x))  x = 0, f(x) = y = 0.5
      映射到 0~1之间
      sigmoid函数连续，光滑，严格单调，以(0,0.5)中心对称，是一个非常良好的阈值函数。
      当x趋近负无穷时，y趋近于0；趋近于正无穷时，y趋近于1；x=0时，y=0.5。
      当然，在x超出[-6,6]的范围后，函数值基本上没有变化，值非常接近，在应用中一般不考虑。

      导数：
      f′(x) = f(x) * (1 − f(x))

      定义：
      layer {
        name: "encode1neuron"
        bottom: "encode1"
        top: "encode1neuron"
        type: "Sigmoid"
      }

### 2.3.3  TanH / Hyperbolic Tangent  双曲正切
      请注意sigmoid函数和TanH函数在纵轴上的区别。
      sigmoid函数将实数映射到(0,1)。
      TanH将实数映射到(-1,1)。
       tanh(x) = ( exp(x) − exp(−x) ) / ( exp(x) + exp(−x) )

       定义：
      layer {
        name: "layer"
        bottom: "in"
        top: "out"
        type: "TanH"
      }
    
### 2.3.4 绝对值激活 Absolute Value
      ABSVAL层通过 y =  abs(x) 计算每一个输入x的输出。

      定义：
      layer {
        name: "layer"
        bottom: "in"
        top: "out"
        type: "AbsVal"
      }
      
### 2.3.5 Power 平移乘方激活 
      POWER层通过 y = (shift + scale * x) ^ power计算每一个输入x的输出。

      定义：
      layer {
        name: "layer"
        bottom: "in"
        top: "out"
        type: "Power"
        power_param {
          power: 1
          scale: 1
          shift: 0
        }
      }
      
### 2.3.6 BNLL 二项正态对数似然 激活
      BNLL (binomial normal log likelihood) 层通过 
      y = log(1 + exp(x)) 计算每一个输入x的输出

      定义：
      layer {
        name: "layer"
        bottom: "in"
        top: "out"
        type: BNLL
      }

## 2.4 视觉层（vision_layers）

