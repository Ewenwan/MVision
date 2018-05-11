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
    type: "Data"
    数据格式一般有 LevelDB和 LMDB
    数据层 一般无 bottom: ,会有多个 top: 
    例如：
    top: "data"     数据 x
    top: "label"    标签 y   对应的是分类模型 监督学习
    incude{
       phase:TRAIN   一般训练和测试时是不一样的，这里表示训练阶段的层，如果没有include标签，表示即在训练阶段又在测试阶段
    }

