
[ Caffe框架源码剖析](https://blog.csdn.net/tianrolin)   

# 2. caffe 模型配置文件 prototxt 详解
[博客参考](https://blog.csdn.net/maweifei/article/details/72848185?locationNum=15&fps=1)

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

      layer {  
        name: "data"  
        type: "Data"  
        top: "data"  
        top: "label"  
        include {  
          phase: TRAIN     # 用于训练 
        }  
        transform_param {  
          mirror: 1            # 镜像
          crop_size: 227   #
          # substract mean value（RGB three channel）: 
          #these mean_values can equivalently be replaced with a mean.binaryproto file as   
          # mean_file: name_of_mean_file.binaryproto  
          mean_value: 104  # 去中心化（减去平均值）
          mean_value: 117  
          mean_value: 123  
        }  
        data_param {  
          source: "examples/imagenet/ilsvrc12_train_lmdb"  # 数据库文件名
          batch_size: 32   # 每次处理的样本数目
          backend: LMDB    # 数据库类型，默认为LMDB，可选LevelDB
          # rand_skip：在开始的时候跳过rand_skip个输入数据，这个对异步SGD有效
        }   
      }  


      均值文件 name_of_mean_file.binaryproto 
      cd ~/caffe  
      build/tools/compute_image_mean examples/imagenet/ilsvr12_train_lmdb   
      data/ilsvrc12/imagenet_mean.binaryproto  

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
```c++
template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
	// (a*x,x) 正常无上截断
  }
}
//////////////////////////////////////////////
////================================
// 前传
template <typename Dtype>
__global__ void ReLUXForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope, Dtype relux_cur_max) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
    out[index] = out[index] < relux_cur_max ? out[index] : relux_cur_max;
	// (a*x, x|cur_max) 上截断=========
  }
}

template <typename Dtype>
__global__ void ReLUXRForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope, Dtype relux_cur_max, Dtype rate) {
  CUDA_KERNEL_LOOP(index, n) 
  {
    if (in[index] < 0) {
      out[index] = in[index] * negative_slope; // (a*x, x|cur_max, relux_cur_max+ det*ret)
    } 
	else if (in[index] < relux_cur_max) 
	{
      out[index] = in[index];
    } 
	else 
	{
      out[index] = relux_cur_max + (in[index] - relux_cur_max) * rate;// 一次调整斜率 收缩
    }
  }
}

template <typename Dtype>
__global__ void ReLUYRForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope, Dtype relux_cur_max, Dtype relux_last_max, Dtype rate) {
  CUDA_KERNEL_LOOP(index, n) {
    if (in[index] < 0) 
	{
      out[index] = in[index] * negative_slope;
    } 
	else if (in[index] < relux_cur_max) 
	{
      out[index] = in[index];
    } 
	else if (in[index] < relux_last_max) 
	{
      out[index] = relux_cur_max + (in[index] - relux_cur_max) * rate;//  一次调整斜率 收缩
    } 
	else 
	{
      out[index] = relux_cur_max + (relux_last_max - relux_cur_max) * rate;// 二次调整斜率 收缩
    }
  }
}
/////// 反向传播   倒数===================================

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) 
  {//   (a*diff, diff)  正常无上截断====
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * negative_slope);
  }
}

template <typename Dtype>
__global__ void ReLUXBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope, Dtype relux_cur_max) 
	{
  CUDA_KERNEL_LOOP(index, n) 
  {// 有上截断 (a*diff, diff, 0)
    out_diff[index] = in_diff[index] * (((in_data[index] > 0) && (in_data[index] <= relux_cur_max))
        + (in_data[index] <= 0) * negative_slope);
  }
}

template <typename Dtype>
__global__ void ReLUXRBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope, Dtype relux_cur_max, float rate) 
	{
  CUDA_KERNEL_LOOP(index, n) 
  {// 一次斜率收缩======== (a*diff, diff, rate)
    if (in_data[index] > relux_cur_max) {
      out_diff[index] = in_diff[index] * rate;
    } 
	else {
      out_diff[index] = in_diff[index] * ((in_data[index] > 0)
          + (in_data[index] <= 0) * negative_slope);
    }
  }
}

template <typename Dtype>
__global__ void ReLUYRBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope, Dtype relux_cur_max, Dtype relux_last_max, float rate) 
	{
  CUDA_KERNEL_LOOP(index, n) 
  {// 二次斜率调整
    if (in_data[index] > relux_last_max) 
	{
      out_diff[index] = 0;// 这里有点问题？？
    } 
	else if (in_data[index] > relux_cur_max) 
	{
      out_diff[index] = in_diff[index] * rate;
    } 
	else 
	{
      out_diff[index] = in_diff[index] * ((in_data[index] > 0)
          + (in_data[index] <= 0) * negative_slope);
    }
  }
}
```
      

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
### 2.4.1 卷积层(Convolution)

      类型：CONVOLUTION 
      例子：
      layers { 
          name: "conv1"             # 名字
          type: CONVOLUTION         # 卷积层类型
          bottom: "data"            # 输入层
          top: "conv1"              # 输出层
          blobs_lr: 1               # 权重卷积核参数更新学习率 learning rate multiplier for the filters 
          blobs_lr: 2               # 偏置参数更新学习率 learning rate multiplier for the biases 
          weight_decay: 1           # 权重卷积核参数更新 衰减系数 weight decay multiplier for the filters 
          weight_decay: 0           # 偏置参数更新 衰减系数 weight decay multiplier for the biases 
          convolution_param { 
              num_output: 96        # 卷积核数量，即输出通道数量 learn 96 filters     必选
              kernel_size: 11       # 卷积核尺寸 11*11 each filter is 11x11         必选
              stride: 4             # 卷积步长，输出特征图尺寸，尺寸变为 1/4 step 4 pixels between each filter application 
              weight_filler {       # 权重初始化设置
                  type: "gaussian"  # 高斯分布初始化卷积核参数 initialize the filters from a Gaussian 
                  std: 0.01         # 标准差0.01，默认均值0 distribution with stdev 0.01 (default mean: 0) } 
                  bias_filler {     # 偏置初始化设置
                      type: "constant" # 常量 initialize the biases to zero (0) 
                      value: 0         # 0
                   } 
              }
          }
      }

#### 2.4.1.1 可选参数：
      bias_filler：             偏置的初始化方法
      bias_term [default true]：指定是否是否开启偏置项  y = w*x + b  或则  y = w*x
      pad (or pad_h and pad_w) [default 0]：         指定在输入图的每一边加上多少个像素 一般为 卷积核尺寸-1的一半
      stride (or stride_h and stride_w) [default 1]：指定滤波器的步长
      group (g) [default 1]: 如果g>1，那么将每个滤波器都限定只与某个输入的子集有关联。
                             换句话说，将输入分为g组，同时将输出也分为g组。
                             那么第i组输出只与第i组输入有关。  之后再经过点卷积 或者通道重排 结合不同通道信息

#### 2.4.1.2通过卷积后的大小变化：
      输入：
           [n,i_w,i_h,W]
           特征图大小 i_w,i_h
           通道数量 W
           个数 n
      卷积核尺寸:
           [k_w,k_h,W] × V        步长 [s_h,s_w]
           卷积核尺寸 [k_w,k_h,W]
           个数       V
      输出：
           [n,o_w,o_h,V]
           特征图大小 o_w,o_h
           通道数量 V
           个数 n
      其中：
           o_h = (i_h + 2×padh − kernelh)/ s_h + 1,
           o_w = (o_w + 2×padw − kernelw)/ s_w + 1,
      一般填充数量都会等于 (kernel - 1)/2
      所以卷积核输出的特征图尺寸一般不会变换，变化的是 通道数量
      如果有步长，则，等于 原尺寸/步长
      
#### 2.4.1.3in examples\mnist\lenet_train_test.prototxt   
    layer {  
      name: "conv1"       // 层的名字  
      type: "Convolution" // 层的类型，说明具体执行哪一种计算  
      bottom: "data" // 层的输入数据Blob的名字  
      top: "conv1"   // 层的输出数据Blob的名字  
      param {        // 层的权值和偏置相关参数  
        lr_mult: 1   // 权重学习率
      }   
      param {  
        lr_mult: 2   // 偏置学习率
      }  
      convolution_param { // 卷积层卷积运算相关的参数  
        num_output: 20    // 输出通道数量 卷积核个数
        kernel_size: 5    // 5*5卷积核尺寸
        stride: 1         // 步长为1 特征图尺寸不变 有填充
        weight_filler {   // 权重 初始化
          type: "xavier"  
        }  
        bias_filler {     // 偏置初始化
          type: "constant"
        }  
      }  
    }  

### 2.4.2 池化层（Pooling）
      类型：POOLING

      例子：
      layers { 
          name: "pool1"        # 名字
          type: POOLING        # 池化层类型
          bottom: "conv1"      # 输入层
          top: "pool1"         # 输出层
          pooling_param {      # 池化层 参数
              pool: MAX        # 最大值池化MAX  均值池化MEAN 
              kernel_size: 3   # 池化核大小 pool over a 3x3 region   必须要的参数
              stride: 2        # 步长 降低分辨率 step two pixels (in the bottom blob) between pooling regions 
          }
      }

#### 2.4.2.1 可选参数：
      pool [default MAX]：pooling的方法，目前有MAX, AVE, 和STOCHASTIC三种方法
      pad (or pad_h and pad_w) [default 0]：指定在输入的每一遍加上多少个像素
      stride (or stride_h and stride_w) [default 1]：指定过滤器的步长

#### 2.4.2.2 通过池化后的大小变化 
      输入：
           [n,i_w,i_h,W]

      池化核尺寸:
           [k_w,k_h]  数量 W  步长 [s_h,s_w]
      输出：
           [n,o_w,o_h,W]
      其中：
           o_h = (i_h + 2×padh − k_h)/ s_h + 1,
           o_w = (o_w + 2×padw − k_w)/ s_w + 1,
      一般填充数量都会等于 (kernel - 1)/2
      所以输出尺寸基本上 等于 原尺寸/步长
### 2.4.3 局部响应归一化层 LRN(Local Response Normalization)
      类型：LRN

      可选参数：
      local_size [default 5]：
            对于cross channel LRN为 需要求和的  邻近channel的数量；
            对于within channel LRN为 需要求和的 空间区域的边长；

      alpha [default 1]： scaling参数,缩放比例；
      beta [default 5]：  指数β；
      norm_region [default ACROSS_CHANNELS]: 
             选择LRN实现的方法：
                1. ACROSS_CHANNELS ；
                2. WITHIN_CHANNEL
      计算公式：
      对每一个输入除以 xi / (1 + (α/n)⋅ ∑ xi^2 )^β

      在这里，参数α是scaling参数，参数β是指数。而参数 n 对应local_size 的大小。
#### 解析：
      一种提高准确度的技术方法。
      跟激活函数是有区别的，LRN一般是在激活、池化后进行的一中处理方法。
      类似的还有 BN层 批归一化。
      是对一个局部的输入区域进行的归一化。
      有两种不同的形式：
      1. ACCROSS_CHANNEL；
      2. WITHIN_CHANNEL。
      其实很好从字面上进行理解。

      第一种方法综合了不同的channel(类似点卷积的左右)，
          而在一个channel里面只取1*1（所以size是localsize×1×1）。
      第二种方法中，
          不在channel方向上扩展，只在单一channel上进行空间扩展（所以size是1×localsize×localsize）。
     

## 2.5 损失层（Loss Layers）
      深度学习是通过最小化 网络输出和目标的 误差Loss 来 驱动学习。
      
### 2.5.1 指数归一化 对数误差 Softmax loss  softmax+Loss组成

      类型: SoftmaxWithLoss
      
      Softmax Loss层应用于多标签分类。
      对于输入，计算了multinomial logistic loss。
      在概念上近似等于一个Softmax层加上一个multinomial logistic loss层。
      但在梯度的计算上更加稳定。

      ai = zi/sum(exp(zi))   softmax 指数 归一化
      Loss = -log(aj)        指定类别 负对数 误差
      对指定类别的 输出概率(归一化后为0~1之间) 做log
      越接近1，越接近目标值，loss越趋近于0
      在0~1之间 log为负数，所以在前面加了一个 符号
      
### 2.5.2 欧氏距离误差   EuclideanLoss
      类型: EuclideanLoss
      Euclidean loss层计算了两个输入之差的平方和
      sum(zi-yi)^2
      
### 2.5.3 HINGE_LOSS    “最大间隔”分类误差  （max margin）
      最著名的应用是作为SVM的目标函数。
      其二分类情况下，公式如下： 
          l(y) =  max( 0, 1 − t⋅y)
      其中，y是预测值（-1到1之间），t为目标值（±1）。
      其含义为，y的值在-1到1之间就可以了，并不鼓励|y|>1，
      即并不鼓励分类器过度自信，让某个可以正确分类的样本距离分割线的距离超过1并不会有任何奖励。
      
      类型: HingeLoss

      例子：

      带有L1正则化项:
      L1 Normlayers { 
          name: "loss" 
          type: HINGE_LOSS 
          bottom: "pred" 
          bottom: "label"
      } 

      带有L2正则化项:
      L2 Normlayers { 
          name: "loss" 
          type: HINGE_LOSS 
          bottom: "pred" 
          bottom: "label" 
          top: "loss" 
          hinge_loss_param { 
              norm: L2 
          }
      }
      可选参数：
            norm [default L1]: 选择L1或者L2范数

      输入：
            n×c×h×w Predictions   预测值
            n×1×1×1 Labels        真实标签
      输出
            1×1×1×1 Computed Loss

### 2.5.4 Sigmoid 交叉熵损失函数 
      类型：SigmoidCrossEntropyLoss
      sigmod 将输出 映射到 0~1之间： pi = 1/(1+exp(-zi))
      交叉熵损失： 1/n * sum (yi*log(pi))

### 2.5.5 信息增益损失函数（InformationGain Loss）
      类型:InfogainLoss
      这是在文本处理中用到的损失函数.


### 2.5.6 Accuracy and Top-k
      类型：Accuracy

      用来计算输出和目标的正确率，事实上这不是一个loss，而且没有backward这一步。
      
## 2.6 一般层（Common Layers）

### 2.6.1 全连接层 Inner Product    FC
      类型：InnerProduct

      例子：
      layer {
        name: "fc8"            # 名字
        type: "InnerProduct"   # 类型
        # 权重学习率、衰减 learning rate and decay multipliers for the weights
        param { lr_mult: 1 decay_mult: 1 }
        # 偏置学习率、衰减 learning rate and decay multipliers for the biases
        param { lr_mult: 2 decay_mult: 0 }
        inner_product_param {
          num_output: 1000     # 输  出 数量
          weight_filler {
            type: "gaussian"   # 权重初始化
            std: 0.01
          }
          bias_filler {        # 偏置初始化
            type: "constant"
            value: 0
          }
        }
        bottom: "fc7"
        top: "fc8"
      }

      通过全连接层后的大小变化：

      输入：n×ci×hi×wi   其实需要先经过 flatten层摊平 1*N * N*m --> 1*m
      输出：n×co×1×1

### 2.6.2 分割层 Splitting

      类型：Split
      Splitting层可以把一个输入blob分离成多个输出blobs。
      这个用在当需要把一个blob输入到多个输出层的时候。

### 2.6.3 摊平层  Flattening
      类型：Flatten

      Flatten层是把一个输入的大小为n * c * h * w变成一个简单的向量，其大小为 n * (c*h*w) * 1 * 1。

### 2.6.4 变形层 Reshape
    类型：Reshape

    例子：
    layer {
        name: "reshape"
        type: "Reshape"
        bottom: "input"
        top: "output"
        reshape_param {
          shape {
            dim: 0  # copy the dimension from below  直接从底层复制
            dim: 2
            dim: 3
            dim: -1 # infer it from the other dimensions 从其他的数据里面推测这一维应该是多少。
          }
        }
      }
#### 2.6.4.1 说明  
      输入：单独的一个blob，可以是任意维；

      输出：同样的blob，但是它的维度已经被我们人为地改变，维度的数据由reshap_param定义。

      可选参数：
            shape
      Reshape层被用于改变输入的维度，而不改变输入的具体数据。
      就像Flatten层一样。只是维度被改变而已，这个过程不涉及数据的拷贝。

      输出的维度由ReshapeParam proto控制。
      可以直接使用数字进行指定。设定输入的某一维到输出blob中去。
      此外，还有两个数字值得说一下：

      0 直接从底层复制。例如，如果是底层是一个2 在它的第一维，那么顶层在它的第一维也有一个2。

      -1 从其他的数据里面推测这一维应该是多少。

### 2.6.5 链接层 Concatenation 通道扩展链接
    类型：Concat

    例子：
    layer {
      name: "concat"
      bottom: "in1"
      bottom: "in2"
      top: "out"
      type: "Concat"
      concat_param {
        axis: 1
      }
    }
    
####  2.6.5.1 说明
      可选参数：
      ·axis [default 1]：0代表链接num，1代表链接channels

      通过全连接层后的大小变化：

      输入：
            从1到K的每一个blob的大小：ni×ci×h×w  ni个   ci为通道数量

      输出：
            如果axis = 0: (n1+n2+...+nK)×c1×h×w，需要保证所有输入的ci相同。

            如果axis = 1: n1×(c1+c2+...+cK)×h×w，需要保证所有输入的n_i 相同。

      通过Concatenation层，可以把多个的blobs链接成一个blob。

### 2.6.6 Slicing
      类型：Slice

      例子：
      layer {
        name: "slicer_label"
        type: "Slice"
        bottom: "label"
        ## Example of label with a shape N x 3 x 1 x 1
        top: "label1"
        top: "label2"
        top: "label3"
        slice_param {
          axis: 1
          slice_point: 1
          slice_point: 2
        }
      }
      Slice层可以将输入层变成多个输出层。
      这些输出层沿一个给定的维度存在。
      axis指定了目标的轴，slice_point则指定了选择维度的序号。

### 2.6.7   Elementwise Operations

### 2.6.8 Argmax

### 2.6.9 Softmax

### 2.6.10 Mean-Variance Normalization

## 3  以上为层初始化  还有 数据前向传播 以及 误差反向传播
      每种类型的layer需要定义三种关键操作LayerSetUp, Forward, Backward：

      LayerSetUp: 网络构建时初始化层和层的连接
      Forward:    网络数据前向传递，给定bottom输入数据，计算输出到top
      Backward：  网络误差反向传递，给定top的梯度，计算bottom的梯度并存储到bottom blob

      Layer的设计主要就是SetUp、Forward、Backward函数（层一开始的时候的设置、然后就是前传和反传）

      这其中的SetUp的实现又依赖于CheckBlobCounts、LayerSetUp、Reshape等的实现。
                 这其中Reshape又是必须要实现的，因为它是纯虚函数
      这其中的Forward中又依赖于Forward_cpu、Forward_gpu，
                 这其中Forward_cpu又是必须要实现的。
      这其中的Backward中又依赖于Backward_cpu、Backward_gpu，
                 这其中Backward_cpu 又是必须要实现的。

      首先layer必须要实现一个forward function，前递函数当然功能可以自己定义啦，
      在forward中呢他会从input也就是Layer的bottom，
      对了caffe里面网络的前一层是叫bottom的，从bottom中获取blob，并且计算输出的Blob，
      前向传播：
           al+1 = 激活函数(W * al + bl+1)
           a为神经元激活后的输出
      当然他们也会实现一个反向传播backward function，
      根据他们的input的blob以及output blob的 error gradient 梯度误差 计算得到该层的梯度误差。

      反向传播：
             gl = 激活函数导数(zl) * W 转置 * gl+1
             g为 损失函数对z的偏导数
             z = W*a + b

[可参考台大反向传播视频](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/Lecture/DNN%20backprop.ecm.mp4/index.html)

### layer.hpp 源文件

    #ifndef CAFFE_LAYER_H_  
    #define CAFFE_LAYER_H_  

    #include <algorithm>  
    #include <string>  
    #include <vector>  

    #include "caffe/blob.hpp"  //数据
    #include "caffe/common.hpp"  //通用
    #include "caffe/layer_factory.hpp"  
    #include "caffe/proto/caffe.pb.h"  
    #include "caffe/util/device_alternate.hpp"  
    namespace caffe {  //caffe命名空间

    template <typename Dtype>// 通用 数据类型模板 Dtype 

    class Layer {  
     public:  
    /* 
    首先获得当前网络的Phase 模式，是 训练train还是 测试test，
    在初始化 列表初始化LayerParameter,之后blobs_这里存放的是一个指向blob类的shared_ptr指针
    的一个vector，在这里是申请空间，然后将传入的layer_param中的blob拷贝过来。 
    */ 
    // 显示的构造函数==========【1】==========================
    // 显示的构造函数不需要重写，任何初始工作在SetUp()中完成  
    // 构造方法只复制层参数说明的值，如果层说明参数中提供了权值和偏置参数，也复制
      explicit Layer(const LayerParameter& param)  
        : layer_param_(param) {  
          // Set phase and copy blobs (if there are any).  
    //1. 训练还是测试？phase 
          phase_ = param.phase();  
          if (layer_param_.blobs_size() > 0) {  
           //2. 将blobs_的大小设置为参数中的大小    
            blobs_.resize(layer_param_.blobs_size());  
            for (int i = 0; i < layer_param_.blobs_size(); ++i) {  
              // 3. 新建若干个Blob   
              blobs_[i].reset(new Blob<Dtype>());  
              // 4. 从blob文件中获取数据  
              blobs_[i]->FromProto(layer_param_.blobs(i));  
            }  
          }//用protobuf 传入的参数对blobs_ 做初始化，blobs_ 是一个vector 存放指向Blob类的智能指针。  

          #ifdef USE_MPI  // 
          //If this is a gather layer, all it subsequent layer doesn't need gradient sync.  
          //We will only change itself's property here,  
          //subsequent layers will be inferred in the Net  
            if (is_gathering()){  
                set_need_sync(false);  
              }else{  
                set_need_sync(true);  
              }  
          #endif  
        }  
      virtual ~Layer() {}//虚析构函数 需要重写

    //=====================================【2】==========================================    
    ////////////////初始化函数SetUp，每个Layer对象都必须遵循固定的调用模式/////////////////////
    // 实现每个layer对象的setup函数
    // 此方法非虚函数，不用重写，模式固定 
    void SetUp(const vector<Blob<Dtype>* >& bottom, // 输入层 输入数据  blob中的存储空间已申请
      const vector<Blob<Dtype>* >& top) {  // 输出层 输出数据，blob对象以构造但是其中的存储空间未申请
      // 具体空间大小需根据bottom blob大小和layer_param_共同决定，具体在Reshape函数现实 
    CheckBlobCounts(bottom, top); // 1. 检查输入输出blob个数是否满足要求，每个层能处理的输入输出数据不一样
    LayerSetUp(bottom, top);      // 2. 调用LayerSetUp函数初始化特殊的层，每个Layer子类需重写这个函数完成定制的初始化 
    Reshape(bottom, top);         // 3. 调用Reshape函数为top blob分配合适大小的存储空间
    SetLossWeights(top);          // 4. 为每个top blob设置损失权重乘子，非LossLayer 层的top blob其值为零
    } 

    //=====================================【3】==========================================
    /////////////////每个子类Layer必须重写的初始化函数LayerSetUp， 完成定制的初始化 
    //定制初始化，每个子类layer必须实现此虚函数
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, //输入blob, 数据成员data_和diff_存储了相关数据
      const vector<Blob<Dtype>*>& top)// 输出blob, blob对象已构造但数据成员的空间尚未申请

      {}  //此方法执行一次定制化的层初始化，包括从layer_param_读入并处理相关的层权值和偏置参数， 
          // 调用Reshape函数申请top blob的存储空间 

    //=====================================【4】==========================================
    /////////////////////每个子类Layer必须重写的Reshape函数，完成top blob形状的设置并为其分配存储空间，
    // 根据bottom blob的形状和layer_param_计算top blob的形状并为其分配存储空间
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
      const vector<Blob<Dtype>*>& top) = 0;

    //=====================================【5】==========================================
    //////////////前向传播函数 Forward    非虚函数 实际回调用虚函数  forward_cpu或者forward_gpu，
    // 首先是Forward.这其实是一个装饰器，
    // 继承之后在调用的调用其相应的forward_cpu或者forward_gpu，
    // 根据输入的input data blob计算相应的output data blob，
    // 同时会反应这一层layer的total loss. 
    inline Dtype Forward(const vector<Blob<Dtype>*>& bottom,  //输入blob
      const vector<Blob<Dtype>*>& top); // 输出blob

    //=====================================【5】==========================================
    ////////////// 反向传播函数Backward  
    //给定top blob 的 error gradient 误差梯度 计算得到bottom 的 误差梯度 error gradient。
    // 在Ouput blobs里面的diff存储的就是其相应的error gradients。
    // 其中propagate_down这个参数跟Bottom的长度是一样的
    // 其每一个Index用来指定是否需要反向传播error gradients 到对应的bottom blob。
    // 而bottom 这里面的diff 区域存放的就是BackWard计算出来相应的gradient error. 
    inline void Backward( const vector<Blob<Dtype>* >& top, // 输入是 output blobs，top blobs
      const vector<bool>& propagate_down,                 // 更新标志
      const vector<Blob<Dtype>* >& bottom);               // 输出是 bottom blobs

    //=====================================【6】==========================================

    // 返回数据===========================
    vector<shared_ptr<Blob<Dtype> > >& blobs() {  
    return blobs_;//返回vector  blobs_  
    }       

    //返回layer parameter  ===============
    const LayerParameter& layer_param() const { return layer_param_; }  

    //将layer plarameter 写入模型配置文件 protobuf ============= 
    virtual void ToProto(LayerParameter* param, bool write_diff = false);  

    //返回 一个blob top 在给定 index 的 loss ============
    inline Dtype loss(const int top_index) const {  
    return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);  
    }  

    // 设置一个blob top 在给定 index 的 loss============= 
    inline void set_loss(const int top_index, const Dtype value) {  
    if (loss_.size() <= top_index) {  
      loss_.resize(top_index + 1, Dtype(0));  
    }  
    loss_[top_index] = value;  
    } 

    // 虚函数，而且还是内联的，返回层类型 ================= 
    virtual inline const char* type() const { return ""; } 
    // 虚函数，获得bottom blob的精确个数 ==================   
    virtual inline int ExactNumBottomBlobs() const { return -1; }    
    // 虚函数，获得bottom blob的最小个数 ================   
    virtual inline int MinBottomBlobs() const { return -1; }    

    // 虚函数，获得bottom blob的最大个数 ============ 
    virtual inline int MaxBottomBlobs() const { return -1; }    

    // 虚函数，获得top blob的精确个数   =================== 
    virtual inline int ExactNumTopBlobs() const { return -1; }    

    // 虚函数，获得top blob的最小个数  ================  
    virtual inline int MinTopBlobs() const { return -1; }    

    // 虚函数，获得top blob的最大个数    ====================
    virtual inline int MaxTopBlobs() const { return -1; }    

    // 虚函数，bottom blob和top blob的个数是否一致 ========================   
    virtual inline bool EqualNumBottomTopBlobs() const { return false; }

    // 返回当前层是否自动创建匿名top blobs =====================   
    // 如果返回true，表明网络初始化的时候创建了了足够多的匿名top blobs    
    // 来满足ExactNumTopBlobs或者MinTopBlobs所要求的top blobs的个数    
    virtual inline bool AutoTopBlobs() const { return false; }  

    // AllowforceBackward用来设置是否强制梯度返回================
    // 因为有些层其实不需要梯度信息 ，后面两个函数分别查看以及设置是是否需要计算梯度。 
    // 对于一个给定的bottom blob，返回是否允许强制反传    
    virtual inline bool AllowForceBackward(const int bottom_index) const {    
    return true;    
    }

    // 设置  哪些 bottom 需要反向传播========
    inline bool param_propagate_down(const int param_id) {  
    return (param_propagate_down_.size() > param_id) ?  
        param_propagate_down_[param_id] : false;  
    }  
    inline void set_param_propagate_down(const int param_id, const bool value) {  
    if (param_propagate_down_.size() <= param_id) {  
      param_propagate_down_.resize(param_id + 1, true);  
    }  
    param_propagate_down_[param_id] = value;  
    } 
    #ifdef USE_MPI  
    // 多线程  并行 配置
    inline virtual bool is_gathering() {return false;}
    inline virtual bool is_scattering() {return false;}  
    inline bool need_sync(){return need_sync_;}  
    inline void set_need_sync(bool val){need_sync_ = val;}  
    #endif

    protected:  
    //The protobuf that stores the layer parameters
    // 层说明参数，从protocal buffers格式的网络结构说明文件中读取  
    LayerParameter layer_param_;  
    // The phase: TRAIN or TEST 
    // 层状态，参与网络的训练还是测试  
    Phase phase_;  
    // The vector that stores the learnable parameters as a set of blobs.
    // 层权值和偏置参数，使用向量是因为权值参数和偏置是分开保存在两个blob中的  
    vector<shared_ptr<Blob<Dtype> > > blobs_;  
    // Vector indicating whether to compute the diff of each param blob 
    // 标志每个top blob是否需要计算反向传递的梯度值 
    // 参数是否需要更新
    vector<bool> param_propagate_down_;

    // 非LossLayer为零，LossLayer中表示每个top blob计算的loss的权重  
    vector<Dtype> loss_;  

    #ifdef USE_MPI  
    bool need_sync_;  // 并行
    #endif

    //=====================================【7】==========================================  
    ///////前向传播函数 Forward 这个非虚函数，它们内部会调用如下虚函数完成数据前向传
    // 前向传播 CPU版本
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top) = 0;  
    // 前向传播 GPU版本
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top) {  
    // LOG(WARNING) << "Using CPU code as backup.";  
    return Forward_cpu(bottom, top);  
    } 
    //=====================================【8】==========================================  
    //////反向传播函数Backward 这个非虚函数，它们内部会调用如下虚函数完成 误差反向传播
    // 误差反向传播，根据执行环境的不同每个子类Layer必须重写CPU和GPU版本
    // CPU版本
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,  // 输入误差 top Blob
      const vector<bool>& propagate_down,                     // 更新标志
      const vector<Blob<Dtype>*>& bottom) = 0;                // 输出 bottom Blob
    // GPU版本
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,  
      const vector<bool>& propagate_down,  
      const vector<Blob<Dtype>*>& bottom) {  
    // LOG(WARNING) << "Using CPU code as backup.";  
    Backward_cpu(top, propagate_down, bottom);  
    } 

    //=====================================【9】========================================== 
    //  1. 检查输入输出blob个数是否满足要求，每个层能处理的输入输出数据不一样
    virtual void CheckBlobCounts(const vector<Blob<Dtype>*>& bottom, // 输入层
                               const vector<Blob<Dtype>*>& top) {  // 输出层
    if (ExactNumBottomBlobs() >= 0) {   
      CHECK_EQ(ExactNumBottomBlobs(), bottom.size())  
          << type() << " Layer takes " << ExactNumBottomBlobs()  
          << " bottom blob(s) as input.";  
    }// 保证输入bottom 数量和要求的相同

    if (MinBottomBlobs() >= 0) {  
      CHECK_LE(MinBottomBlobs(), bottom.size())  
          << type() << " Layer takes at least " << MinBottomBlobs()  
          << " bottom blob(s) as input.";  
    }//保证输入的bottom数量大于或等于 要求的最小数量  

    if (MaxBottomBlobs() >= 0) {  
      CHECK_GE(MaxBottomBlobs(), bottom.size())  
          << type() << " Layer takes at most " << MaxBottomBlobs()  
          << " bottom blob(s) as input.";  
    }//保证输入的bottom数量小于或等于 要求的最大数量 

    if (ExactNumTopBlobs() >= 0) {  
      CHECK_EQ(ExactNumTopBlobs(), top.size())  
          << type() << " Layer produces " << ExactNumTopBlobs()  
          << " top blob(s) as output.";  
    }// 保证输入top数量和要求的相同  

    if (MinTopBlobs() >= 0) {  
      CHECK_LE(MinTopBlobs(), top.size())  
          << type() << " Layer produces at least " << MinTopBlobs()  
          << " top blob(s) as output.";  
    }//保证输入的top数量大于或等于 要求的最小数量  

    if (MaxTopBlobs() >= 0) {  
      CHECK_GE(MaxTopBlobs(), top.size())  
          << type() << " Layer produces at most " << MaxTopBlobs()  
          << " top blob(s) as output.";  
    }//保证输入的top数量小于或等于 要求的最大数量 

    if (EqualNumBottomTopBlobs()) {  
      CHECK_EQ(bottom.size(), top.size())  
          << type() << " Layer produces one top blob as output for each "  
          << "bottom blob input.";  
    }//保证输入的bottom数量 和 输出的top数量相同  
    } 

    //=====================================【10】==========================================
    // 4. 为每个top blob设置损失权重乘子，非LossLayer 层的top blob其值为零
    // SetLoss是非常重要的一个步骤，是被SetUp调用来初始化top bottom的weights，
    // 并且存储非零的loss weights 在diff blob里面 
    inline void SetLossWeights(const vector<Blob<Dtype>*>& top) {  
    const int num_loss_weights = layer_param_.loss_weight_size();  
    if (num_loss_weights) {  
      CHECK_EQ(top.size(), num_loss_weights) << "loss_weight must be "  
          "unspecified or specified once per top blob.";  
      for (int top_id = 0; top_id < top.size(); ++top_id) {  
        const Dtype loss_weight = layer_param_.loss_weight(top_id);  
        if (loss_weight == Dtype(0)) { continue; }//如果为0不对loss进行操作  
        this->set_loss(top_id, loss_weight);  
        const int count = top[top_id]->count();  
        Dtype* loss_multiplier = top[top_id]->mutable_cpu_diff();  
        caffe_set(count, loss_weight, loss_multiplier);//将loss_multiplier设为loss_weight  
      }   
    }  
    }    
    DISABLE_COPY_AND_ASSIGN(Layer);  
    };  // class Layer  

    //=====================================【11】==========================================
    // 网络前向传播 调用接口 逻辑函数========
    // 传播调用对应的Forward_cpu或者Forward_gpu
    // 而我们知道Forward_cpu是纯虚函数，必须要实例化，
    // 而Forward_gpu是虚函数，如果不实现，就会调用 Forward_cpu函数了。
    // 所以前传（你必须实现自己的Forward_cpu，实现Forward_gpu是可选的） 
    template <typename Dtype>  
    inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,// 输入 层 
    const vector<Blob<Dtype>*>& top) {  // 输出 层
    Dtype loss = 0;   
    // 根据bottom设置top的形状    
    Reshape(bottom, top);  // 申请 内存 
    // 设置运行模式CPU or GPU    
    switch (Caffe::mode()) {    
      case Caffe::CPU:    
          // 调用CPU的前传    
        Forward_cpu(bottom, top);    
          // 前传计算完之后计算损失（只有最后一层才进行计算，其余层都不用）    
        for (int top_id = 0; top_id < top.size(); ++top_id) {    
          if (!this->loss(top_id)) { continue; }    
          const int count = top[top_id]->count();    
            // 获取前传的数据    
          const Dtype* data = top[top_id]->cpu_data();    
            // 获取梯度（\frac{\partial Loss}{\partial net}）    
          const Dtype* loss_weights = top[top_id]->cpu_diff();    
            // data与loss_weight的点积，即得损失函数关于当前层权重的偏导了    
        // \frac{\partial Loss}{\partial net} * \frac{\partial net}{\frac{W}}    
        // = \frac{\partial Loss}{\partial W}    
          loss += caffe_cpu_dot(count, data, loss_weights);// cpu 计算点积
        }   
        break;    
      case Caffe::GPU:    
        // GPU前传    
        Forward_gpu(bottom, top);    
       #ifndef CPU_ONLY    
        // 同上，只不过这里用GPU来计算点积了    
        for (int top_id = 0; top_id < top.size(); ++top_id) {    
          if (!this->loss(top_id)) { continue; }    
          const int count = top[top_id]->count();    
          // 获取GPU上的数据    
          const Dtype* data = top[top_id]->gpu_data();    
          const Dtype* loss_weights = top[top_id]->gpu_diff();    
          Dtype blob_loss = 0;    
          caffe_gpu_dot(count, data, loss_weights, &blob_loss); // gpu 计算点积   
          loss += blob_loss;    
        }    
       #endif    
        break;  
      default:  
        LOG(FATAL) << "Unknown caffe mode.";  
    }  
    return loss;  
    }  

    //=====================================【12】==========================================
    // 网络 反向传播 调用接口 逻辑函数========
    template <typename Dtype>  
    inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,  // 输入 误差 从后向前传播
    const vector<bool>& propagate_down,   // 更新标志 
    const vector<Blob<Dtype>*>& bottom) { // 误差输出
    switch (Caffe::mode()) {   
      case Caffe::CPU:  
        Backward_cpu(top, propagate_down, bottom);  
        //根据blob top 的error 梯度（diff）计算bottom 的 error 梯度。 propagate_down 是长度   
        //和bottom 相同的vector ，用于控制是否需要对对应的bottom 元素传播梯度。具体layer具体定义。  
        break;  
      case Caffe::GPU:  
        Backward_gpu(top, propagate_down, bottom);  
        break;  
      default:  
        LOG(FATAL) << "Unknown caffe mode.";  
    }  
    } 

    //=====================================【12】==========================================
    //Layer的序列化函数,将layer的层说明参数layer_param_，层权值和偏置  
    //参数blobs_复制到LayerParameter对象，便于写到磁盘，  
    // Serialize LayerParameter to protocol buffer  
    template <typename Dtype>  
    void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) {  
    param->Clear();  
    param->CopyFrom(layer_param_); // 复制层说明参数layer_param_  
    param->clear_blobs();  
    // 复制层权值和偏置参数blobs_  
    for (int i = 0; i < blobs_.size(); ++i) {  
    blobs_[i]->ToProto(param->add_blobs(), write_diff);  
    }  
    }  

    }  // namespace caffe  

    #endif  // CAFFE_LAYER_H_  


