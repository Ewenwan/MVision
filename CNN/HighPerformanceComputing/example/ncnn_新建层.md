
# 在ncnn中 新建 一个层
      在 ncnn/src/layer/ 下新建两个文件 mylayer.h mylayer.cpp
      
## 步骤1： 新建空的类==================
```c
      // 1.文件1 mylayer.h
      #include "layer.h"
      using namespace ncnn;
      
      // a new layer type called MyLayer
      class MyLayer : public Layer // 公有继承自 Layer 类
      {
      };
      
      // 2.文件2 mylayer.cpp
      #include "mylayer.h"
      DEFINE_LAYER_CREATOR(MyLayer) // 注册 新定义的层
```
      
## 步骤2： 定义层参数parameters 和 权重 weights
```c
// mylayer.h
#include "layer.h"
using namespace ncnn;

class MyLayer : public Layer
{

private: // 私有参数
      int channels; // 通道数量
      float gamma;  // 参数
      Mat weight;   // 权重
}；


// mylayer.cpp
#include "mylayer.h"
DEFINE_LAYER_CREATOR(MyLayer) // 注册 新定义的层

```

## 步骤3： 实现载入 参数和权重的函数function()
```c
// mylayer.h========================================
#include "layer.h"
using namespace ncnn;

class MyLayer : public Layer
{
public:  // 公有方法
      // 头文件中声明为 虚函数
      virtual int load_param(const ParamDic& pd);// 虚函数，可以被子类覆盖
      virtual int load_model(const ModelBin& mb);   // 

private: // 私有参数
      int channels;   // 参数1 通道数量
      float eps;      // 参数2 精度
      Mat gamma_data; // 权重
}；


// mylayer.cpp======================================
#include "mylayer.h"
DEFINE_LAYER_CREATOR(MyLayer)

// 实现 load_param() 载入网络层参数====
int MyLayer::load_param(const ParamDict& pd)
{
      // 使用pd.get(key,default_val); 从param文件中（key=val）获取参数
      channels = pd.get(0, 0);      // 解析 0=<int value>, 默认为0
      eps      = pd.get(1, 0.001f); // 解析 1=<float value>, 默认为0.001f
      
      return 0; // 载入成功返回0
}

// 实现 load_model() 载入模型权重===
int MyLayer::load_model(const ModelBin& mb)
{
      // 读取二进制数据的长度为 channels * sizeof(float)
      // 0 自动判断数据类型， float32 float16 int8
      // 1 按 float32读取  2 按float16读取 3 按int8读取
      gamma_data = mb.load(channels, 1);// 按 float32读取 
      if(gamma_data.empty())
            return -100; // 错误返回非0数，-100表示 out-of-memory 
      return 0; //  载入成功返回0
}


```

## 步骤4： 定义类构造函数，确定层 前向传播行为
```c
// mylayer.h========================================
#include "layer.h"
using namespace ncnn;

class MyLayer : public Layer
{
public:  // 公有方法
      // 定义构造函数 ====new code
      MyLayer():
      // 头文件中声明为 虚函数
      virtual int load_param(const ParamDic& pd);// 虚函数，可以被子类覆盖
      virtual int load_model(const ModelBin& mb);   // 

private: // 私有参数
      int channels;   // 参数1 通道数量
      float eps;      // 参数2 精度
      Mat gamma_data; // 权重
}；


// mylayer.cpp======================================
#include "mylayer.h"
DEFINE_LAYER_CREATOR(MyLayer)

// 实现构造函数 new code=====
MyLayer::MyLayer()
{
      // 是否为 1输入1输出层
      // 1输入1输出层： Convolution, Pooling, ReLU, Softmax ...
      // 反例       ： Eltwise, Split, Concat, Slice ...
      one_blob_only = true;

      // 是否可以在 输入blob 上直接修改 后输出
      // 支持在原位置上修改： Relu、BN、scale、Sigmod...
      // 不支持： Convolution、Pooling ...
      support_inplace = true;
}


// 实现 load_param() 载入网络层参数====
int MyLayer::load_param(const ParamDict& pd)
{
      // 使用pd.get(key,default_val); 从param文件中（key=val）获取参数
      channels = pd.get(0, 0);      // 解析 0=<int value>, 默认为0
      eps      = pd.get(1, 0.001f); // 解析 1=<float value>, 默认为0.001f
      
    // 可以通过 载入的参数 来修改层的行为====
    // if (eps == 0.001f)
    // {
    //     one_blob_only = false;
    //     support_inplace = false;
    // }
      
      return 0; // 载入成功返回0
}

// 实现 load_model() 载入模型权重===
int MyLayer::load_model(const ModelBin& mb)
{
      // 读取二进制数据的长度为 channels * sizeof(float)
      // 0 自动判断数据类型， float32 float16 int8
      // 1 按 float32读取  2 按float16读取 3 按int8读取
      gamma_data = mb.load(channels, 1);// 按 float32读取 
      if(gamma_data.empty())
            return -100; // 错误返回非0数，-100表示 out-of-memory 
            
    // 可以通过 载入的权重 来修改层的行为====
    // if (gamma_data[0] == 0.f)
    // {
    //     one_blob_only = false;
    //     support_inplace = false;
    // }
      
            
      return 0; //  载入成功返回0
}


```

## 步骤5： 根据行为 选择合适的 forward()函数接口
```c
Layer类定义了四种 forward()函数：
// 1 ：多输入多输出，const 不可直接对输入进行修改
virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

// 2 : 单输入单输出，const 不可直接对输入进行修改
virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

// 3 ：多输入多输出，可直接对输入进行修改
virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;

// 4 ：单输入单输出，可直接对输入进行修改
virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

// one_blob_only   support_inplace   函数类别： 1       2      3      4
// false               false                 must  
// false               true                 optional        must 
// true                false                         must 
// true                true                         optional       must


```

## 步骤6： 实现对应的 forward() 函数

```c
// mylayer.h========================================
#include "layer.h"
using namespace ncnn;

class MyLayer : public Layer
{
public:  // 公有方法
      // 定义构造函数 ====new code
      MyLayer():
      // 头文件中声明为 虚函数
      virtual int load_param(const ParamDic& pd);// 虚函数，可以被子类覆盖
      virtual int load_model(const ModelBin& mb);   // 
      
      // 网络运行===new code
      virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;// 单输入单输出
      virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;         // 单入单出本地修改 

private: // 私有参数
      int channels;   // 参数1 通道数量
      float eps;      // 参数2 精度
      Mat gamma_data; // 权重
}；


// mylayer.cpp======================================
#include "mylayer.h"
DEFINE_LAYER_CREATOR(MyLayer)

// 实现构造函数 new code=====
MyLayer::MyLayer()
{
      // 是否为 1输入1输出层
      // 1输入1输出层： Convolution, Pooling, ReLU, Softmax ...
      // 反例       ： Eltwise, Split, Concat, Slice ...
      one_blob_only = true;

      // 是否可以在 输入blob 上直接修改 后输出
      // 支持在原位置上修改： Relu、BN、scale、Sigmod...
      // 不支持： Convolution、Pooling ...
      support_inplace = true;
}


// 实现 load_param() 载入网络层参数====
int MyLayer::load_param(const ParamDict& pd)
{
      // 使用pd.get(key,default_val); 从param文件中（key=val）获取参数
      channels = pd.get(0, 0);      // 解析 0=<int value>, 默认为0
      eps      = pd.get(1, 0.001f); // 解析 1=<float value>, 默认为0.001f
      
    // 可以通过 载入的参数 来修改层的行为====
    // if (eps == 0.001f)
    // {
    //     one_blob_only = false;
    //     support_inplace = false;
    // }
      
      return 0; // 载入成功返回0
}

// 实现 load_model() 载入模型权重===
int MyLayer::load_model(const ModelBin& mb)
{
      // 读取二进制数据的长度为 channels * sizeof(float)
      // 0 自动判断数据类型， float32 float16 int8
      // 1 按 float32读取  2 按float16读取 3 按int8读取
      gamma_data = mb.load(channels, 1);// 按 float32读取 
      if(gamma_data.empty())
            return -100; // 错误返回非0数，-100表示 out-of-memory 
            
    // 可以通过 载入的权重 来修改层的行为====
    // if (gamma_data[0] == 0.f)
    // {
    //     one_blob_only = false;
    //     support_inplace = false;
    // }      
      return 0; //  载入成功返回0
}


// 单入单出 前向传播网络 不可修改 非const
int MyLayer::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
      // 检查实际输入的blob的 维度是否和 所给的网络参数 一致
      if(bottom_blob.c != channels)
            return -1; // 返回非0，表示错误
      
      // 实现运算 x = (x + eps) * gamma_per_channel
      
      int w = bottom_blob.w;
      int h = bottom_blob.h;
      size_t elemsize = bottom_blob.elemsize;// 输入blob参数数量， 用来申请内存
      int size = w * h;// 单channel中参数数量
      
      // 输出需要新建，不可直接在输入blob上修改
      top_blob.create(w, h, channels, elemsize, opt.blob_allocator);
      if(top_blob.empty())
            return -100; // 返回-100, 表示内存溢出，申请空间出错
      
      // 使用openmp 并行计算
      #pragam omp parallel for num_threads(opt.num_threads);
      for (int q=0; q<channels; q++)// 遍历每个通道channel
      {
            const float* in_ptr = bottom_blob.channel(q);// 输入数据 const不可变
            float* out_ptr = top_blob.channel(q);// 输出数据
            const float gamma = gamma_data[q];   // 运算参数 每个channel有一个
            
            // 遍历一个map的数据
            for (int i=0; i<size; i++)
            {
                  out_ptr[i] = (in_ptr[i] + eps)*gamma;// 运算结果，保存在输出blob中
            }
      
      }
      
      return 0；// 无错误 返回0
}


// 单入单出 前向传播网络  可在输入blob上修改
int MyLayer::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
      // 检查实际输入的blob的 维度是否和 所给的网络参数 一致
      if(bottom_blob.c != channels)
            return -1; // 返回非0，表示错误
      
      // 实现运算 x = (x + eps) * gamma_per_channel
      
      int w = bottom_blob.w;
      int h = bottom_blob.h;
      int size = w * h;// 单channel中参数数量
      
      // 输出不需要新建，可直接在输入blob上修改

      // 使用openmp 并行计算
      #pragam omp parallel for num_threads(opt.num_threads);
      for (int q=0; q<channels; q++)// 遍历每个通道channel
      {
            float* in_out_ptr = bottom_top_blob.channel(q);// 输入数据 需要可变
            const float gamma = gamma_data[q];   // 运算参数 每个channel有一个
            // 遍历一个map的数据
            for (int i=0; i<size; i++)
            {
                  in_out_ptr[i] = (in_out_ptr[i] + eps)*gamma;// 直接在输入数据上修改
            }
      
      }
      
      return 0；// 无错误 返回0
}




```

# 步骤7： 集和进ncnn库
```c
// 1. 修改  caffe2ncnn.cpp \ mxnet2ncnn.cpp net.cpp/net.h(???) 模型转换
// 待具体分析=====
// example param file content
// Input            input   0 1 input
// Convolution      conv2d  1 1 input conv2d 0=32 1=1 2=1 3=1 4=0 5=0 6=768
// MyLayer          mylayer 1 1 conv2d mylayer0
// Pooling          maxpool 1 1 mylayer0 maxpool 0=0 1=3 2=2 3=-233 4=0


// 2. app应用中添加，新层注册宏
// app 应用=====

ncnn::Net net;

// register custom layer before load param and model
// the layer creator function signature is always XYZ_layer_creator, which defined in DEFINE_LAYER_CREATOR macro
net.register_custom_layer("MyLayer", MyLayer_layer_creator);// 注册新层

net.load_param("model.param");
net.load_model("model.bin");

```
      
     
      
