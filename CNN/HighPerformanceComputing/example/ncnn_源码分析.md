# 1. param 和 bin 文件分析
## param
      7767517   # 文件头 魔数
      75 83     # 层数量  输入输出blob数量
                # 下面有75行
      Input            data             0 1 data 0=227 1=227 2=3
      Convolution      conv1            1 1 data conv1 0=64 1=3 2=1 3=2 4=0 5=1 6=1728
      ReLU             relu_conv1       1 1 conv1 conv1_relu_conv1 0=0.000000
      Pooling          pool1            1 1 conv1_relu_conv1 pool1 0=0 1=3 2=2 3=0 4=0
      Convolution      fire2/squeeze1x1 1 1 pool1 fire2/squeeze1x1 0=16 1=1 2=1 3=1 4=0 5=1 6=1024
      ...
      层类型            层名字   输入blob数量 输出blob数量  输入blob名字 输出blob名字   参数字典
      
      参数字典，每一层的意义不一样：
      数据输入层 Input            data             0 1 data 0=227 1=227 2=3   图像宽度×图像高度×通道数量
      卷积层    Convolution  ...   0=64     1=3      2=1    3=2     4=0    5=1    6=1728           
               0输出通道数 num_output() ; 1卷积核尺寸 kernel_size();  2空洞卷积参数 dilation(); 3卷积步长 stride(); 
               4卷积填充pad_size();       5卷积偏置有无bias_term();   6卷积核参数数量 weight_blob.data_size()；
                                                                  C_OUT * C_in * W_h * W_w = 64*3*3*3 = 1728
      池化层    Pooling      0=0       1=3       2=2        3=0       4=0
                          0池化方式:最大值、均值、随机     1池化核大小 kernel_size();     2池化核步长 stride(); 
                          3池化核填充 pad();   4是否为全局池化 global_pooling();
      激活层    ReLU       0=0.000000     下限阈值 negative_slope();
               ReLU6      0=0.000000     1=6.000000 上下限
      
      综合示例：
      0=1 1=2.5 -23303=2,2.0,3.0
      
      数组关键字 : -23300 减去 0 ~ 19  例如 -23303 表示3个元素的数组
[各层详细表格](https://github.com/Tencent/ncnn/wiki/operation-param-weight-table)
      
## bin

        +---------+---------+---------+---------+---------+---------+
        | weight1 | weight2 | weight3 | weight4 | ....... | weightN |
        +---------+---------+---------+---------+---------+---------+
        ^         ^         ^         ^
        0x0      0x80      0x140     0x1C0

      所有权重数据连接起来, 每个权重占 32bit

      权重数据 weight buffer

      [flag] (optional 可选)
      [raw data]
      [padding] (optional 可选)

          flag : unsigned int, little-endian, indicating the weight storage type, 
                 0 => float32, 
                 0x01306B47 => float16, 
                 otherwise => quantized int8, 
                      may be omitted if the layer implementation forced the storage type explicitly。
          raw data : raw weight data, little-endian, 
                     float32 data or float16 data or quantized table 
                     and indexes depending on the storage type flag。
          padding : padding space for 32bit alignment, may be omitted if already aligned。



## 2. 轻模式
      开启轻模式省内存 set_light_mode(true)
      
      每个layer都会产生blob，除了最后的结果和多分支中间结果，大部分blob都不值得保留，
      开启轻模式可以在运算后自动回收，省下内存。
      
      举个例子：某网络结构为 A -> B -> C，在轻模式下，向ncnn索要C结果时，A结果会在运算B时自动回收，
      而B结果会在运算C时自动回收，最后只保留C结果，后面再需要C结果会直接获得，满足绝大部分深度网络的使用方式。
      
## 3. 网络和运算是分开的
      
      ncnn的net是网络模型，实际使用的是extractor，
      也就是同个net可以有很多个运算实例，而且运算实例互不影响，中间结果保留在extractor内部，
      在多线程使用时共用网络的结构和参数数据，初始化网络模型和参数只需要一遍.
      
      举个例子：全局静态的net实例，初始化一次后，就能不停地生成extractor使用.
      
## 4. 新建 一个新的层
      在 ncnn/src/layer/ 下新建两个文件 mylayer.h mylayer.cpp
      
> 步骤1： 新建空的类==================
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
      
> 步骤2： 定义层参数parameters 和 权重 weights
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

>  步骤3： 实现载入 参数和权重的函数function()
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

> 步骤4： 定义类构造函数，确定层 前向传播行为
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

> 步骤5： 根据行为 选择合适的 forward()函数接口
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

> 步骤6： 实现对应的 forward() 函数

```c


```

> 步骤7： 
```c


```
      
     
      
