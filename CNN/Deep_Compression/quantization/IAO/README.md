
## 1. Introduction  
[论文](https://arxiv.org/pdf/1712.05877.pdf)

[tensflow 代码](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/quantize)

[量化细节](https://github.com/google/gemmlowp/blob/master/doc/quantization.md)

[量化示例代码](https://github.com/google/gemmlowp/blob/master/doc/quantization_example.cc)

    这篇文章提出了一种量化神经网络到INT8的通用解决方案，包括量化后精度损失，怎么通过训练来挽救精度，干货满满,  
    同时还提供了源代码，包含优化过的inference代码。    
    paper中有些地方说的比较模糊，参考文献[1]讲的就比较清楚了，  
    作者们甚至实现了一份未优化的代码来帮助大家理解，实在是太良心了。    

    文章在第一部分也指出了目前很多压缩和加速算法在AlexNet, VGG等大网络上实验，  
    这些网络为了追求更高的精度，本身就有很多冗余，压缩率当然高。    
    同时类似于XNOR， BWN， TWN等算法只给出了理论加速，
    但是由于这些算法需要特定的硬件支持，所以并不能广泛推广。    

    这篇文章主要的贡献有三点：    
    > - 提出了一种通用的量化方案，同时量化weight和activation  
    > - 提出了弥补量化后精度损失的训练方案  
    > - 在MobileNet这种本身就很紧凑的网络上做实验以证明其有效性  
    
## 谷歌IAO算法实现细节
	对量化的实现是通过把常见操作转换为等价的八位版本达到的。
	涉及的操作包括卷积，矩阵乘法，激活函数，池化操作，以及拼接。
	转换脚本先把每个已知的操作替换为等价的量化版本。
	然后在操作的前后加上含有转换函数的子图，将input从浮点数转换成8 bit，
	再把output从8 bit转回浮点数。下面是 ReLu 的例子：
	
浮点版本 relu层:
![](http://fjdu.github.io/pictures/2016-07-07-quantization0.png)

量化版本 relu层：
![](http://fjdu.github.io/pictures/2016-07-07-quantization1.png)


```c
a. 记录各层 激活输入、卷积核参数、激活输出的参数范围 max min,而量化范围为0~255 uint_8

b. 计算量化参数，缩放尺度S  和  零点(偏移量) Z
   float S     = (max-min）/ (255-0);
   uint8_t Z = round(0 – min/S) ;

c. 量化输入in_ 和 卷积参数 w_
   in_quan = in_/S_in + Z_in;
    w_quan = w_/S_w  + Z_w;

d. 浮点矩阵乘法 变成 量化卷积乘法
    out_ = sum( in_(i) * w_(i) );// 浮点矩阵乘法
         = (S_in * S_W) * sum(  (in_quan-Z_in) * (w_quan – Z_w) );
    ==> out_quan = out_/S_out  + Z_out;
    ==>          = 
       (S_in * S_W/ S_out) * sum(  (in_quan-Z_in) * (w_quan – Z_w) ) + Z_out
       式中   (S_in * S_W/ S_out) 仍为浮点数，所以也需要转换成整数
       浮点乘子 real_multiplier = S_in * S_W / S_out;
       量化乘子 quantized_multiplier = round(real_multiplier * (1 << 31)); 扩大2^31次方变成32位整数

e.  之后再将整数结果转换成 浮点结果 用于后续计算
     out_ =(out_quan  -Z_out) * S_out;
```
## 2. Quantization Inference  

### Inference  
    量化就是把float型的数值映射到int8或者uint8来进行卷积运算  

![formula1](https://github.com/Ewenwan/camel007.github.io/blob/master/img/2018-06-10/formula1.jpg)

    r - 需要量化的float型数值  
    q - 量化后的uint8类型数值  
    Z - 量化前r = 0时，量化后q的数值  
    S - 为了能把量化后的q还原到r, 引入了一个缩放系数  
    
    例如一直input的最大值是30.0，最小值是-10.0，则量化后的值为
    Quantized | Float
    --------- | -----
    0         | -10.0
    255       | 30.0
    128       | 10.0
    
    如何把float类型的乘法用int8替代，paper中的公式写的很明白，  
    输入： r1 = S1 * (q1 - Z1)
          r2 = S2 * (q2 - Z2)
    乘法： out = r1 * r2 = S1 * (q1 - Z1) * S2 * (q2 - Z2)  
          q_out = out / S3 + Z3 
                =  S1 * S2 / S3 * (q1 - Z1) * (q2 - Z2)  + Z3
                = M * (q1 - Z1) * (q2 - Z2)  + Z3   公式(4)
          乘子 M = S1 * S2 / S3
          
    如何把float类型的乘法用int8替代，paper中的公式写的很明白，  
    这我们重点说一下paper中公式(5)，是怎么转化为int8来计算的    
![formula5](https://github.com/Ewenwan/camel007.github.io/blob/master/img/2018-06-10/formula5.jpg)  

    本来公式中全部都是int8类型了，只有的个M仍然是float类型的，  
    但是据经验值M是一个大于0小于1的数值，于是我们对M做一些小操作：    
![formula6](https://github.com/Ewenwan/camel007.github.io/blob/master/img/2018-06-10/formula6.jpg)  

    公式6中，令M0在[0.5, 1)范围内  
    举个例子， M = 0.3， 那么M0 = 0.3 * 2， 于是n = 1，然后我们把M0量化到整形数，  
    具体是16位还是32根据机器决定，以32位为例，M0 = M0 * (1 << 31>>),  
    取整后M0就是一个32位的整型数，此时n = 32，  
    因此公式(4)中加号后半部分全部为整型乘法和移位操作(   
    这里M0和另外一部分都为32位的整型，其乘积结果应该是64位的整型)  

### Quantization  

**step 1**  

![formula7](https://github.com/Ewenwan/camel007.github.io/blob/master/img/2018-06-10/formula7.jpg) 

max 和 min 为矩阵中最大最小元素的值  

**step 2**  

![formula8](https://github.com/Ewenwan/camel007.github.io/blob/master/img/2018-06-10/formula8.jpg)  

**step 3**  

![formula9](https://github.com/Ewenwan/camel007.github.io/blob/master/img/2018-06-10/formula9.jpg)  

### Training

    *大家可能还有另外一个疑惑，weight和input的max,max, S,Z这些都很好计算，  
    paper中S3,Z3这些参数在inference的时候是不知道的。  
    这一波操作精妙的地方在于一开始就假设r3也是int8的，  
    所以整型矩阵相乘后通过bit shift等操作，结果仍然是int8类型的，  
    直接进入下一次卷积操作，而不需要dequantize操作，  
    至于S3,Z3这些参数是在训练过程中计算出来的*

    激活值中的max, min都是在训练过程中使用EMA计算出来的，  
    作者还提到在训练刚开始不太稳定的时候不要对网络进行量化，  
    待稳定后再量化，可以尽快的使整个网络收敛。  

**WARNING: 本文仅仅是我在看paper中一些自认为比较关键的点和一些当时不太明白的地方的记录，如果要看懂整个论文还需要去看参考文献中的资料**


## 3. Reference  
1. [gemmlowp document](https://github.com/google/gemmlowp/tree/master/doc)  
2. [Quantization and Training of Neural Networks for Efficient
Integer-Arithmetic-Only Inference](https://arxiv.org/pdf/1712.05877.pdf)

# 量化代码
## 1. 找到浮点数的 数据范围 最大值和最小值
```c
// Find the min and max value in a float matrix.
template <gemmlowp::MapOrder tOrder>
void FindMinMax(const gemmlowp::MatrixMap<float, tOrder>& m, float* min,
                float* max) {
  *min = *max = m(0, 0);// 初始化最大最小值 为矩阵(0,0)处的值
  for (int i = 0; i < m.rows(); i++) {
    for (int j = 0; j < m.cols(); j++) {
      const float val = m(i, j);
      *min = std::min(*min, val);// 最小值
      *max = std::max(*max, val);// 最小值
    }
  }
}

```
## 2. 计算量化参数 尺度scale 偏移量zero
```c
struct QuantizationParams {
  float scale;            // 缩放尺度
  std::uint8_t zero_point;// 零点(偏移量)
};
// 根据 最大值和最小值计算量化参数
QuantizationParams ChooseQuantizationParams(float min, float max) {
  min = std::min(min, 0.f);// 确保最小值 <= 0
  max = std::max(max, 0.f);// 确保最大值 >= 0

  // the min and max quantized values, as floating-point values
  // uint8 量化的数据范围
  const float qmin = 0;
  const float qmax = 255;

  // 计算量化缩放尺度================================
  // 每一个量化数据单位代表的浮点数大小 0.0~510.0 量化到 0~255 则每一个量化数代表 2
  const double scale = (max - min) / (qmax - qmin);
  
  // 计算 零点(偏移量)===========================
  const double initial_zero_point = qmin - min / scale;

  // 对  零点(偏移量) 限幅 并取整=================
  std::uint8_t nudged_zero_point = 0;
  if (initial_zero_point < qmin) {
    nudged_zero_point = qmin;
  } else if (initial_zero_point > qmax) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point =
        static_cast<std::uint8_t>(std::round(initial_zero_point));
  }

  QuantizationParams result;
  result.scale = scale;                   // 量化缩放尺度  float
  result.zero_point = nudged_zero_point;  // 零点(偏移量)  uint8_t
  return result;
}
```
## 3. 量化 0.0-510.0 量化到  0-255
```c
void Quantize(const QuantizationParams& qparams, const std::vector<float>& src,
              std::vector<std::uint8_t>* dst) 
{
  assert(src.size() == dst->size());
  for (std::size_t i = 0; i < src.size(); i++) // 遍历每一个浮点数
  {
    const float real_val = src[i]; // 每一个浮点数
    const float transformed_val = qparams.zero_point + real_val / qparams.scale;
    const float clamped_val = std::max(0.f, std::min(255.f, transformed_val));// 限制幅度 0.0 ~ 255.0 之间
    (*dst)[i] = static_cast<std::uint8_t>(std::round(clamped_val));// 取整
  }
}

```
## 4. 反量化  0-255 反量化到 0.0-510.0 
```c
void Dequantize(const QuantizationParams& qparams,
                const std::vector<std::uint8_t>& src, std::vector<float>* dst) 
{
  assert(src.size() == dst->size());
  for (std::size_t i = 0; i < src.size(); i++) {// 遍历每一个 uint8
    const std::uint8_t quantized_val = src[i];
    (*dst)[i] = qparams.scale * (quantized_val - qparams.zero_point);// 变换到浮点数
  }
}


```
## 浮点数卷积运算 FloatMatMul()
```c
void FloatMatrixMultiplication(
    const gemmlowp::MatrixMap<const float, tLhsOrder>& lhs,
    const gemmlowp::MatrixMap<const float, tRhsOrder>& rhs,
    gemmlowp::MatrixMap<float, tResultOrder>* result) {
  assert(lhs.cols() == rhs.rows());
  assert(lhs.rows() == result->rows());
  assert(rhs.cols() == result->cols());
  for (int i = 0; i < lhs.rows(); i++) 
  {// 每行
    for (int k = 0; k < rhs.cols(); k++) 
    {// 每列
      (*result)(i, k) = 0;
      for (int j = 0; j < lhs.cols(); j++)
      {
      // lhs_quantized_val  = Quantize(lhs);// 上层feature map 输入
      // rhs_quantized_val  = Quantize(rhs);// 本层卷积核
       //  (*result)(i, k) += lhs(i, j) * rhs(j, k);// 卷积块内 求和 使用浮点数
       (*result)(i, k) += lhs_scale * rhs_scale * 
                          (lhs_quantized_val(i, j) - lhs_zero_point) * 
                          (rhs_quantized_val(j, k)-rhs_zero_point);
       // 使用量化数表示浮点数之后运算 得到浮点数结果
       // 而浮点数结果 也需要进行量化
       // result_real_value = result_scale *(result_quantized_value - result_zero_point)
       // result_quantized_value = result_zero_point + result_real_value / result_scale
      }
    }
  }
}

```
## 量化卷积运算   QuantizedMatMul()
[具体 使用 gemmlowp  低精度乘法计算框架 ](https://github.com/Ewenwan/gemmlowp)

```c
// ========================================
  for (int i = 0; i < lhs.rows(); i++) 
  {// 每行
    for (int k = 0; k < rhs.cols(); k++) 
    {// 每列
      (*result)(i, k) = 0;
      for (int j = 0; j < lhs.cols(); j++)
      {
          // lhs_quantized_val  = Quantize(lhs);// 上层feature map 输入
          // rhs_quantized_val  = Quantize(rhs);// 本层卷积核
           //  (*result)(i, k) += lhs(i, j) * rhs(j, k);// 卷积块内 求和 使用浮点数
           //(*result)(i, k) += lhs_scale * rhs_scale * 
           //                   (lhs_quantized_val(i, j) - lhs_zero_point) * 
           //                   (rhs_quantized_val(j, k)-rhs_zero_point);
// (*result)(i, k) += (lhs_quantized_val(i, j) - lhs_zero_point) * (rhs_quantized_val(j, k)-rhs_zero_point); // uint8计算
// === 循环之后 (*result)(i, k) *= lhs_scale * rhs_scale // 得到浮点数
           
           // 使用量化数表示浮点数之后运算 得到浮点数结果
           // 而浮点数结果 也需要进行量化
           // result_real_value = result_scale *(result_quantized_value - result_zero_point)
           // result_quantized_value = result_zero_point + result_real_value / result_scale

(*result_quantized_value)(i,k) += (lhs_quantized_val(i, j) - lhs_zero_point) * (rhs_quantized_val(j, k)-rhs_zero_point); // uint8计算 
      }
// 循环之后 
(*result_quantized_value)(i,k) = (*result_quantized_value)(i,k) * lhs_scale * rhs_scale / result_scale + result_zero_point;
// 得到 量化数
    }
  }

//// 总结版本===================================
  for (int i = 0; i < lhs.rows(); i++) 
  {// 每行
    for (int k = 0; k < rhs.cols(); k++) 
    {// 每列
     // (*result_quantized_value)(i, k) = 0;
     int32_accumulator = 0;
      for (int j = 0; j < lhs.cols(); j++)
      {
//(*result_quantized_value)(i,k) += (lhs_quantized_val(i, j) - lhs_zero_point) * (rhs_quantized_val(j, k) - rhs_zero_point); // uint8计算 
int32_accumulator += (lhs_quantized_val(i, j) - lhs_zero_point) * (rhs_quantized_val(j, k) - rhs_zero_point);
      }
// 循环之后 
(*result_quantized_value)(i,k) = int32_accumulator * lhs_scale * rhs_scale / result_scale + result_zero_point;
// 这里三个尺度参数 lhs_scale 、 rhs_scale 、 result_scale 差不多都是在 0~1.0 范围内的小数。
// 把这部分的 浮点运算也转换一下，变成 32位整数编制
    }
  }

```
## 小于1的小数 转换成 32位的整数
```c
void QuantizeMultiplierSmallerThanOne(float real_multiplier, // 实际小数 乘数
                                      std::int32_t* quantized_multiplier,
                                      int* right_shift) 
{
// 确保在 0.0~1.0 之间==================                              
  assert(real_multiplier > 0.f);
  assert(real_multiplier < 1.f);
// 改变范围 到 0.5~1 之间===============
  int s = 0;// 扩大倍数 记录， 之后右移相同的数量，就会还原
  while (real_multiplier < 0.5f) {
    real_multiplier *= 2.0f;
    s++;// 扩大倍数 记录， 之后右移相同的数量，就会还原
  }
  
// 转换浮点数乘子 [1/2, 1) 到 32位固定点整数
  std::int64_t q = static_cast<std::int64_t>(std::round(real_multiplier * (1ll << 31))); 
  // 1左移31位，后面的是两个ll 长整形，相当于扩大 2^31次方 =========
  assert(q <= (1ll << 31));
// 如果原数 real_multiplier 比较趋近于1，将其减半，同时扩大倍数记录-1
  if (q == (1ll << 31)) {
    q /= 2;// 将其减半
    s--;// 同时扩大倍数记录-1
  }
  assert(s >= 0);
  assert(q <= std::numeric_limits<std::int32_t>::max());
  *quantized_multiplier = static_cast<std::int32_t>(q);
  *right_shift = s;
}

// 调用===================================
  const float real_multiplier = lhs_scale * rhs_scale / result_scale;
  
  int right_shift;// 除去 左移31位，原数的扩大倍数记录
  
  std::int32_t quantized_multiplier = QuantizeMultiplierSmallerThanOne(real_multiplier, &quantized_multiplier, &right_shift);

```
## 量化卷积计算步骤
```c
1. 输入 量化的特征图 lhs_quantized_val, uint8类型, 偏移量 lhs_zero_point, int32类型;
2. 输入 量化的卷积核 rhs_quantized_val, uint8类型, 偏移量 rhs_zero_point, int32类型;
3. 转换 unit8 到 int32类型;
4. 每一块卷积求和(int32乘法求和有溢出风险，可换成固定点小数树乘法);
   int32_accumulator += (lhs_quantized_val(i, j) - lhs_zero_point) * (rhs_quantized_val(j, k) - rhs_zero_point);
5. 输入 量化的乘子 quantized_multiplier, int32类型 和 右移次数记录 right_shift, int类型;
6. 计算乘法，得到int32类型的结果 (int32乘法有溢出风险，可换成固定点小数树乘法);
   quantized_multiplier * int32_accumulator
   
7. 再左移动 right_shift 位还原，得到 int32的结果;
8. 最后再加上 结果的偏移量 result_zero_point;
   (7和8的顺序和 官方说的先后顺序颠倒);
9. 将int32类型结果 限幅到[0, 255], 再强制转换到 uint8类型;
10. 之后再 反量化到浮点数，更新统计输出值分布信息 max, min;
11. 再量化回 uint8;
11. 之后 经过 量化激活层;
12. 最后 反量化到 浮点数，本层网络输出;

13. 进入下一层
    循环执行 1~12 步骤
```


### 当遇到连续的被量化的操作时
	有一个优化是当连续出现多个被量化了的操作时，没有必要在每个操作前做反序列化/序列化，
	因为上一个操作的反序列化和下一个操作的序列化是会被互相抵消的。
	例如下图：
	
反量化和量化会抵消，左边是量化展开的，右边是去除冗余量化的
![](http://fjdu.github.io/pictures/2016-07-07-quantization2.png)
## 浮点数 Relu激活  FloatRelu()


## 量化数 Relu激活  QuantizedRelu()



# caffe中的代码
```c
void Quantization::ChooseIAOQuantizationParams(float min, float max, uint8_t* zero_point, float* scale) 
{
  min = std::min(min, 0.f);// 确保最小值 <= 0
  max = std::max(max, 0.f);// 确保最大值 >= 0

  // the min and max quantized values, as floating-point values
  // uint8 量化的数据范围
  const float qmin = 0;
  const float qmax = 255;

  // 计算量化缩放尺度================================
  // 每一个量化数据单位代表的浮点数大小 0.0~510.0 量化到 0~255 则每一个量化数代表 2
  const double scale_t = (max - min) / (qmax - qmin);
  
  // 计算 零点(偏移量)===========================
  const double initial_zero_point = qmin - min / scale_t;

  // 对  零点(偏移量) 限幅 并取整=================
  uint8_t nudged_zero_point = 0;
  if (initial_zero_point < qmin) 
  {
    nudged_zero_point = qmin;
  } 
  else if (initial_zero_point > qmax) 
  {
    nudged_zero_point = qmax;
  } 
  else 
  {
    nudged_zero_point = static_cast<uint8_t>(round(initial_zero_point));
  }
  
  *zero_point = nudged_zero_point;
  *scale = scale_t;
}



// 谷歌iao 整形 uint8 量化方法===========================
void Quantization::Quantize2IntegerArithmeticOnly()
{
 
 //=====计算每一层的量化参数==========================
  for (int i = 0; i < layer_names_.size(); ++i) 
  {
  //  il_in_.push_back((int)ceil(log2(abs_max_in_[i])+1));
  //  il_out_.push_back((int)ceil(log2(abs_max_out_[i])+1));
  //  il_params_.push_back((int)ceil(log2(abs_max_params_[i])+1));
  
  // 尺度==========================
   scale_in_.push_back(0);
   scale_params_.push_back(0); 
   scale_out_.push_back(0);
   
  // 偏移量 零点===================
   zero_point_in_.push_back(0),
   zero_point_params_.push_back(0), 
   zero_point_out_.push_back(0);
   
   ChooseIAOQuantizationParams(min_in_[i],  max_na_in_[i],  &zero_point_in_[i],  &scale_in_[i]);
   ChooseIAOQuantizationParams(min_out_[i], max_na_out_[i], &zero_point_out_[i], &scale_out_[i]);
   ChooseIAOQuantizationParams(min_params_[i], max_na_params_[i], &zero_point_params_[i], &scale_params_[i]);
  }
  
  // Debug
  for (int k = 0; k < layer_names_.size(); ++k) 
  {
    LOG(INFO) << "Layer " << layer_names_[k] <<
        ", zero point input=" << (int)zero_point_in_[k] << " scale input = " << scale_in_[k] << 
        ", zero point output=" << (int)zero_point_out_[k] << " scale output = " << scale_out_[k] << 
        ", zero point parameters=" << (int)zero_point_params_[k]<< " scale parameters = " << scale_params_[k];
  }

// 修改网络 并测试
  NetParameter param;  // 网络参数
  float accuracy;      // 网络精度
  Net<float>* net_test;
  caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
  param.mutable_state()->set_phase(caffe::TEST);
  
  EditNetDescriptionIAO(&param, "Convolution_and_InnerProduct","Parameters_and_Activations");
///*
  net_test = new Net<float>(param);

  net_test->CopyTrainedLayersFrom(weights_);
  
  RunForwardBatches(NULL, iterations_, net_test, &accuracy, this->net_type_); // 需要 特定的的iao层来支持网络前传
  delete net_test;
//*/
  param.release_state();
  
  WriteProtoToTextFile(param, model_quantized_);

  // Write summary of dynamic fixed point analysis to log
  LOG(INFO) << "------------------------------";
  LOG(INFO) << "Network accuracy analysis for";
  LOG(INFO) << "Convolutional (CONV) and fully";
  LOG(INFO) << "connected (FC) layers.";
  LOG(INFO) << "Baseline 32bit float: " << test_score_baseline_;
  
 LOG(INFO) << "Integer Arithmetic Only uint8 net: ";
  LOG(INFO) << "Accuracy: " << accuracy;
//  LOG(INFO) << "Please fine-tune.";
}

void Quantization::EditNetDescriptionIAO(NetParameter* param, const string layers_2_quantize, const string net_part)
// 需要量化的层 layers_2_quantize ： "Convolution_and_InnerProduct"  卷积层、全链接层
// 量化层的那些部件       net_part： "Parameters_and_Activations"    卷积核参数w，激活输入输出
{
 for (int i = 0; i < param->layer_size(); ++i) // 遍历每一层===========================
  {
  // 尺度==========================
 //  scale_in_.push_back(0);
 //  scale_params_.push_back(0); 
 //  scale_out_.push_back(0);
   
  // 偏移量 零点===================
 //  zero_point_in_.push_back(0),
 //  zero_point_params_.push_back(0), 
  // zero_point_out_.push_back(0);
   
// 卷积层========================================

  caffe::QuantizationParameter_Precision precision =
        caffe::QuantizationParameter_Precision_INTEGER_ARITHMETRIC_ONLY;
  
    if (layers_2_quantize.find("Convolution") != string::npos && param->layer(i).type().find("Convolution") != string::npos) 
    {
      // 卷积核参数w 部分
      if (net_part.find("Parameters") != string::npos) 
      {
        LayerParameter* param_layer = param->mutable_layer(i);
        param_layer->set_type("ConvolutionRistretto");
        param_layer->mutable_quantization_param()->set_scale_params( scale_params_[ConvlayerInLayers(param->layer(i).name())] );
        param_layer->mutable_quantization_param()->set_zero_point_params( zero_point_params_[ConvlayerInLayers(param->layer(i).name())] );
		
		 param_layer->mutable_quantization_param()->set_precision(precision);
      }
      // 激活输入输出
      if (net_part.find("Activations") != string::npos) 
      {
        LayerParameter* param_layer = param->mutable_layer(i);
        
        param_layer->set_type("ConvolutionRistretto");
        
        param_layer->mutable_quantization_param()->set_scale_in( scale_in_[ConvlayerInLayers(param->layer(i).name())] );
        param_layer->mutable_quantization_param()->set_zero_point_in(zero_point_in_[ConvlayerInLayers(param->layer(i).name())]);
        
        param_layer->mutable_quantization_param()->set_scale_out( scale_out_[ConvlayerInLayers(param->layer(i).name())] );
        param_layer->mutable_quantization_param()->set_zero_point_out( zero_point_out_[ConvlayerInLayers(param->layer(i).name())] );

		param_layer->mutable_quantization_param()->set_precision(precision);
      }
    }

// 全连接层=========================================
    if (layers_2_quantize.find("InnerProduct") != string::npos && (param->layer(i).type().find("InnerProduct") != string::npos ||
        param->layer(i).type().find("FcRistretto") != string::npos)) 
    {
      // 卷积核参数w 部分
      if (net_part.find("Parameters") != string::npos) 
	  {
        LayerParameter* param_layer = param->mutable_layer(i);
        param_layer->set_type("FcRistretto");
        param_layer->mutable_quantization_param()->set_scale_params( scale_params_[ConvlayerInLayers(param->layer(i).name())] );
        param_layer->mutable_quantization_param()->set_zero_point_params( zero_point_params_[ConvlayerInLayers(param->layer(i).name())] );
		
		param_layer->mutable_quantization_param()->set_precision(precision);
		
      }
      // 激活输入输出
      if (net_part.find("Activations") != string::npos) 
      {
        LayerParameter* param_layer = param->mutable_layer(i);
        
        param_layer->set_type("FcRistretto");
        
        param_layer->mutable_quantization_param()->set_scale_in( scale_in_[ConvlayerInLayers(param->layer(i).name())] );
        param_layer->mutable_quantization_param()->set_zero_point_in( zero_point_in_[ConvlayerInLayers(param->layer(i).name())]);
        
        param_layer->mutable_quantization_param()->set_scale_out( scale_out_[ConvlayerInLayers(param->layer(i).name())] );
        param_layer->mutable_quantization_param()->set_zero_point_out( zero_point_out_[ConvlayerInLayers(param->layer(i).name())] );

		param_layer->mutable_quantization_param()->set_precision(precision);
		
      }
    }
  }
}
// IAO 总网络id 在 卷积/全链接层 量化参数表中的id
int Quantization::ConvlayerInLayers(const string layer_name) {
  int pos = find(layer_names_.begin(), layer_names_.end(), layer_name)
      - layer_names_.begin();
  return pos;
}


```


