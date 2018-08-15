
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

## 2. Quantization Inference  

### Inference  
    量化就是把float型的数值映射到int8或者uint8来进行卷积运算  

![formula1](https://github.com/Ewenwan/camel007.github.io/blob/master/img/2018-06-10/formula1.jpg)

    r - 需要量化的float型数值  
    q - 量化后的uint8类型数值  
    Z - 量化前r = 0时，量化后q的数值  
    S - 为了能把量化后的q还原到r, 引入了一个缩放系数  

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
## 浮点数卷积运算
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
## 量化卷积运算
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
// 得到 量化数
    }
  }

```



