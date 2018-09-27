# ncnn 支持的层
```c
├── absval.cpp                       // 绝对值层
├── absval.h
├── argmax.cpp                       // 最大值层
├── argmax.h
├── arm ============================ arm平台下的层
│   ├── absval_arm.cpp               // 绝对值层
│   ├── absval_arm.h
│   ├── batchnorm_arm.cpp            // 批归一化 去均值除方差
│   ├── batchnorm_arm.h
│   ├── bias_arm.cpp                 // 偏置
│   ├── bias_arm.h
│   ├── convolution_1x1.h            // 1*1 float32 卷积
│   ├── convolution_1x1_int8.h       // 1*1 int8    卷积
│   ├── convolution_2x2.h            // 2*2 float32 卷积
│   ├── convolution_3x3.h            // 3*3 float32 卷积
│   ├── convolution_3x3_int8.h       // 3*3 int8    卷积
│   ├── convolution_4x4.h            // 4*4 float32 卷积
│   ├── convolution_5x5.h            // 5*5 float32 卷积
│   ├── convolution_7x7.h            // 7*7 float32 卷积
│   ├── convolution_arm.cpp          // 卷积层
│   ├── convolution_arm.h
│   ├── convolutiondepthwise_3x3.h      // 3*3 逐通道 float32 卷积
│   ├── convolutiondepthwise_3x3_int8.h // 3*3 逐通道 int8    卷积 
│   ├── convolutiondepthwise_arm.cpp    // 逐通道卷积
│   ├── convolutiondepthwise_arm.h
│   ├── deconvolution_3x3.h             // 3*3 反卷积
│   ├── deconvolution_4x4.h             // 4*4 反卷积
│   ├── deconvolution_arm.cpp           // 反卷积
│   ├── deconvolution_arm.h
│   ├── deconvolutiondepthwise_arm.cpp  // 反逐通道卷积
│   ├── deconvolutiondepthwise_arm.h
│   ├── dequantize_arm.cpp              // 反量化
│   ├── dequantize_arm.h
│   ├── eltwise_arm.cpp                 // 逐元素操作，product(点乘), sum(相加减) 和 max(取大值)
│   ├── eltwise_arm.h
│   ├── innerproduct_arm.cpp            // 即 fully_connected (fc)layer, 全连接层
│   ├── innerproduct_arm.h
│   ├── lrn_arm.cpp                     // Local Response Normalization，即局部响应归一化层
│   ├── lrn_arm.h
│   ├── neon_mathfun.h                  // neon 数学函数库
│   ├── pooling_2x2.h                   // 2*2 池化层
│   ├── pooling_3x3.h                   // 3*3 池化层
│   ├── pooling_arm.cpp                 // 池化层
│   ├── pooling_arm.h
│   ├── prelu_arm.cpp                   // (a*x,x) 前置relu激活层
│   ├── prelu_arm.h
│   ├── quantize_arm.cpp                // 量化层
│   ├── quantize_arm.h
│   ├── relu_arm.cpp                    // relu 层 (0,x)
│   ├── relu_arm.h
│   ├── scale_arm.cpp                   // BN层后的 平移和缩放层 scale
│   ├── scale_arm.h
│   ├── sigmoid_arm.cpp                 // sigmod 负指数倒数归一化 激活层  1/（1 + e^(-z)）
│   ├── sigmoid_arm.h
│   ├── softmax_arm.cpp                 // softmax 指数求和归一化 激活层   e^(zi) / sum(e^(zi))
│   └── softmax_arm.h
|
|
|================================ 普通平台 x86等，待优化=============
├── batchnorm.cpp             // 批归一化 去均值除方差
├── batchnorm.h
├── bias.cpp                  // 偏置
├── bias.h
├── binaryop.cpp              // 二进制操作层
├── binaryop.h
├── bnll.cpp                  // 
├── bnll.h
├── clip.cpp                  // 通道分路
├── clip.h
├── concat.cpp                // 通道叠加
├── concat.h
├── convolution.cpp           // 普通卷积层
├── convolutiondepthwise.cpp  // 逐通道卷积
├── convolutiondepthwise.h
├── convolution.h 
├── crop.cpp                  // 剪裁层
├── crop.h
├── deconvolution.cpp         // 反卷积
├── deconvolutiondepthwise.cpp// 反逐通道卷积
├── deconvolutiondepthwise.h
├── deconvolution.h
├── dequantize.cpp            // 反量化
├── dequantize.h
├── detectionoutput.cpp       // ssd 的检测输出层
├── detectionoutput.h
├── dropout.cpp               // 随机失活层
├── dropout.h
├── eltwise.cpp               // 逐元素操作， product(点乘), sum(相加减) 和 max(取大值)
├── eltwise.h
├── elu.cpp                   // 指数线性单元relu激活层 Prelu : (a*x, x) ----> Erelu : (a*(e^x - 1), x) 
├── elu.h
├── embed.cpp                 // 嵌入层 
├── embed.h
├── expanddims.cpp
├── expanddims.h
├── exp.cpp
├── exp.h
├── flatten.cpp
├── flatten.h
├── innerproduct.cpp
├── innerproduct.h
├── input.cpp
├── input.h
├── instancenorm.cpp
├── instancenorm.h
├── interp.cpp
├── interp.h
├── log.cpp
├── log.h
├── lrn.cpp
├── lrn.h
├── lstm.cpp
├── lstm.h
├── memorydata.cpp
├── memorydata.h
├── mvn.cpp
├── mvn.h
├── normalize.cpp
├── normalize.h
├── padding.cpp
├── padding.h
├── permute.cpp
├── permute.h
├── pooling.cpp
├── pooling.h
├── power.cpp
├── power.h
├── prelu.cpp
├── prelu.h
├── priorbox.cpp
├── priorbox.h
├── proposal.cpp
├── proposal.h
├── quantize.cpp
├── quantize.h
├── reduction.cpp
├── reduction.h
├── relu.cpp
├── relu.h
├── reorg.cpp
├── reorg.h
├── reshape.cpp
├── reshape.h
├── rnn.cpp
├── rnn.h
├── roipooling.cpp
├── roipooling.h
├── scale.cpp
├── scale.h
├── shufflechannel.cpp
├── shufflechannel.h
├── sigmoid.cpp
├── sigmoid.h
├── slice.cpp
├── slice.h
├── softmax.cpp
├── softmax.h
├── split.cpp
├── split.h
├── spp.cpp
├── spp.h
├── squeeze.cpp
├── squeeze.h
├── tanh.cpp
├── tanh.h
├── threshold.cpp
├── threshold.h
├── tile.cpp
├── tile.h
├── unaryop.cpp
├── unaryop.h
|
|==============================x86下特殊的优化层=====
├── x86
│   ├── avx_mathfun.h
│   ├── convolution_1x1.h
│   ├── convolution_1x1_int8.h
│   ├── convolution_3x3.h
│   ├── convolution_3x3_int8.h
│   ├── convolution_5x5.h
│   ├── convolutiondepthwise_3x3.h
│   ├── convolutiondepthwise_3x3_int8.h
│   ├── convolutiondepthwise_x86.cpp
│   ├── convolutiondepthwise_x86.h
│   ├── convolution_x86.cpp
│   ├── convolution_x86.h
│   └── sse_mathfun.h
├── yolodetectionoutput.cpp
└── yolodetectionoutput.h
```
