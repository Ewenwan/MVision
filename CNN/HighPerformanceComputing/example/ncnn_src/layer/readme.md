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
│   ├── sigmoid_arm.cpp                 // sigmod 负指数倒数归一化 激活层  1/（1 + e^(-zi)）
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
├── binaryop.cpp              // 二元操作: add，sub， div， mul，mod等
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
├── detectionoutput.cpp       // ssd 的检测输出层================================
├── detectionoutput.h
├── dropout.cpp               // 随机失活层
├── dropout.h
├── eltwise.cpp               // 逐元素操作， product(点乘), sum(相加减) 和 max(取大值)
├── eltwise.h
├── elu.cpp                   // 指数线性单元relu激活层 Prelu : (a*x, x) ----> Erelu : (a*(e^x - 1), x) 
├── elu.h
├── embed.cpp                 // 嵌入层，用在网络的开始层将你的输入转换成向量
├── embed.h
├── expanddims.cpp            // 增加维度
├── expanddims.h
├── exp.cpp                   // 指数映射
├── exp.h
├── flatten.cpp               // 摊平层
├── flatten.h
├── innerproduct.cpp          // 全连接层
├── innerproduct.h
├── input.cpp                 // 数据输入层
├── input.h
├── instancenorm.cpp          // 单样本 标准化 规范化
├── instancenorm.h
├── interp.cpp                // 插值层 上下采样等
├── interp.h
├── log.cpp                   // 对数层
├── log.h
├── lrn.cpp                   // Local Response Normalization，即局部响应归一化层
├── lrn.h                     // 对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，
|                             // 并抑制其他反馈较小的神经元，增强了模型的泛化能力
├── lstm.cpp                
├── lstm.h                    // lstm 长短词记忆层
├── memorydata.cpp            // 内存数据层
├── memorydata.h
├── mvn.cpp
├── mvn.h
├── normalize.cpp             // 归一化
├── normalize.h
├── padding.cpp               // 填充，警戒线
├── padding.h
├── permute.cpp               //  ssd 特有层 交换通道顺序 [bantch_num, channels, h, w] ---> [bantch_num, h, w, channels]]=========
├── permute.h
├── pooling.cpp               // 池化层
├── pooling.h
├── power.cpp                 // 平移缩放乘方 : (shift + scale * x) ^ power
├── power.h
├── prelu.cpp                 // Prelu  (a*x,x)
├── prelu.h
├── priorbox.cpp              // ssd 独有的层 建议框生成层 L1 loss 拟合============================
├── priorbox.h
├── proposal.cpp              // faster rcnn 独有的层 建议框生成，将rpn网络的输出转换成建议框======== 
├── proposal.h
├── quantize.cpp              // 量化层
├── quantize.h
├── reduction.cpp             // 将输入的特征图按照给定的维度进行求和或求平均
├── reduction.h
├── relu.cpp                  // relu 激活层： (0,x)
├── relu.h
├── reorg.cpp                 // yolov2 独有的层， 一拆四层，一个大矩阵，下采样到四个小矩阵=================
├── reorg.h
├── reshape.cpp               // 变形层： 在不改变数据的情况下，改变输入的维度
├── reshape.h
├── rnn.cpp                   // rnn 循环神经网络
├── rnn.h
├── roipooling.cpp            // faster Rcnn 独有的层， ROI池化层： 输入m*n 均匀划分成 a*b个格子后池化，得到固定长度的特征向量 ==========
├── roipooling.h
├── scale.cpp                 // bn 层之后的 平移缩放层
├── scale.h
├── shufflechannel.cpp        // ShuffleNet 独有的层，通道打乱，通道混合层=================================
├── shufflechannel.h
├── sigmoid.cpp               // 负指数倒数归一化层  1/(1 + e^(-zi))
├── sigmoid.h
├── slice.cpp                 // concat的反向操作， 通道分开层，适用于多任务网络
├── slice.h
├── softmax.cpp               // 指数求和归一化层  e^(zi) / sum(e^(zi))
├── softmax.h
├── split.cpp                 // 将blob复制几份，分别给不同的layer，这些上层layer共享这个blob。
├── split.h
├── spp.cpp                   // 空间金字塔池化层 1+4+16=21 SPP-NET 独有===================================
├── spp.h
├── squeeze.cpp               // squeezeNet独有层， Fire Module, 一层conv层变成两层：squeeze层+expand层, 1*1卷积---> 1*1 + 3*3=======
├── squeeze.h
├── tanh.cpp                  // 双曲正切激活函数  (e^(zi) - e^(-zi)) / (e^(zi) + e^(-zi))
├── tanh.h
├── threshold.cpp             // 阈值函数层
├── threshold.h
├── tile.cpp                  // 将blob的某个维度，扩大n倍。比如原来是1234，扩大两倍变成11223344。
├── tile.h
├── unaryop.cpp               // 一元操作: abs， sqrt， exp， sin， cos，conj（共轭）等
├── unaryop.h
|
|==============================x86下特殊的优化层=====
├── x86
│   ├── avx_mathfun.h                    // x86 数学函数
│   ├── convolution_1x1.h                // 1*1 float32 卷积
│   ├── convolution_1x1_int8.h           // 1×1 int8 卷积
│   ├── convolution_3x3.h                // 3*3 float32 卷积
│   ├── convolution_3x3_int8.h           // 3×3 int8 卷积
│   ├── convolution_5x5.h                // 5*5 float32 卷积 
│   ├── convolutiondepthwise_3x3.h       // 3*3 float32 逐通道卷积
│   ├── convolutiondepthwise_3x3_int8.h  // 3*3 int8 逐通道卷积
│   ├── convolutiondepthwise_x86.cpp     //  逐通道卷积
│   ├── convolutiondepthwise_x86.h
│   ├── convolution_x86.cpp              //  卷积
│   ├── convolution_x86.h
│   └── sse_mathfun.h                    // sse优化 数学函数
├── yolodetectionoutput.cpp              // yolo-v2 目标检测输出层=========================================
└── yolodetectionoutput.h
```

## 	caffe中layer的一些特殊操作，比如split
```c
slice：在某一个维度，按照给定的下标，blob拆分成几块。
       比如要拆分channel，总数50，下标为10,20,30,40，
       那就是分成5份，每份10个channel，输出5个layer。

concat：在某个维度，将输入的layer组合起来，是slice的逆过程。

split：将blob复制几份，分别给不同的layer，这些上层layer共享这个blob。

tile：将blob的某个维度，扩大n倍。比如原来是1234，扩大两倍变成11223344。

reduction：将某个维度缩减至1维，方法可以是sum、mean、asum、sumsq。

reshape：这个很简单，就是matlab里的reshape。

eltwise：将几个同样大小的layer，合并为1个，合并方法可以是相加、相乘、取最大。

flatten：将中间某几维合并，其实可以用reshape代替。
```

## reshape 变形
    # 假设原数据为：32*3*28*28， 表示32张3通道的28*28的彩色图片
     reshape_param {
       shape {
       dim: 0  32->32 # 0表示和原来一致
       dim: 0  3->3
       dim: 14 28->14
       dim: -1 #让其推断出此数值
       }
    }
    # 输出数据为：32*3*14*56


