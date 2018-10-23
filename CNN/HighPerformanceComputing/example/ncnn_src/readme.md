# ncnn　源文件 注释解读
[ncnn 源码分析](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/example/ncnn_%E6%BA%90%E7%A0%81%E5%88%86%E6%9E%90.md)

    /src目录：
        目录顶层下是一些基础代码，如宏定义，平台检测，mat数据结构，layer定义，blob定义，net定义等。
      platform.h.in 平台检测
      benchmark.cpp benchmark.h 测试各个模型的执行速度
      allocator.cpp allocator.h 内存对齐
      paramdict.cpp paramdict.h 层参数解析 读取二进制格式、字符串格式、密文格式的参数文件
      opencv.cpp opencv.h       opencv 风格的数据结构 的 最小实现
                                大小结构体 Size 
                                矩阵框结构体 Rect_ 交集 并集运算符重载
                                点结构体     Point_
                                矩阵结构体   Mat     深拷贝 浅拷贝 获取指定矩形框中的roi 读取图像 写图像 双线性插值算法改变大小
      mat.cpp mat.h             3维矩阵数据结构, 在层间传播的就是Mat数据，Blob数据是花架子
                                substract_mean_normalize(); 去均值并归一化
                                half2float();               float16 的 data 转换成 float32 的 data
                                copy_make_border();         矩阵周围填充
                                resize_bilinear_image();    双线性插值
      modelbin.cpp modelbin.h   从文件中载入模型权重、从内存中载入、从数组中载入
      layer.cpp layer.h         层接口，四种前向传播接口函数
      Blob数据是花架子

      net.cpp net.h             ncnn框架接口：
                                注册 用户定义的新层 Net::register_custom_layer();
                                网络载入 模型参数   Net::load_param();
                                载入     模型权重   Net::load_model();
                                网络blob 输入       Net::input();
                                网络前向传          Net::forward_layer();    被Extractor::extract() 执行
                                创建网络模型提取器   Net::create_extractor();
                                模型提取器提取某一层输出 Extractor::extract();

        ./src/layer下是所有的layer定义代码
        ./src/layer/arm是arm下的计算加速的layer
        ./src/layer/x86是x86下的计算加速的layer。

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
├── dropout.cpp               // 随机失活层 在训练时由于舍弃了一些神经元,因此在测试时需要在激励的结果中乘上因子p进行缩放.
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
    
##  innerproduct（以下称IP层）

    // 1*1 卷积 全链接 ==
    // 假设innerproduct（以下称IP层）层的层参数output_num为n, IP层输入blob尺寸为m*c*w*h,
    //那么输出尺寸为m*n   ,即m*n*1*1,不难看出IP层把输入blob的m个二维矩阵（图片）通过与权重矩阵的卷积运算，
    // 最终转化为了一个m*n的二维矩阵，很显然，这个二维矩阵感知了输入的m张图片的综合特征。
