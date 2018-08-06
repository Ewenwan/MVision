# TSQ Two-Step Quantization 两步量化策略  激活编码 + 权重量化转换
[Wang_Two-Step_Quantization_for_CVPR_2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Two-Step_Quantization_for_CVPR_2018_paper.pdf)

# 也属于 增量型量化 的方式 Incremental quantization 
      增量型量化的基本思路是，部分的对参数或者激活函数进⾏量化处理，使得整个量化过程更
      加平稳。这⾥的 部分 可以有多种层⾯，可以是数量上的部分，也可以是程度上的部分。

# 简介 这里
      在量化神经网络的硬件设计中，每一点都很重要。
      然而，极低的位表示通常会导致较大的精度下降。
      因此，如何训练 高精度的极低比特 神经网络 具有重要的意义。
      大多数现有的网络量化方法同时学习变换（低比特权重）以及编码（低比特激活）。
      这种紧密耦合使得优化问题变得困难，从而防止网络学习最优表示。
      在本文中，我们提出了一个简单但有效的两步量化（TSQ）框架，
      通过将网络量化问题分解为两个步骤：
          激活的编码学习 和 权重的变换函数学习。
          
      1、 对于第一步，我们提出了稀疏量化的编码学习方法。
          先是对 activation 做离散化处理，采⽤的⽅式和 Half-Wave Gaussian Quantizer (HWGQ) 半波高斯量化，⽆他，
          只是⼜强⾏加⼊了 sparse 的处理，让整个激活值更加的稀疏，从结果上来看，⼀定程度的稀疏，反⽽能让效果更好。
      
      2、 第二步可以表述为具有 低比特约束 的 非线性最小二乘回归问题，它可以以迭代的方式有效地求解。
          ⼀层⼀层的对 权重weight进行 量化学习。
          通过转换，最后是⽤⾮梯度下降⽅法，半闭式的解出了 离散化 weights。
          
          
 
# |low-bit weights 权重   | transformations 变换 |non-linear least square regression problem 非线性最小二乘方法|

# low-bit activations 激活| encodings 编码       |sparse suantization 稀疏-半波高斯量化 (HWGQ) |

# 评论
      按照这篇论⽂的提法，他是 decouple 了 activation 和 weights，但是这种说法其实
      只是形式上的问题。
      这篇⽂章对activation的处理并没有太⼤新意，对weights的处理，其实和之前的⽅法
      ⽆太⼤差别，从某种程度上可以理解成⼀种distillation。
      
# 其他 增量式量化
[增量式量化参考](https://github.com/compression-friendlies/Paper-Collection-of-Efficient-ML/blob/13f9be280121c71d3ab801b0d2f209b5e7684164/%E5%A2%9E%E9%87%8F%E5%9E%8B%E9%87%8F%E5%8C%96%E6%80%BB%E7%BB%93.pdf)

      1. INQ            数量上逐步量化
            http://arxiv.org/abs/1702.03044
      2. Bottom-to-Top  分层 来增量型量化。   是否可以结合  分层分量来增量量化。
            http://arxiv.org/abs/1607.02241 
      3. BinaryRelax
             http://arxiv.org/abs/1801.06313
      4. Stochastic-Quantization  逐步 随机量化
             https://arxiv.org/pdf/1708.01001.pdf
[代码](https://github.com/Ewenwan/Stochastic-Quantization)
      
      
      
