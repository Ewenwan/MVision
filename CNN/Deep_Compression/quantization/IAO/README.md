---
layout:     post
title:      Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference Notes
date:       2018-06-10
author:     Yao
header-img: img/2018-06-10/header.jpg
catalog: true
tags:
    - Deep Learning
    - INT8
    - Speedup
    - quantization
---

## 1. Introduction  
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

### Graph illustration

#### simple graph for single layer

- origin

![simple_graph_origin](https://github.com/ICEORY/iceory.gitbook.io/tree/39ddec85f635ab3b4ac127efe6172e53e487f9fa/Network%20Quantization/fig/integer_arithmetic_only/simple_origin.png)

- quantized

![simple_graph_origin](https://github.com/ICEORY/iceory.gitbook.io/tree/39ddec85f635ab3b4ac127efe6172e53e487f9fa/Network%20Quantization/fig/integer_arithmetic_only/simple_quantize.png)

#### layer with bypass

- origin

![simple_graph_origin](https://github.com/ICEORY/iceory.gitbook.io/tree/39ddec85f635ab3b4ac127efe6172e53e487f9fa/Network%20Quantization/fig/integer_arithmetic_only/bypass_origin.png)

- quantized

![simple_graph_origin](https://github.com/ICEORY/iceory.gitbook.io/tree/39ddec85f635ab3b4ac127efe6172e53e487f9fa/Network%20Quantization/fig/integer_arithmetic_only/bypass_quantize.png)

#### convolutional layer with batch normalization

- training

![simple_graph_origin](https://github.com/ICEORY/iceory.gitbook.io/tree/39ddec85f635ab3b4ac127efe6172e53e487f9fa/Network%20Quantization/fig/integer_arithmetic_only/conv_bn_training.png)

- inference

![simple_graph_origin](https://github.com/ICEORY/iceory.gitbook.io/tree/39ddec85f635ab3b4ac127efe6172e53e487f9fa/Network%20Quantization/fig/integer_arithmetic_only/conv_bn_inference.png)

- training with fold

![simple_graph_origin](https://github.com/ICEORY/iceory.gitbook.io/tree/39ddec85f635ab3b4ac127efe6172e53e487f9fa/Network%20Quantization/fig/integer_arithmetic_only/conv_bn_training_fold.png)

- training with fold quantized

![simple_graph_origin](https://github.com/ICEORY/iceory.gitbook.io/tree/39ddec85f635ab3b4ac127efe6172e53e487f9fa/Network%20Quantization/fig/integer_arithmetic_only/conv_bn_training_fold_quantize.png)
