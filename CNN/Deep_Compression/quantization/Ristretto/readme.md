# 7. Ristretto是一个和Caffe类似的框架。

    Ristretto是一个自动化的CNN近似工具，可以压缩32位浮点网络。
    Ristretto是Caffe的扩展，允许以有限的数字精度测试、训练和微调网络。
    
    本文介绍了几种网络压缩的方法，压缩特征图和参数。
    方法包括定点法（Fixed Point Approximation）、
    动态定点法（Dynamic Fixed Point Approximation）、
    迷你浮点法（Minifloat Approximation）和
    乘法变移位法（Turning Multiplications Into Bit Shifts），
    所压缩的网络包括LeNet、CIFAR-10、AlexNet和CaffeNet等。
    注：Ristretto原指一种特浓咖啡（Caffe），本文的Ristretto沿用了Caffe的框架。
    
[Ristretto: Hardware-Oriented Approximation of Convolutional Neural Networks](https://arxiv.org/pdf/1605.06402.pdf)

[caffe+Ristretto 工程代码](https://github.com/pmgysel/caffe)

[代码 主要修改](https://github.com/MichalBusta/caffe/commit/55c64c202fc8fca875e108b48c13993b7fdd0f63)

# Ristretto速览
    Ristretto Tool：
           Ristretto工具使用不同的比特宽度进行数字表示，
           执行自动网络量化和评分，以在压缩率和网络准确度之间找到一个很好的平衡点。
    Ristretto Layers：
           Ristretto重新实现Caffe层并模拟缩短字宽的算术。
    测试和训练：
           由于将Ristretto平滑集成Caffe，可以改变网络描述文件来量化不同的层。
           不同层所使用的位宽以及其他参数可以在网络的prototxt文件中设置。
           这使得我们能够直接测试和训练压缩后的网络，而不需要重新编译。

# 逼近方案
    Ristretto允许以三种不同的量化策略来逼近卷积神经网络：
        1、动态固定点：修改的定点格式。
        2、Minifloat：缩短位宽的浮点数。
        3、两个幂参数：当在硬件中实现时，具有两个幂参数的层不需要任何乘法器。


    这个改进的Caffe版本支持有限数值精度层。所讨论的层使用缩短的字宽来表示层参数和层激活（输入和输出）。
    由于Ristretto遵循Caffe的规则，已经熟悉Caffe的用户会很快理解Ristretto。
    下面解释了Ristretto的主要扩展：
    
# Ristretto的主要扩展
    1、Ristretto Layers
    
    Ristretto引入了新的有限数值精度层类型。
    这些层可以通过传统的Caffe网络描述文件（* .prototxt）使用。 
    下面给出一个minifloat卷积层的例子：
    
```
    layer {
      name: "conv1"
      type: "ConvolutionRistretto"
      bottom: "data"
      top: "conv1"
      convolution_param {
        num_output: 96
        kernel_size: 7
        stride: 2
        weight_filler {
          type: "xavier"
        }
      }
      quantization_param {
        precision: MINIFLOAT   # MANT:mantissa，尾数(有效数字) 
        mant_bits: 10
        exp_bits: 5
      }
    }
```
    该层将使用半精度（16位浮点）数字表示。
    卷积内核、偏差以及层激活都被修剪为这种格式。
    
    注意与传统卷积层的三个不同之处：
        1、type变成了ConvolutionRistretto；
        2、增加了一个额外的层参数：quantization_param；
        3、该层参数包含用于量化的所有信息。
        
    2、 Blobs
        Ristretto允许精确模拟资源有限的硬件加速器。
        为了与Caffe规则保持一致，Ristretto在层参数和输出中重用浮点Blob。
        这意味着有限精度数值实际上都存储在浮点数组中。
    3、Scoring
        对于量化网络的评分，Ristretto要求
          a. 训练好的32位FP网络参数
          b. 网络定义降低精度的层
        第一项是Caffe传统训练的结果。Ristretto可以使用全精度参数来测试网络。
             这些参数默认情况下使用最接近的方案，即时转换为有限精度。
        至于第二项——模型说明——您将不得不手动更改Caffe模型的网络描述，
             或使用Ristretto工具自动生成Google Protocol Buffer文件。
            # score the dynamic fixed point SqueezeNet model on the validation set*
            ./build/tools/caffe test --model=models/SqueezeNet/RistrettoDemo/quantized.prototxt \
            --weights=models/SqueezeNet/RistrettoDemo/squeezenet_finetuned.caffemodel \
            --gpu=0 --iterations=2000

# Ristretto: SqueezeNet 示例

    1、下载原始 32bit FP 浮点数 网络权重
[地址](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.0)

    2、微调再训练一个低精度 网络权重 
[微调了一个8位动态定点SqueezeNet]()
    3、
    4、
    5、


