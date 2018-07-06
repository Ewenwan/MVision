# 二值系数网络 BWN

[代码 lua 版本](https://github.com/Ewenwan/XNOR-Net)

[XNOR-Net-caffe](https://github.com/Ewenwan/XNOR-Net-1)

[theano-xnor-net](https://github.com/Ewenwan/theano-xnor-net)

[fast-xnor-net 纯C实现](https://github.com/Ewenwan/fast-xnor-net)

![](https://img-blog.csdn.net/20160715112109483)


### Implementation in PyTorch

The implementation of XNOR-Net is similar to those of BWN which including three steps: quantization, forward propagation and backward propagation.

### Experimental Results

In this paper, the authors conducted experiments on AlexNet and ResNet and validated the proposed methods using validation data set of ImageNet with single crop. The optimization method used in there experiments is ADAM as it can converge faster and have better performance with binary inputs. Experimental results are shown as follows:

| Model               | Top-1 Accuracy/ Top-5 Accuracy | Accuracy Loss    |
| ------------------- | ------------------------------ | ---------------- |
| AlexNet reference   | 56.6% / 80.2%                  |                  |
| AlexNet BC          | 35.4% / 61.0%                  | -21.2% / -19.2%  |
| AlexNet BWN         | 56.8% / 79.4%                  | 0.2% / -0.8%     |
| AlexNet BNN         | 27.9 % / 50.42%                | -28.7% / -29.78% |
| AlexNet XNOR-Net    | 44.2% / 69.2%                  | -12.4% / -11%    |
| ResNet-18 reference | 69.3% / 89.2%                  |                  |
| ResNet-18 BWN       | 60.8% / 83.0%                  | -8.5% / -5.8%    |
| ResNet-18 XNOR-Net  | 51.2% / 73.2%                  | -18.1% / -16%    |
| GoogLeNet reference | 71.3% / 90.0%                  |                  |
| GoogLeNet BWN       | 65.5% / 86.1%                  | -5.8% / -3.9%    |


      每年的年初是机器学习相关会议扎堆的时段，Matthieu与Itay于3月17日更新了他们的合作论文，
      进行了一些细节的调整，看起来是投稿前的最后准备。
      但就在一天前的3月16日，来自西雅图的Allen institute for AI和
      华盛顿大学的Mohammad Rastegari等人用新方法改进了二值系数网络BinaryConnect和全二值网络BinaryNet
      ，在大规模数据集ImageNet上分别提高预测准确率十几个百分点。
      其中，改进后的 二值系数网络BWN已达到被普遍接受的神经网络质量标准：
      只差单精度AlexNet3个百分点。


      BWN(Binary-Weights-Networks) 仅有参数二值化了，激活量和梯度任然使用全精度。XNOR-Net是BinaryNet的升级版。 
      主要思想： 
          1. 二值化时增加了缩放因子，同时梯度函数也有相应改变：
          W≈W^=αB=1n∑|W|ℓ1×sign(W)
          ∂C∂W=∂C∂W^(1n+signWα)

          2. XNOR-Net在激活量二值化前增加了BN层 
          3. 第一层与最后一层不进行二值化 
      实验结果： 
          在ImageNet数据集AlexNet架构下，BWN的准确率有全精度几乎一样，XNOR-Net还有较大差距(Δ=11%) 
          减少∼32×的参数大小，在CPU上inference阶段最高有∼58× 的加速。
          
      Binary-Weight-Networks 只是对CNN网络的滤波器进行二值化近似，使其占内存降低32倍多，
      XNOR-Networks 则对CNN网络的滤波器及其输入都进行二值化近似，
      
# 对于每一次CNN网络，我们使用一个三元素 《I,W,#》来表示，I 表示卷积输入，W表示滤波器，#表示卷积算子

## BWN

该网络主要是对W 进行二值化，主要是一些数学公式的推导，公式推导如下:
      
      对W进行二值化，使用 B 和缩放比例 a 来近似表达W
![](https://img-blog.csdn.net/20160715113533581)
      
      全精度权重W 和 加权二进制权重 aB 的误差函数，求解缩放比例a和二值权重B，使得误差函数值最小
![](https://img-blog.csdn.net/20160715113542440)

      误差函数展开
![](https://img-blog.csdn.net/20160715113549674)
      
      二值权重B的求解，误差最小，得到 W转置*B最大
![](https://img-blog.csdn.net/20160715113600645)

      缩放比例a的求解，由全精度权重W求解得到
![](https://img-blog.csdn.net/20160715113609159)

## BWN网络的训练


![](https://img-blog.csdn.net/20160715113831914)


## 异或网络 XNOR-Networks  对 I(神经元激活输出，下一层的输入) 及 W(权重参数) 都二值化

     最开始的输入X，权重W, 使用b*H代替X, 使用a*B代替W , a,b为缩放比例，H,B为 二值矩阵。
![](https://img-blog.csdn.net/20160715114052958)
      
     网络中间隐含层的量化，二值化的矩阵相乘，在乘上一个矩阵和一个缩放因子。
![](https://img-blog.csdn.net/20160715114402250)

      主框架:
![](https://img-blog.csdn.net/20160715114256287)

      由于在一般网络下，一层卷积的 kernel 规格是固定的，kernel 和 input 在进行卷积的时候，
      input 会有重叠的地方，所以在进行量化因子的运算时，先对 input 全部在 channel 维求平均，
      得到的矩阵 A，再和一个 w x h 的卷积核 k 进行卷积得到比例因子矩阵 K，
      
      其中：
      Kij = 1 / (w x h)
      





