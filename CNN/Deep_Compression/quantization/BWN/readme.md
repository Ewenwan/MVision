# 二值系数网络 BWN

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
