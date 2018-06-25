# 量化网络 QNN Quantized Neural Networks
    对BNN的简单扩展，量化激活函数，有线性量化与log量化两种，其1-bit量化即为BinaryNet。
    在正向传播过程中加入了均值为0的噪音。



## QNN
[QNN Quantized Neural Networks ](https://arxiv.org/pdf/1609.07061.pdf)

        对BNN的简单扩展，
        量化激活函数，
        有线性量化与log量化两种，
        其1-bit量化即为BinaryNet。
        在正向传播过程中加入了均值为0的噪音。 
        BNN约差于XNOR-NET（<3%），
        QNN-2bit activation 略优于DoReFaNet 2-bit activation


    激活函数量 线性量化：

        LinearQuant(x, bitwidth)= Clip(round(x/bitwidth)*bitwidth,  minV, maxV )

        激活函数为 整数阶梯函数  
        最小值 minV
        最大值 maxV
        中间   线性阶梯整数

    log量化：

        LogQuant(x, bitwidth)、 = Clip (AP2(x), minV, maxV )

        AP2(x) = sign(x) × 2^round(log2|x|)
         平方近似


## QCNN

[Quantized-CNN for Mobile Devices 代码](https://github.com/Ewenwan/quantized-cnn)

[代码](https://github.com/Ewenwan/quantized-cnn)

    一种量化CNN的方法（Q-CNN），
    量化卷积层中的滤波器和全连接层中的加权矩阵，
    通过量化网络参数，
    用近似内积计算有效地估计卷积和全连接层的响应,
    最小化参数量化期间每层响应的估计误差，
    更好地保持模型性能。

    步骤：
    首先，全连接的层保持不变,用纠错量化所有卷积层。
    其次，利用ILSVRC-12训练集对量化网络的全连接层进行微调，恢复分类精度。
    最后，纠错量化微调的层网络的全连接。
