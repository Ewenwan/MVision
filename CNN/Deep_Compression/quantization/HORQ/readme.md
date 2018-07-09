# 2.7 高阶残差量化网络 二值网络+量化残差二值网络 HORQ
论文：Performance Guaranteed Network Acceleration via High-Order Residual Quantization

[论文链接](https://arxiv.org/abs/1708.08687.pdf)


	本文是对 XNOR-Networks 的改进，将CNN网络层的输入 进行高精度二值量化，
	从而实现高精度的二值网络计算，XNOR-Networks 也是对每个CNN网络层的权值和输入进行二值化，
	这样整个CNN计算都是二值化的，这样计算速度快，占内存小。
	一般对输入做二值化后模型准确率会下降特别厉害，
	而这篇文章提出的对权重和输入做high-order residual quantization的方法,
	可以在保证准确率的情况下大大压缩和加速模型。
	
	XNOR-Networks 对输入进行一级量化
	HORQ          对输入进行多级量化(对上级量化的残差再进行量化)
	
![](https://static.leiphone.com/uploads/new/article/740_740/201710/59e2f036c850f.png?imageMogr2/format/jpg/quality/90)
	
