# 训练三值量化 TTQ  训练浮点数量化
[Trained Ternary Quantization  TTQ](https://arxiv.org/pdf/1612.01064.pdf)

[代码](https://github.com/Ewenwan/ternarynet)

## Experimental Results:

#### Error Rate of Finetuned TTQ ResNet models on CIFAR-10:

| Network       | Full Precision | TTQ         |
| ------------- | ------------- | ----------- |
| ResNet-20     | 8.23          | 8.87        |
| ResNet-32     | 7.67          | 7.63        |
| ResNet-44     | 7.18          | 7.02        |
| ResNet-56     | 6.80          | 6.44        |

#### Error Rate of TTQ AlexNet model on ImageNet from scratch:

| Network       | Full Precision | TTQ         |
| ------------- | ------------- | ----------- |
| Top1-error    | 42.8          | 42.5        |
| Top5-error    | 19.7          | 20.3        |


[博客参考](https://blog.csdn.net/yingpeng_zhong/article/details/80382704)

    提供一个三值网络的训练方法。
    对AlexNet在ImageNet的表现，相比32全精度的提升0.3%。
    与TWN类似，
    只用参数三值化(训练得到的浮点数)，
    但是正负缩放因子不同，
    且可训练，由此提高了准确率。
    ImageNet-18模型仅有3%的准确率下降。

    对于每一层网络，三个值是32bit浮点的{−Wnl,0,Wpl}，
    Wnl、Wpl是可训练的参数。
    另外32bit浮点的模型也是训练的对象，但是阈值Δl是不可训练的。 
    由公式(6)从32bit浮点的到量化的三值： 
![](https://img-blog.csdn.net/20180520154233906?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lpbmdwZW5nX3pob25n/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

    由(7)算出Wnl、Wpl的梯度:
![](https://img-blog.csdn.net/20180520154430336?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lpbmdwZW5nX3pob25n/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
    
    其中:
![](https://img-blog.csdn.net/20180520154553848?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lpbmdwZW5nX3pob25n/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

    由(8)算出32bit浮点模型的梯度
![](https://img-blog.csdn.net/20180520154744861?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lpbmdwZW5nX3pob25n/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

    由(9)给出阈值，这种方法在CIFAR-10的实验中使用阈值t=0.05。
    而在ImageNet的实验中，并不是由通过钦定阈值的方式进行量化的划分，
    而是钦定0值的比率r，即稀疏度。 
![](https://img-blog.csdn.net/20180520155802296?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lpbmdwZW5nX3pob25n/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
    
    整个流程:
![](https://img-blog.csdn.net/20180520154900829?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lpbmdwZW5nX3pob25n/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
