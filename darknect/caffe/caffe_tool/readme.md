# caffe 工具
[参考](https://github.com/Ewenwan/caffe_tools)

## 计算模型参数等

## 吸收BN层
[模型优化：BatchNorm合并到卷积中](https://blog.csdn.net/wfei101/article/details/78635557)


      bn层即batch-norm层，一般是深度学习中用于加速训练速度和一种方法，
      一般放置在卷积层（conv层）或者全连接层之后，
      将数据归一化并加速了训练拟合速度。
      但是ｂｎ层虽然在深度学习模型训练时起到了一定的积极作用，
      但是在预测时因为凭空多了一些层，影响了整体的计算速度并占用了更多内存或者显存空间。
      所以我们设想如果能将ｂｎ层合并到相邻的卷积层或者全连接层之后就好了.
      
      
      源网络 prototxt去除 BN和 scale
      
      每一层的 BN和scale参数 被用来修改 每一层的权重W 和 偏置b
      

## temsorflow 模型转 caffe
[temsorflow 模型](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

      dump_tensorflow_weights.py ： 
      模型优化:BatchNorm合并到卷积中， dump the weights of conv layer and batchnorm layer.
         
      load_caffe_weights.py ：
      load the dumped weights to deploy.caffemodel.
   
##  caffe  coco模型 转 voc模型
      coco2voc.py


## 模型修改
```py
import numpy as np
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

# //修改后的prototxt
src_prototxt = "xxx.prototxt"

# //原始的prototxt
old_prototxt = "s.prototxt"
old_caffemodel = "s.caffemodel"

# 创建网络模型对象
caffe.set_mode_cpu()
net = caffe.Net(src_prototxt, caffe.TEST)
net_old = caffe.Net(old_prototxt, old_caffemodel, caffe.TEST)

src_net_params = caffe_pb2.NetParameter()
text_format.Merge(open(src_prototxt).read(), src_net_params)

#拷贝相同名字层的参数
for k,v in net_old.params.items():
    # print (k,v[0].data.shape)
    # print (np.size(net_old.params[k]))
    if(k in net.layer_dict.keys()):
        print(k, v[0].data.shape)
        print(np.size(net_old.params[k]))
        for i in range(np.size(net_old.params[k])):
           net.params[k][i].data[:] = np.copy(net_old.params[k][i].data[:])
net.save("eur_single.caffemodel")
```


## 模型计算量
[参考](https://github.com/Captain1986/CaptainBlackboard/blob/master/D%230023-CNN%E6%A8%A1%E5%9E%8B%E8%AE%A1%E7%AE%97%E9%87%8F%E4%BC%B0%E8%AE%A1/D%230023.md)

在我们训练的深度学习模型在资源受限的嵌入式设备上落地时，**精度不是我们唯一的考量因素**，我们还需要考虑

1. **安装包的大小**，如果你的模型文件打包进app一起让客户下载安装，那么动辄数百MB的模型会伤害用户的积极性；

2. 模型速度，或者说**计算量的大小**。现在手机设备上的图片和视频的分辨率越来越大，数据量越来越多；对于视频或者游戏，FPS也越来越高，这都要求我们的模型在计算时，速度越快越好，计算量越小越好；

3. 运行时**内存占用大小**，内存一直都是嵌入式设备上的珍贵资源，占用内存小的模型对硬件的要求低，可以部署在更广泛的设备上，降低我们**算法落地的成本**；况且，一些手机操作系统也不会分配过多的内存给单一一个app，当app占用内存过多，系统会kill掉它；

4. **耗电量大小**，智能手机发展到今天，最大的痛点一直是电池续航能力和发热量，如果模型计算量小，内存耗用小的话，自然会降低电量的消耗速度。

### 计算量评价指标

一个朴素的评估模型速度的想法是评估它的计算量。一般我们用FLOPS，即每秒浮点操作次数FLoating point OPerations per Second这个指标来衡量GPU的运算能力。这里我们用MACC，即乘加数Multiply-ACCumulate operation，或者叫MADD，来衡量模型的计算量。

不过这里要说明一下，用MACC来估算模型的计算量只能**大致地**估算一下模型的速度。模型最终的的速度，不仅仅是和计算量多少有
关系，还和诸如**内存带宽**、优化程度、CPU流水线、Cache之类的因素也有很大关系。

为什么要用乘加数来评估计算量呢？因为CNN中很多的计算都是类似于y = w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + ... + w[n-1]*x[n-1]这样的点乘然后累加的形式，其中w和x是向量，结果y是标量。

在CNN中最常见的卷积层和全连接层中，w是学习到的权重值，而x是该层的输入特征图，y是该层的输出特征图。一般来说，每层输出不止一张特征图，所以我们上面的乘加计算也要做多次。这里我们约定w[0]*x[0] + ...算一次乘加运算。这样来算，像上面两个长度为n的向量w和x相乘，就有n次乘法操作和n-1次加法操作，大约可等于n次乘加操作。



