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
