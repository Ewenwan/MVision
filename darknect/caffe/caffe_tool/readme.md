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

