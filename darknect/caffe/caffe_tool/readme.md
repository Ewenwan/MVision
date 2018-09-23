# caffe 工具
[参考](https://github.com/Ewenwan/caffe_tools)

## 计算模型参数等

## 吸收BN层


## temsorflow 模型转 caffe
[temsorflow 模型](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

   dump_tensorflow_weights.py ： 
      模型优化:BatchNorm合并到卷积中， dump the weights of conv layer and batchnorm layer.
      
   load_caffe_weights.py ：
       load the dumped weights to deploy.caffemodel.
   
