# 深度神经网络压缩 Deep Compression （ICLR2016 Best Paper）

[源码注释](https://github.com/Ewenwan/DeepCompression-caffe)

## 1. caffe修改
    对pruning过程来说，可以定义一个mask来“屏蔽”修剪掉的权值，
    对于quantization过程来说，需定义一个indice来存储索引号，以及一个centroid结构来存放聚类中心。 
    在include/caffe/layer.hpp中为Layer类添加以下成员变量： 

      //vector<int> masks_;
      Blob<int> masks_;
      Blob<int> indices_;
      //vector<int> indices_;
      Blob<Dtype> centroids_;
      //vector<Dtype> centroids_;
    
    以及成员函数：
       virtual void ComputeBlobMask() {}
      
    由于只对卷积层和全连接层做压缩，因此，只需修改这两个层的对应函数即可。
    在include/caffe/layers/base_conv_layer.hpp添加成员函数 :
       virtual void ComputeBlobMask() {}
    这两处定义的函数都是基类的虚函数，不需要具体实现。
    在include/caffe/layers/conv_layer.hpp中添加成员函数声明：  
       virtual void ComputeBlobMask() {}
