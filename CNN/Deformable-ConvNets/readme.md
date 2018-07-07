# Deformable-ConvNets 可变卷积（Deformable ConvNets）算法 卷积核形状可变 
[博客参考](https://www.jianshu.com/p/940d21c79aa3)


* Deformable-ConvNets  
[Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211)
[中文版](http://noahsnail.com/2017/11/29/2017-11-29-Deformable%20Convolutional%20Networks%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
[中英文对照](http://noahsnail.com/2017/11/29/2017-11-29-Deformable%20Convolutional%20Networks%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)

![](https://upload-images.jianshu.io/upload_images/4998541-ed9b39548dc4e463.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/429)

      Figure 1 展示了卷积核大小为 3x3 的正常卷积和可变形卷积的采样方式：
      (a) 所示的正常卷积规律的采样 9 个点（绿点），
      (b)(c)(d) 为可变形卷积，在正常的采样坐标上加上一个位移量（蓝色箭头），
      其中 (c)(d) 作为 (b) 的特殊情况，
      展示了可变形卷积可以作为尺度变换，比例变换和旋转变换的特殊情况

![https://upload-images.jianshu.io/upload_images/4998541-dbdcd5cf2fe67d5c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/475]

