# caffe使用
[caffe 安装](https://github.com/Ewenwan/MVision/blob/master/darknect/caffe/caffe_%E5%AE%89%E8%A3%85.md)

[Caffe代码解析](http://alanse7en.github.io/caffedai-ma-jie-xi-2/)

[caffe网络模型结构在线可视化](http://ethereon.github.io/netscope/#/editor)

[CAFFE使用 源码分析等](https://blog.csdn.net/fangjin_kl/article/list/3)

[caffe 模型配置文件 prototxt 详解](https://github.com/Ewenwan/MVision/blob/master/darknect/caffe/caffe_%E7%AE%80%E4%BB%8B_%E4%BD%BF%E7%94%A8.md)

[caffe.proto 系统变量层类型参数配置文件](https://github.com/Ewenwan/MVision/blob/master/darknect/caffe/caffe.proto%E7%AE%80%E4%BB%8B.md)

[ caffe_简介_使用.md](https://github.com/Ewenwan/MVision/blob/master/darknect/caffe/caffe_%E7%AE%80%E4%BB%8B_%E4%BD%BF%E7%94%A8.md)

[Caffe使用教程_c++接口](https://github.com/Ewenwan/MVision/blob/master/darknect/caffe/Caffe%E4%BD%BF%E7%94%A8%E6%95%99%E7%A8%8B_c%2B%2B%E6%8E%A5%E5%8F%A3.md)

[caffe 模型搜集](https://github.com/SnailTyan/caffe-model-zoo)
![screenshot](https://user-images.githubusercontent.com/21311442/33640664-cbcbeff2-da6c-11e7-97c8-1ad8d7fdf4c0.png)


# prototxt文件的可视化

      1.使用在线工具netscope。

      https://ethereon.github.io/netscope/quickstart.html

      2.使用自带draw_net.py脚本。

      参考：

      http://www.cnblogs.com/zjutzz/p/5955218.html

      caffe绘制网络结构图

      http://yanglei.me/gen_proto/

      另一个在线工具。

# caffe 中 BatchNorm层  要和  Scale 层一起用才有 批规范化的效果
[参考 ](https://blog.csdn.net/Suan2014/article/details/79623495)
> 批规范化：

      1) 输入归一化 x_norm = (x-u)/std, 其中u和std是个累计计算的均值和方差。
      2）y=alpha×x_norm + beta，对归一化后的x进行比例缩放和位移。其中alpha  和beta是通过迭代学习的。
      caffe中的bn层其实只做了第一件事； 
      scale 层做了第二件事；
      scale层里为什么要设置bias_term=True，这个偏置就对应2）件事里的beta。

# 1. yolo 模型转换到 caffe下
      1.1  yolov1的caffe实现
[caffe-yolo v1 python](https://github.com/xingwangsfu/caffe-yolo)

[caffe-yolo v1  c++](https://github.com/yeahkun/caffe-yolo)

      1.2. yolov2新添了route、reorg(passtrough层)、region层(最后输入解码)，好在github上有人已经实现移植。
[移植yolo2到caffe框架](https://github.com/hustzxd/z1)

      region_layer.cpp
      region_layer.cu
      region_layer.hpp

      reorg_layer.cpp
      reorg_layer.cu
      reorg_layer.hpp

      util/ math_functions.hpp  需要修改


[caffe-yolov2](https://github.com/gklz1982/caffe-yolov2)

## 上面 的两个 caffe 的实现下载后需要修改 cudnn.hpp文件 和CMAKE文件
      \include\caffe\util\cudnn.hpp
      Makefile.config
      修改后的文件 见 文件夹

## 1.2 三个文件的作用
      1. create_yolo_prototxt.py ：  
            用来将原来的yolo的cfg文件 转成 caffe的prototxt文件，这是模型的配置文件，是描述模型的结构。
      2. create_yolo_caffemodel.py ：
            用来将yolo的weights文件转成caffe的caffemodel文件， 这是模型的参数，里面包含了各层的参数。
      3. yolo_detect.py ：这个Python程序里import了caffe，caffe的python库。
            运行这个python程序需要指定用上两个python程序转好的prototxt文件和caffemodel文件，用于初始化caffe的网络。
            并在输入的图像上得到检测结果。
            python里能够import caffe 
            你需要在caffe文件夹下make pycaffe，并设置PYTHONPATH环境变量。

### 1.2.1 yolo的cfg文件 转成 caffe的prototxt
    python create_yolo_prototxt.py yolov1_test.prototxt  yolov1.cfg
### 1.2.2 yolo的weights文件转成caffe的caffemodel
    python create_yolo_caffemodel.py -m yolov1_test.prototxt -w yolov1.weights -o yolov1.caffemodel
    python yolo_weight_to_caffemodel_v1.py -m yolov1_caffe_test.prototxt -w yolov1.weights -o yolov1_caffe.caffemodel
### 1.2.3 检测 
    python yolo_main.py -m model_filename -w weight_filename -i image_filename   
    python yolov1_caffe_main.py -m yolov1_caffe_test.prototxt -w yolov1.caffemodel -i dog.jpg

# 2. caffe 模型配置文件 prototxt 详解
[博客参考](https://blog.csdn.net/maweifei/article/details/72848185?locationNum=15&fps=1)

![](https://img-blog.csdn.net/20160327122151958?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

[caffe 模型配置文件 prototxt 详解](https://github.com/Ewenwan/MVision/blob/master/darknect/caffe/caffe_%E7%AE%80%E4%BB%8B_%E4%BD%BF%E7%94%A8.md)

[caffe 版本 yolo 过程记录](https://blog.csdn.net/u012235274/article/details/52120152)

[caffe-yolo 训练](https://blog.csdn.net/u012235274/article/details/52399425)

[YOLO算法的Caffe实现](https://blog.csdn.net/u014380165/article/details/72553074)
