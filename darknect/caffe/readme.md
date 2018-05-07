  # caffe 安装
[caffe的安装](https://blog.csdn.net/yhaolpz/article/details/71375762)

   ## 问题记录
    安装  caffe首先在你要安装的路径下 clone ：git clone https://github.com/BVLC/caffe.git1
    进入 caffe ，将 Makefile.config.example 文件复制一份并更名为 Makefile.config ，
    也可以在 caffe 目录下直接调用以下命令完成复制操作 ：

    sudo cp Makefile.config.example Makefile.config1

    复制一份的原因是编译 caffe 时需要的是 Makefile.config 文件
    ，而Makefile.config.example 只是caffe 给出的配置文件例子，不能用来编译 caffe。
    然后修改 Makefile.config 文件，
    在 caffe 目录下打开该文件：

    sudo gedit Makefile.config1

    修改 Makefile.config 文件内容：

 ### 1.应用 cudnn将
    #USE_CUDNN := 1
    修改成： 
    USE_CUDNN := 11234
### 2.应用 opencv 版本将
    #OPENCV_VERSION := 3 
    修改为： 
    OPENCV_VERSION := 3

### 3.使用 python 接口将
    #WITH_PYTHON_LAYER := 1 
    修改为 
    WITH_PYTHON_LAYER := 1
    
### 4.修改 python 路径INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
    LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib 
    修改为： 
    INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
    LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux


### 5.编译
        make all -j8

### 6.测试
        sudo make runtest -j8
        不成功多运行几次

### 7.安装python 接口
 
        sudo make pycaffe -j8
    7.1 numpy依赖错误
       python/caffe/_caffe.cpp:10:31: fatal error: numpy/arrayobject.h: 没有那个文件或目录
       解决方法：
       sudo apt-get install python-numpy

    7.2 测试  
       python
       import caffe

       ImportError: No module named caffe

       导入路径
       sudo echo export PYTHONPATH="~/caffe-master/python" >> ~/.bashrc caffe实际安装路径下python文件夹
       source ~/.bashrc


       错误2：
       ImportError: No module named skimage.io1 
       解决方法：
       pip install -U scikit-image #若没有安装pip: sudo apt install python-pip
       有时安装不了
# yolo 模型转换到 caffe下
1. yolov1的caffe实现
[caffe-yolo v1 python](https://github.com/xingwangsfu/caffe-yolo)

[caffe-yolo v1  c++](https://github.com/yeahkun/caffe-yolo)

2. yolov2新添了route、reorg、region层，好在github上有人已经实现移植。
[移植yolo2到caffe框架](https://github.com/hustzxd/z1)

[caffe-yolov2](https://github.com/gklz1982/caffe-yolov2)


## 三个文件的作用
      1. create_yolo_prototxt.py ：  用来将原来的yolo的cfg文件 转成 caffe的prototxt文件，这是模型的配置文件，是描述模型的结构。
      2. create_yolo_caffemodel.py ：用来将yolo的weights文件转成caffe的caffemodel文件， 这是模型的参数，里面包含了各层的参数。
      3. yolo_detect.py ：这个Python程序里import了caffe，caffe的python库。
                        运行这个python程序需要指定用上两个python程序转好的prototxt文件和caffemodel文件，用于初始化caffe的网络。
                        并在输入的图像上得到检测结果。
                        python里能够import caffe 
                        你需要在caffe文件夹下make pycaffe，并设置PYTHONPATH环境变量。

### yolo的cfg文件 转成 caffe的prototxt
    python create_yolo_prototxt.py
### yolo的weights文件转成caffe的caffemodel
    python create_yolo_caffemodel.py -m yolo_train_val.prototxt -w yolo.weights -o yolo.caffemodel

