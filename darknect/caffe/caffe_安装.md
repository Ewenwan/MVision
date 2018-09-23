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
    错误1：
      ImportError: No module named caffe
      导入路径
      sudo echo export PYTHONPATH="~/caffe-master/python" >> ~/.bashrc caffe实际安装路径下python文件夹
      sudo echo export PYTHONPATH="/home/wanyouwen/ewenwan/software/caffe-yolo/caffe/python" >> ~/.bashrc
      source ~/.bashrc
      
      export PYTHONPATH=$PYTHONPATH:/home/wanyouwen/ewenwan/software/caffe-yolo/caffe/python

    错误2：
      ImportError: No module named skimage.io1 
    解决方法：
    pip install -U scikit-image #若没有安装pip: sudo apt install python-pip
    有时安装不了
### 8. cudnn 新老版本 编译错误兼容性问题
[cudnn.hpp](https://github.com/Ewenwan/MVision/blob/master/darknect/caffe/caffe_src_change/cudnn.hpp)

### 9. 单机多 caffe 版本问题  编译出错 error: ‘AnnotatedDatum’ does not name a type　说明没找到定义

> 修改 caffe/Makefile

    # Complete build flags.
    # 407行附近
    # COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-isystem $(includedir)) 修改为:
      COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I $(includedir))
      
## 10. 非GPU版本 
    修改 Makefile.config： 
    # CPU-only switch (uncomment to build without GPU support).
    # CPU_ONLY := 1
    
    修改为：
    CPU_ONLY := 1
    
    
    


