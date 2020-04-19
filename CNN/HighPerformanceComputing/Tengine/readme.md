# Tengine 高性能神经网络推理引擎

[源码](https://github.com/Ewenwan/Tengine)

[Tengine 推断引擎：树莓派也能玩转深度学习](https://shumeipai.nxez.com/2018/12/07/tengine-inference-engine-raspberry-pi-deep-learning.html)


# 编译

>  安装相关工具

    sudo apt-get instal git cmake

    git 是一个版本控制系统，稍后将用来从 github 网站上下载Tengine的源码
    cmake 是一个编译工具，用来产生make过程中所需要的Makefile文件
    
> 安装支持库

sudo apt-get install libprotobuf-dev protobuf-compiler libboost-all-dev libgoogle-glog-dev libopencv-dev libopenblas-dev

    protobuf 是一种轻便高效的数据存储格式，这是caffe各种配置文件所使用的数据格式
    boost 是一个c++的扩展程序库，稍后Tengine的编译依赖于该库
    google-glog 是一个google提供的日志系统的程序库
    opencv 是一个开源的计算机视觉库
    openblas 是一个开源的基础线性代数子程序库

