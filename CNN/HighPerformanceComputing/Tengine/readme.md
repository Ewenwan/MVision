# Tengine 高性能神经网络推理引擎

[源码](https://github.com/Ewenwan/Tengine)

[Tengine 推断引擎：树莓派也能玩转深度学习](https://shumeipai.nxez.com/2018/12/07/tengine-inference-engine-raspberry-pi-deep-learning.html)

[Tengine Winograd快速卷积算法 ](https://aijishu.com/l/1110000000021582)

[基于ARM-v8的Tengine GEMM 矩阵乘法 汇编优化 教程 ](https://github.com/Ewenwan/Tengine_gemm_tutorial)


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

> 特点

重点加速卷积等最为耗时的算子 convolution/FC/Pooling 支持多种卷积计算模式 GEMM/Direct/Winogrid

手工汇编调优，CPU微架构极致优化，Dataflow多线程加速，适配ARM A7/A17/A35/A53/A72/A73/A55/A76

支持F32/F16/Int8动态量化混合精度计算模式

> **gemm  矩阵乘法（全连接层、卷积核和输入展开后的矩阵乘法、卷积winogrid变换后的矩阵乘法）**



> **winogrid变换卷积运算**





