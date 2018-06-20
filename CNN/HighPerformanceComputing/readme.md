# HighPerformanceComputing 
      高性能计算(High performance computing， 缩写HPC) 
      指通常使用很多处理器（作为单个机器的一部分）
      或者某一集群中组织的几台计算机（作为单个计 算资源操作）的计算系统和环境。
      有许多类型的HPC 系统，其范围从标准计算机的大型集群，到高度专用的硬件。
      大多数基于集群的HPC系统使用高性能网络互连，比如那些来自 InfiniBand 或 Myrinet 的网络互连。
      基本的网络拓扑和组织可以使用一个简单的总线拓扑，
      在性能很高的环境中，网状网络系统在主机之间提供较短的潜伏期，
      所以可改善总体网络性能和传输速率。
      
# 在深度神经网络中 特指提高卷积计算方式的方法

     腾讯NCNN框架入门到应用
     
[代码](https://github.com/Ewenwan/ncnn)


## 在Ubuntu上安装NCNN
### 1. 下载编译源码
      git clone https://github.com/Tencent/ncnn.git
      下载完成后，需要对源码进行编译
            cd ncnn
            mkdir build && cd build
            cmake ..
            make -j
            make install

      执行完毕后我们可以看到:
            Install the project...
            -- Install configuration: "release"
            -- Installing: /home/ruyiwei/code/ncnn/build/install/lib/libncnn.a
            -- Installing: /home/ruyiwei/code/ncnn/build/install/include/blob.h
            -- Installing: /home/ruyiwei/code/ncnn/build/install/include/cpu.h
            -- Installing: /home/ruyiwei/code/ncnn/build/install/include/layer.h
            -- Installing: /home/ruyiwei/code/ncnn/build/install/include/mat.h
            -- Installing: /home/ruyiwei/code/ncnn/build/install/include/net.h
            -- Installing: /home/ruyiwei/code/ncnn/build/install/include/opencv.h
            -- Installing: /home/ruyiwei/code/ncnn/build/install/include/platform.h


