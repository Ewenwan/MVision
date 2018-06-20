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
       查看生成了什么工具：
       我们进入 ncnn/build/tools 目录下，如下所示， 
       我们可以看到已经生成了 ncnn2mem可执行文件，以及 caffe/caffe2ncnn 和 mxnet/mxnet2ncnn 可执行文件
       caffe2ncnn的 作用是将caffe模型生成ncnn 模型 
                  .prototxt >>> .param  .caffemodel >>> .bin；
       mxnet2ncnn 的作用是将 mxnet模型生成ncnn 模型；
       ncnn2mem 是对ncnn模型进行加密。
            drwxrwxr-x 6 wanyouwen wanyouwen   4096  6月 21 00:13 ./
            drwxrwxr-x 6 wanyouwen wanyouwen   4096  6月 21 00:14 ../
            drwxrwxr-x 3 wanyouwen wanyouwen   4096  6月 21 00:13 caffe/
            drwxrwxr-x 3 wanyouwen wanyouwen   4096  6月 21 00:13 CMakeFiles/
            -rw-rw-r-- 1 wanyouwen wanyouwen   1606  6月 21 00:13 cmake_install.cmake
            -rw-rw-r-- 1 wanyouwen wanyouwen   7141  6月 21 00:13 Makefile
            drwxrwxr-x 3 wanyouwen wanyouwen   4096  6月 21 00:13 mxnet/
            -rwxrwxr-x 1 wanyouwen wanyouwen 477538  6月 21 00:13 ncnn2mem*
            drwxrwxr-x 3 wanyouwen wanyouwen   4096  6月 21 00:13 onnx
            
### 2. caffe网络模型转换为 ncnn模型 示例
#### caffe下Alexnet网络模型转换为NCNN模型
      我们在测试的过程中需要 .caffemodel文件(模型参数文件)  以及 deploy.prototxt文件(模型框架结构) ,
      所以我们再将caffe模型转换为NCNN模型的时候，
      同样也需要 .caffemodel以及deploy.prototxt这两个文件，为了方便，我们使用AlexNet为例讲解。
      
**a. 下载 caffe 模型和参数**

      alexnet 的 deploy.prototxt 可以在这里下载 https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet 
      alexnet 的 .caffemodel 可以在这里下载 http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
      
**b. 转换**

      由于NCNN提供的转换工具只支持转换新版的caffe模型,
      所以我们需要利用caffe自带的工具将旧版的caffe模型转换为新版的caffe模型后,
      再将新版本的模型转换为NCNN模型.

      旧版本caffe模型->新版本caffe模型->NCNN模型。
      
**c. 旧版caffe模型转新版caffe模型**

      模型框架转换：
      ~/code/ncnn/build/tools$ ~/caffe/build/tools/upgrade_net_proto_text deploy.prototxt new_deplpy.prototxt
      模型权重文件转换：
      ~/code/ncnn/build/tools$ ~/caffe/build/tools/upgrade_net_proto_binary bvlc_alexnet.caffemodel new_bvlc_alexnet.caffemodel
      
            上面的命令需要根据自己的caffe位置进行修改

            执行后,就可以生成新的caffe模型.

            因为我们每次检测一张图片,所以要对新生成的deploy.prototxt进行修改:第一个 dim 设为 1 一次输入的图片数量
            layer {
                  name: "data"
                  type: "Input"
                  top: "data"
                  input_param { shape: { dim: 1 dim: 3 dim: 227 dim: 227 } }
            }
            
**d. 新版caffe模型转ncnn模型**
      
      ./caffe/caffe2ncnn new_deplpy.prototxt new_bvlc_alexnet.caffemodel alexnet.param alexnet.bin
      
       caffe2ncnn的 作用是将caffe模型生成ncnn 模型 
            .prototxt >>> .param  .caffemodel >>> .bin；
            
      执行上面命令后就可以生成NCNN模型需要的param 与bin 文件.
      
**e. 对模型参数加密**
      
      ./ncnn2mem alexnet.param alexnet.bin alexnet.id.h alexnet.mem.h
      最后可以生成 alexnet.param.bin 这样的二进制加密文件.
      
**f. 模型载入**

      对于加密文件的读取也和原来不同,在源码中,
      
      非加密param读取方式为：
            ncnn::Net net;
            net.load_param("alexnet.param");
            net.load_model("alexnet.bin");

      加密param.bin读取方式为：
      ncnn::Net net;
      net.load_param_bin("alexnet.param.bin");
      net.load_model("alexnet.bin");
      
## 3. 编译NCNN例程
      前面介绍了如何将caffe模型转为NCNN模型并且加密,
      最后我们来编译NCNN的例程,
      这样可以更直观的运行或者理解NCNN. 
      
      首先我们需要进入ncnn/examples目录 
      新建一个makefile,内容如下,最重要的是,NCNN例程序只支持opencv2,不支持opencv3.
      
            NCNN = /home/wanyouwen/ewenwan/software/ncnn
            OPENCV = /home/ruyiwei/Downloads/opencv-2.4.13 #opencv路径
            INCPATH =       -I${NCNN}/build/install/include \
                            -I${OPENCV}/modules/objdetect/include \
                            -I${OPENCV}/modules/highgui/include \
                            -I${OPENCV}/modules/imgproc/include \
                            -I${OPENCV}/modules/core/include
            # 库
            LIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc  \
                            -fopenmp -pthread

            LIBPATH = -L${OPENCV}/build/lib

            %:%.cpp
                    $(CXX) $(INCPATH) $(LIBPATH) $^ ${NCNN}/build/install/lib/libncnn.a $(LIBS) -o $@
