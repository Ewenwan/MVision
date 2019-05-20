# 高通SNPE 神经网络处理引擎（SNPE）

可运行于搭载了高通Zeroth机器智能平台的820芯片处理器，开发者可以在SNPE上搭建自己的深度学习网络模型。更详细的介绍可以登录高通SNPE相关网页了解：https://developer.qualcomm.com/software/snapdragon-neural-processing-engine

高通提供了用户定义层（UDL）功能，通过回调函数可以自定义算子，并通过重编译C++代码将自定义文件编译到可执行文件中。如果开发就是使用的C++，那比较容易实现用户定义层，但如果是运行在Android上就比较麻烦了，上层java代码需要通过JNI来调用snpe原生的C++编译好的.so文件，因为用户定义层的代码是不可能预先编译到snpe原生.so文件中的，所以用snpe提供的Java
 API是无法获得用户定义层的功能的，所以，必须重新开发SNPE的JNI。


## linux 下开发

一、下载地址

    https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk

二、配置步骤

   2.1  $ unzip -X snpe-X.Y.Z.zip

   2.2  $ source snpe-X.Y.Z/bin/dependencies.sh

   2.3  $ source snpe-X.Y.Z/bin/check_python_depends.sh

   2.4 下载caffe

      $ git clone  https://github.com/BVLC/caffe.git

      $ git checkout d8f79537977f9dbcc2b7054a9e95be00eb6f26d0 （切换到这个分支，SNPE文档如此提示）

   2.4 指定PYTHONPATH路径

      export SNPE_ROOT=/home/pengcuo/work/snpe/snpe-1.19.2
      export ANDROID_NDK_ROOT=/home/pengcuo/buff/android-ndk-r17
      export PYTHONPATH=/home/pengcuo/work/snpe/snpe-1.19.2/lib/python:/home/pengcuo/work/caffe/python:$PYTHONPATH

三、生成dlc文件

     $ ./bin/x86_64-linux-clang/snpe-caffe-to-dlc -c small.prototxt

