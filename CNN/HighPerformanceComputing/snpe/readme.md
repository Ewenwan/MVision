# 高通SNPE 神经网络处理引擎（SNPE）

[snpe-1.6.0/helper.md ](https://github.com/RuiZeWu/Android-OpenPose/blob/master/snpe-1.6.0/helper.md)


可运行于搭载了高通Zeroth机器智能平台的820芯片处理器，开发者可以在SNPE上搭建自己的深度学习网络模型。更详细的介绍可以登录高通SNPE相关网页了解：https://developer.qualcomm.com/software/snapdragon-neural-processing-engine

高通提供了用户定义层（UDL）功能，通过回调函数可以自定义算子，并通过重编译C++代码将自定义文件编译到可执行文件中。如果开发就是使用的C++，那比较容易实现用户定义层，但如果是运行在Android上就比较麻烦了，上层java代码需要通过JNI来调用snpe原生的C++编译好的.so文件，因为用户定义层的代码是不可能预先编译到snpe原生.so文件中的，所以用snpe提供的Java
 API是无法获得用户定义层的功能的，所以，必须重新开发SNPE的JNI。


> 使用SNPE，用户可以：
 
1.执行任意深度的神经网络  
2.在SnapdragonTM CPU，AdrenoTM GPU或HexagonTM DSP上执行网络。  
3.在x86 Ubuntu Linux上调试网络执行  
4.将Caffe，Caffe2，ONNXTM和TensorFlowTM模型转换为SNPE深度学习容器（DLC）文件  
5.将DLC文件量化为8位定点，以便在Hexagon DSP上运行  
6.使用SNPE工具调试和分析网络性能  
7.通过C ++或Java将网络集成到应用程序和其他代码中  


模型训练在流行的深度学习框架上进行（SNPE支持Caffe，Caffe2，ONNX和TensorFlow模型。）训练完成后，训练的模型将转换为可加载到SNPE运行时的DLC文件。 然后，可以使用此DLC文件使用其中一个Snapdragon加速计算核心执行前向推断传递。

> 基本的SNPE工作流程只包含几个步骤：

1.将网络模型转换为可由SNPE加载的DLC文件。  
2.可选择量化DLC文件以在Hexagon DSP上运行。  
3.准备模型的输入数据。  
4.使用SNPE运行时加载并执行模型。  


> 配置环境，用Snapdragon NPE SDK进行人工智能的开发需要满足一些先决条件的，具体如下所述：

1.需要运行一个卷积模型的一个或多个垂直行业，包括手机、汽车、物联网、AR，机器人，和机器人  
2.知道怎样去设置并且训练一个模型或者已经有一个训练好的模型文件。  
3.选择的framework应该是Caffe/Caffe2或者TensorFlow  
4.你做Android 的JAVA APPs或者使用Android或LInux本地的应用。  
5.需要有ubuntu 14.04的开发环境  
6.有一个支持的设备用来检测应用。


构建示例Android APP 
 
Android APP结合了Snapdragon NPE运行环境（/android/snpe-release.aar Android库提供）和 上述Caffe Alexnet示例生成的DLC模型。 

1.复制运行环境和模型，为构建APP作好准备 

•cd $SNPE_ROOT/examples/android/image-classifiers  
•cp ../../../android/snpe- release.aar ./app/libs # copies the NPE runtime library  
•bash ./setup_models.sh # packages the Alexnet example (DLC, labels, imputs) as an Android resource file  

选项A：从Android studio构建Android APK：

1.启动Android Studio。  
2.打开~/snpe-sdk/examples/android/image- classifiers文件夹中的项目。  
3.如有的话，接受Android Studio建议，升级 构建系统组件。  
4.按下“运行应用”按钮，构建并运行APK。  

选项B：从命令行构建Android APK：

•sudo apt-get install libc6:i386 libncurses5:i386 libstdc++6:i386 lib32z1  
•libbz2-1.0:i386 # Android SDK build dependencies on ubuntu  
•./gradlew assembleDebug # build the APK  

上述命令可能需要将ANDROID_HOME和JAVA_HOME 设置为系统中的Android SDK和JRE/JDK所在位置。
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

