# yolo tensorflow 实现

# pip安装
    Ubuntu/Linux 64-bit$ 
    安装 python
          sudo apt-get install python-pip python-dev

          linux 查看python安装路径,版本号安装路径：
          which python版本号:  python

    简单pip安装 
          python2：
          pip install tensorflow==1.4.0      cpu版本
          pip install tensorflow-gpu==1.4.0  gpu版本

          python3：
          pip3 install tensorflow==1.4.0
          pip3 install tensorflow-gpu==1.4.0

    复杂pip安装
          python2.7 
               安装 0.8.0    cpu版本
               sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl

               安装新 0.12.0rc1 cpu版本
               sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc1-cp27-none-linux_x86_64.whl

          python3.4
          安装 0.8.0   cpu版本
            sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl

          1.4版本   cpu版本
            sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp34-cp34m-linux_x86_64.whl


    安装新 版本前需要卸载旧版本
    sudo pip uninstall TensorFlowsudo pip uninstall protobuf 
    
# 源码安装
    最新 的软件仓库安装 不包含一些最新的功能
    ubuntu 软件仓库 https://packages.ubuntu.com/
    
    github 源码安装源码安装介绍
    http://blog.csdn.net/masa_fish/article/details/54096996
    
     源码安装的过程很好理解，大体可以分为以下四步：
     1、从 github 下载 tensorflow 的项目源代码
     2、配置Linux系统的Bazel编译环境
     
     3、使用 bazel 将Tensorflow源代码编译成 Python .whl包
     4、通过pip安装 

    注意以下操作步骤可能导致 开机进入不了 图形界面
    1、下载源码 git clone --recurse-submodules https://github.com/tensorflow/tensorflow
    
    2、步骤
    一、安装JDK81  依赖 > ubuntu(14.04)
         $ sudo add-apt-repository ppa:webupd8team/java$ sudo apt-get update
         $ sudo apt-get install oracle-java8-installer 

         <2>ubuntu(15.10)    
              安装OpenJDK 8:

              $ sudo apt-get install openjdk-8-jdk 
              如果在执行 
              sudo add-apt-repository ppa:webupd8team/java 
              命令时，提示“Cannot add PPA:xxx”，一般是因为CA证书损坏。执行如下命令修复：
              sudo apt-get install --reinstall ca-certificates

    二、配置java环境变量   sudo gedit /etc/environment  

         在打开的文件末尾添加下面一行   
         JAVA_HOME = "/usr/lib/jvm/java-8-oracle"
    
    三、 安装其他需要的包
         sudo apt-get install pkg-config zip g++ zlib1g-dev unzip

    四、 安装 Bazel Installer

         先下载安装文件 https://github.com/bazelbuild/bazel/releases
         选择合适自己系统的版本我的是  bazel-0.4.5-installer-linux-x86_64.sh
         中科大资源较快               http://rec.ustc.edu.cn/s/m0dswe
         然后切换到.sh文件存放的路径，首先 给下载的 bazel-0.4.3-installer-linux-x86_64.sh
         文件添加可执行权限：chmod a+x bazel-0.4.5-installer-linux-x86_64.sh
         然后执行该脚本文件： ./bazel-0.4.5-installer-linux-x86_64.sh --user

         主文件夹 /home/ewenwan/ 下的 /bin不能删除 并把 该目录加入 
         source /home/ewenwan/.bazel/bin/bazel-complete.bash

         安装程序会将bazel安装到$HOME/bin目录下，需要把这个目录加入PATH在 ~/.bashrc文件的末尾添加
         sudo gedit ~/.bashrcexport PATH="$PATH:$HOME/bin"

         7、 添加其他依赖 sudo apt-get install python-numpy python-dev python-wheel
         8、 安装 CUDA 和 cudnn如果不安装支持GPU的版本，此步跳过  
             http://blog.csdn.net/masa_fish/article/details/51882183
         9、 安装其他依赖$ 
             sudo apt-get install libcupti-dev
         注意以上操作步骤可能导致 开机进入不了 图形界面

         进入文字界面 后 startx 报错
         stratx -- vt1 -keeptty > ~/.xorg.log 2>&1

    五、 编译生成 tensorflow .whl包
    1、 切换到第一步下载的tensorflow目录下，
        在终端运行： ./configure
        选择python 的安装路径   which python 可查看   
        一般默认选择python 的库路径    这个变化大     
        一般为   /usr/local/lib/python2.7/dist-packages
         Do you wish to build TensorFlow with MKL support? [y/N] y
         Do you wish to download MKL LIB from the web? [Y/n] n
         Please specify the location where MKL is installed. [Default is /opt/intel/mklml]: /home/ewenwan/ME/software/tensorflow/third_party/mkl/mklml_lnx_2018.0.20170425

         Do you wish to build TensorFlow with OpenCL support? [y/N] n
         Do you wish to build TensorFlow with CUDA support? [y/N] n
         Configuration finished
    
    2、 创建Tensorflow 的whl包
         还是在 tensorflow根目录下，终端运行
             仅 CPU 支持，无 GPU 支持：$  bazel build -c opt //tensorflow/tools/pip_package:build_pip_package   
             有 GPU 支持：$              bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package 
             
    3、pip安装
        生成 pip 安装包$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    安装 生成的包    
    cd /tmp/tensorflow_pkg 
    sudo pip install tensorflow-1.2.0rc0-cp27-cp27mu-linux_x86_64.whl  (替换为生成的whl文件名)
    
    报错   tensorflow-1.2.0rc0-cp27-cp27mu-linux_x86_64.whl is not a supported wheel on this platform.
    
    解决办法   pip  版本不对    sudo pip --versionpip 1.5.4 from /usr/lib/python2.7/dist-packages (python 2.7)
    下载 pip安装文件  https://bootstrap.pypa.io/get-pip.py
    
    安装   python2.7 get-pip.py
    再次查看版本  sudo pip --versionpip 9.0.1 from /usr/local/lib/python2.7/dist-packages (python 2.7)
    再次进行安装  cd /tmp/tensorflow_pkg  sudo pip install tensorflow-1.2.0rc0-cp27-cp27mu-linux_x86_64.whl  (替换为生成的whl文件名)

