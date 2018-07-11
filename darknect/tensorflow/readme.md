# tensorflow  使用

[TFLearn: Deep learning library featuring a higher-level API for TensorFlow ](https://github.com/Ewenwan/tflearn)


# tensorflow  pip安装
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
    
# tensorflow  源码安装
    最新 的软件仓库安装 不包含一些最新的功能
    ubuntu 软件仓库 https://packages.ubuntu.com/
    
    github 源码安装源码安装介绍
    http://blog.csdn.net/masa_fish/article/details/54096996
    
     
