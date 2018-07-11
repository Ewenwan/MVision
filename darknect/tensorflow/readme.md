# tensorflow  使用

[TFLearn: Deep learning library featuring a higher-level API for TensorFlow ](https://github.com/Ewenwan/tflearn)


[Learn_TensorFLow](https://github.com/Ewenwan/Learn_TensorFLow)

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
    
# 学习tensorflow 目录
```asm
* 1. [Simple Multiplication] 两个数相乘 相加 (00_multiply.py) 
* 2. [Linear Regression]     两维变量 线性回归  (01_linear_regression.py)
                             三维变量 线性回归  (01_linear_regression3.py)
       三维变量线性回归 tensorboard 显示优化记录 (01_linear_regression3_graph.py)
* 2. [Logistic Regression]   手写字体 逻辑回归(仅有权重)   (02_logistic_regression.py)
                             手写字体 逻辑回归(权重+偏置)  (02_logistic_regression2.py)
                              tensorboard 显示优化记录    (02_logistic_regression2_tf_board_graph.py
* 3. [Feedforward Neural Network] 多层感知机 无偏置               (03_net.py)
                                  多层感知机 有偏置               (03_net2.py)
* 4. [Deep Feedforward Neural Network] 多层网络 两层 隐含层无偏置 (04_modern_net.py)
                                       多层网络 两层 隐含层有偏置 (04_modern_net2.py)
* 5. [Convolutional Neural Network] 卷积神经网络 无偏置           (05_convolutional_net.py)
                                    卷积神经网络 有偏置           (05_convolutional_net2.py)
                                    tensorboard 显示优化记录      (05_convolutional_net3_board.py)
* 6. [Denoising Autoencoder]        自编码 原理与PCA相似  单层     (06_autoencoder.py)
                                    自编码 原理与PCA相似  两层     (06_autoencoder2.py)
                                    自编码 原理与PCA相似  四层     (06_autoencoder3.py)
* 7. [Recurrent Neural Network (LSTM)]长短时记忆   单一 LSTM网络   (07_lstm.py)
                                      长短时记忆   LSTM+RNN网络    (07_lstm2.py)
* 8. [Word2vec]                       单词转词向量 英文            (08_word2vec.py)
                                      单词转词向量 中文            (08_word2vec2.py)
* 9. [TensorBoard]                    tensorboard 显示优化记录专题 (09_tensorboard.py)
* 10. [Save and restore net]          保存和载入网络模型           (10_save_restore_net.py)
```
