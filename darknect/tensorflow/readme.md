# tensorflow  使用
[深入浅出Tensorflow 中文教程](https://www.ctolib.com/docs-Tensorflow-c-index.html)

[TFLearn: Deep learning library featuring a higher-level API for TensorFlow ](https://github.com/Ewenwan/tflearn)

[Learn_TensorFLow](https://github.com/Ewenwan/Learn_TensorFLow)

[TensorFlow技术内幕（一）：导论](https://www.imooc.com/article/265350)

[参考 刘光聪 著 TensorFlow 内核剖析](https://github.com/Ewenwan/MVision/blob/master/darknect/tensorflow/TensorFlow%E5%86%85%E6%A0%B8%E5%89%96%E6%9E%90.pdf)


![](https://upload-images.jianshu.io/upload_images/12714329-6928cb6461e05052.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/565/format/webp)

    TensorFlow 是一个使用 数据流图 (Dataﬂow Graph)  表达数值计算的开源软件库。
    用 节点 表示抽象的 数学计算，并使用 OP 表达计算的逻辑；
    而 边   表示节点间 传递的数据流，
    并使用 Tensor 表达数据的表示。
    数据流图是一种有 向无环图 (DAG)，当图中的 OP 按照特定的拓扑排序依次被执行时，
    Tensor 在图中流动形成数据流， TensorFlow 因此而得名。
    
    分布式运算： 数据流图的被分裂为多个子图，
                 在一个机器内，注册的子图被二次分裂为更小的子图，
                 它们被部署在本地设备集上并发执行。
    
    TensorFlow 最初由 Google Brain (谷歌大脑) 的研究员和工程师们开发出来，
    用于开展机器学习和深度神经网络的研究，
    包括 语言识别、计算机视觉、自然语言理解、机器人、信息检索。
    
    Google Brain 构建了第一代分布式深度学习框架 DistBelief。
    于 2015.11 重磅推出第二代分布式深度学习框架 TensorFlow。
    
    Tensorflow前端支持多种开发语言，包括Python,C++，Go,Java等，出于性能的考虑，后端实现采用了C++和CUDA。

示例:

```python
import tensor flow as tf
# 生成一个1维度向量，长度为10，初始化为0
b = tf.Variable(tf.zeros([10])) 

#生成一个二维数组，大小为784x10,随机初始化 -1~1
W = tf.Variable(tf.random_uniform([784,10],-1,1)) 

# 生成输入的Placeholder，计算的时候填入输入值
x = tf.palceholder(name="x") # tf.placeholder 定义了一个占位的 OP
# 当 Session.run 时，将通过 feed_dict 的字典提供一个 mini-batch 的
# 样本数据集，从而自动推导出 tf.placeholder 的大小。


#计算最终输出
s  = tf.matmul(W,x) + b
out= tf.nn.relu(s)
#开始计算
with tf.Session() as sess:
    r = sess.run(out, feed_dict={x:input})
    print(r)

# 我们我们算法是输入是 x, 输出是 out = Relu(wx+b) .
# MNIST 拥有 50000 个训练样本，如果 batch_size 为 100，
# 则需要迭代 500 次才能完整地遍历一次训练样本数据集，常称为一个 epoch 周期。

```

![](https://upload-images.jianshu.io/upload_images/12714329-664b59dc942586c9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/330/format/webp)

1. 计算图(graph)：TensorFlow中的计算都可以表示为一个有向图(directed graph),上图就是一个计算图(Graph)。

2. 节点(node)：计算图中的每个运算操作，比如Add,Relu,Matmul就将作为节点，b,w,x,out等也是节点

3. 运算(operation)：例如Add,Relu,Matmul，运算代表一种类型的抽象运算。运算可以有自己的属性，但是所有属性必须被预先设置，或则能在计算的时候被推断出来。

4. 运算核：是一个运算在一个具体硬件(GPU,CPU,ARM等)上的实现。

5. 张量（Tensor）：张量是对Tensorflow中所有数据的抽象，张量有自己的维度和size。有自己是数据类型，比如浮点，整形等。张量沿着计算图的边流动，这也是平台名称tensorflow名字的来源。

6. 会话（session）：会话包含了计算图运行的所有上线问信息。



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
    
    TensorFlow技术内幕（二）：编译与安装
    https://www.imooc.com/article/265349

# 学习tensorflow 目录

[学习tensorflow ](https://github.com/Ewenwan/Learn_TensorFLow)

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

# 1. TensorFlow 前身 DistBelief 分析 
    DistBelief 的编程模型是基于 层 的 DAG 图。
    层可以看做是一种组合多个运算操作符的复合运算符，它完成特定的计算任务。
    例如，全连接层完成 f(W * x + b) 的复合计算，包括
    输入与权重的矩阵乘法，随后再与偏置相加，
    最后在线性加权值的基础上应用激活函数，实施非线性变换。
    
    DistBelief 使用参数服务器 (Parameter Server, 常称为 PS) 的系统架构，
    训练作业包括两个分离的进程：
        1. 无状态的 Worker 进程，用于模型的训练；
        2. 有状态的 PS 进程，用于维护模型的参数。
     
    在分布式训练过程中，
    各个模型副本异步地从 PS 上拉取训练参数 w，
    当完成一步迭代运算后，推送参数的梯度 ∆w 到 PS 上去，
    w' = w - Learning_rate * ∆w 
    并完成参数的更新。

## DistBelief 缺陷:
    但是，对于深度学习领域的高级用户， DistBelief 的编程模型，
    及其基于 PS 的系统架构，缺乏足够的灵活性和可扩展性。
    1. 优化算法：
       添加新的优化算法，必须修改 PS 的实现；
       get(), put() 的抽象方法，对某些优化算法并不高效。
    2. 训练算法：
       支持非前馈的神经网络面临巨大的挑战性；
       例如，包含循环的 RNN，交替训练的对抗网络;
       及其损失函数由分离的代理完成的增强学习模型。
    3. 加速设备：
       DistBelief 设计之初仅支持多核 CPU，并不支持多卡的 GPU，
       遗留的系统架构对支持新的计算设备缺乏良好的可扩展性。 

##  TensorFlow 设计及改进
    TensorFlow 使用数据流图表达计算过程和共享状态，
    使用节点表示抽象计算，使用边表示数据流。
    
    
### 设计原则：
    TensorFlow 的系统架构遵循了一些基本的设计原则，
    用于指导 TensorFlow 的系统实现.
    
    1. 延迟计算：
       图的构造与执行分离，并推迟计算图的执行过程；
       
    2. 原子 OP： 
       OP 是最小的抽象计算单元，支持构造复杂的网络模型；
       
    3. 抽象设备：
       支持 CPU, GPU, ASIC 多种异构计算设备类型；
       
    4. 抽象任务：
       基于任务的 PS（参数服务器），对新的优化算法和网络模型具有良好的可扩展性。
    
### 优势：
    相对于其他机器学习框架， TensorFlow 具有如下方面的优势。
    
    1. 高性能： 
       TensorFlow 升级至 1.0 版本性能提升显著，
       单机多卡 (8 卡 GPU) 环境中， Inception v3 的训练实现了 7.3 倍的加速比；
       在分布式多机多卡 (64 卡 GPU) 环境中， Inception v3 的训练实现了 58 倍的加速比；
       
    2. 跨平台：
       支持多 CPU/GPU/ASIC 多种异构设备的运算；支持台式机，服务器，移动设备等多种计算平台；
       支持 Windows， Linux， MacOS 等多种操作系统；
    3. 分布式：
       支持本地和分布式的模型训练和推理；
    4. 多语言：
       支持 Python, C++, Java, Go 等多种程序设计语言；
    5. 通用性：
       支持各种复杂的网络模型的设计和实现，包括非前馈型神经网络；
    6. 可扩展：
       支持 OP 扩展， Kernel 扩展， Device 扩展，通信协议的扩展；
    7. 可视化：
       使用 TensorBoard 可视化整个训练过程，极大地降低了 TensorFlow 的调试过程；
    8. 自动微分： 
       TensorFlow 自动构造反向的计算子图，完成训练参数的梯度计算；
    9. 工作流： 
       TensorFlow 与 TensorFlow Serving 无缝集成，支持模型的训练、导入、
       导出、发布一站式的工作流，并自动实现模型的热更新和版本管理。
    
# 2. TensorFlow 编程环境
    代码结构，工程构建，理解 TensorFlow 的系统架构
    
## 代码结构
克隆源代码：
> $ git clone git@github.com:tensorflow/tensorflow.git

切换到最新的稳定分支上。例如， r1.4 分支.

> $ cd tensorflow

> $ git checkout r1.4

查看代码结构：
>$ tree -d -L 1 ./tensorflow   目录下 一级目标列表

```
./tensorflow
├── c               C API代码
├── cc              C++ API代码            总53 万行+  C/C++ 代码
├── compiler        XLA,JIT等编译优化相关   大约为 12.5 万行，主要使用 C++ 实现
├── contrib         第三方贡献的代码
├── core            内核代码, 主要由 C++ 实现，大约拥有 26 万行代码
├── docs_src        文档相关文件
├── examples        例子相关代码
├── g3doc           TF文档
├── go              go API相关代码
├── java            java API相关代码
├── python          Python API相关代码   总37 万行+   Python提供的 API 最完善
├── stream_executor 并行计算框架代码，实现了 CUDA 和 OpenCL 的统一封装。  C++ 实现 2.5 万行代码
├── tools           辅助工具工程代码
└── user_ops        tf插件代码
```

    contrib 是第三方贡献的编程库，
    它也是 TensorFlow 标准化之前的实验性编程接口，
    犹如 Boost 社区与 C++ 标准之间的关系。
    当 contrib 的接口成熟后，便会被 TensorFlow
    标准化，并从 contrib 中搬迁至 core, python 中，


core 内核代码 目录：
> tree -d -L 1 ./tensorflow/core

```
./tensorflow/core
├── common_runtime            本地运行时，公共运行库
├── debug                     调试相关
├── distributed_runtime       分布式运行时，分布式执行模块
├── example                   例子代码
├── framework                 基础框架，基础功能模块
├── graph                     图操作，计算图相关
├── grappler                  模型优化模块 Grappler
├── kernels                   Kernel 实现，包括CPU和GPU上的实现
├── lib                       公共基础库
├── ops                       OP 定义，操作代码
├── platform                  各种平台实现相关 
├── profiler
├── protobuf                  Protobuf 定义
├── public
├── user_ops                  OP 定义
└── util                      实用函数库？
```

 Python API相关代码 目录： 大约有 18 万行代码
> tree -d -L 1 ./tensorflow/python         

```
./tensorflow/python
├── client          客户端? 是前端系统的主要组成部分
├── debug
├── estimator
├── feature_column
├── framework
├── grappler
├── kernel_tests
├── layers
├── lib
├── ops
├── platform
├── profiler
├── saved_model
├── summary
├── tools
├── training
├── user_ops
└── util
```


# 3. TensorFlow 系统架构



# 4. C API：分水岭，是衔接前后端系统的桥梁、


# 5. 计算图


# 6. 设备


# 7. 会话  Session 


 
# 8. 变量  Variable



# 9. 队列  QueueRunner  控制异步计算的强大工具 


# 10. OP 本质论 


# 11. 本地执行


# 12. 分布式 TensorFlow


# 13. BP反向传播算法 实现 

# 14. 数据加载


# 15. 模型保存 Saver 


# 16. 会话监视器 MonitoredSession



# 学习思考

    读书有三到，谓心到，眼到，口到。 - 朱熹《训学斋规》

    1. 选择
       耳到，择其善者而从之，择不善者而改之。
       取其精华去其糟粕。

    2. 抽象
       眼到，扫除外物，直觅本来也。
       一眼便能看到的都是假象，看不到，摸不着的往往才是本质。
       简洁之美
       万物理论
       忌 盲目抽象！没有调查就没有发言权
          犹如大规模的预先设计，
          畅谈客户的各种需求，
          畅谈软件设计中各种变化。

    3. 分享
       三人行必有我师焉。
       口到，传道，授业，解惑也。
       分享是一种生活的信念，明白了分享的同时，自然明白了存在的意义。
       喜欢分享知识，并将其当成一种学习动力，督促自己透彻理解问题的。

    4. 领悟
       心到，学而思之，思则得之，不思则不得也。   
       只有通过自己独立思考，归纳总结的知识，才是真正属于自己的。

       可使用图表来总结知识，一方面图的表达力远远大于文字；
       另外，通过画图也逼迫自己能够透彻问题的本质。
   
# 成长
    1. 消除重复
       代码需要消除重复，工作的习惯也要消除重复。
       不要拘于固有的工作状态，
       重复的工作状态往往使人陷入舒服的假象，
       陷入三年效应的危机。

    2. 提炼知识
       首先我们学习的不是信息，而是知识。知识是有价值的，而信息则没有价值。
       只有通过自己的筛选，提炼，总结才可能将信息转变为知识。

    3. 成为习惯
       知识是容易忘记的，只有将知识付诸于行动，并将其融汇到自己的工作状态中去，
       才能永久性地成为自己的财产。

       例如，快捷键的使用，不要刻意地去记忆，而是变成自己的一种工作习惯；
       不要去重复地劳动，使用 Shell 提供自动化程度，
       让 Shell 成为工作效率提升的利器，并将成为一种工作习惯。

    4. 更新知识
       我们需要常常更新既有的知识体系，尤其我们处在一个知识大爆炸的时代。
       终生学习。
       持续学习。
       活到老学到老。
       开放包容接纳。

       在C/Objective-C中，if、while、for之后的判断式并不需要一定传入布尔类型。
       也可以传入整型、指针等类型，只要非0就为真，并且赋值是有副作用的。比如： 
       a = 0 
       上面代码返回a的数值，这样就有可能出现将判断： 
       if ( a == 0 ) 
       错写成： 
       f ( a = 0 ) 
       为避免这个问题，有种变通写法： 
       if ( 0 == a ) 
       这种写法被称为Yoda（倒装）表达式，
       因为《星球大战》中的Yoda大师喜欢使用这样奇特的倒装句子。

    5. 重构自我
       学，然后知不足；教，然后知困。
       不要停留在原点，应该时刻重构自己的知识体系。

    6. 专攻术业
       一专多能。
       多转多能。
       人的精力是有限的，一个人不可能掌握住世界上所有的知识。
       与其在程序设计语言的抉择上犹豫不决，不如透彻理解方法论的内在本质；
       与其在众多框架中悬而未决，不如付出实际，着眼于问题本身。
       总之，博而不精，不可不防。



