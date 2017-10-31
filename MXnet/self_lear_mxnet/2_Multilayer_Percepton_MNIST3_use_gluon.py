#!/usr/bin/env python
#-*- coding:utf-8 -*-
# 多类别逻辑回归   gluon 实现
# 手写字体MNIST  多层感知器Multilayer Percepton (MLP)识别
# 多层神经网络

import mxnet as mx 
from mxnet import gluon, autograd, ndarray 
import numpy as np 

import sys
sys.path.append('..')
import utils #包含了自己定义的一些通用函数 如下载 载入数据集等

##########################################################
#### 准备输入数据 ###
#我们通过gluon的data.vision模块自动下载这个数据
batch_size = 256#每次训练 输入的图片数量
train_data, test_data = utils.load_data_mnist(batch_size)
'''
def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')
#下载数据
#mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
#mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)
mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test  = gluon.data.vision.MNIST(train=False, transform=transform)
# 使用gluon.data.DataLoader载入训练数据和测试数据。
# 这个DataLoader是一个iterator对象类，非常适合处理规模较大的数据集。
train_data  = gluon.data.DataLoader(mnist_train, batch_size=32, shuffle=True) 
test_data   = gluon.data.DataLoader(train_data, batch_size=32, shuffle=False)
'''

###########################################
####定义模型
# 先把模型做个初始化 
net = gluon.nn.Sequential() 
# 然后定义模型架构 
with net.name_scope(): ##线性模型就是使用对应的Dense层
    net.add(gluon.nn.Dense(128, activation="relu")) # 第一层设置128个节点 
    net.add(gluon.nn.Dense(64, activation="relu")) # 第二层设置64个节点 
    net.add(gluon.nn.Dense(10)) # 输出层 



#############################################
##### 设置参数
# 先随机设置模型参数 
# 数值从一个标准差为0.05正态分布曲线里面取 
net.collect_params().initialize(mx.init.Normal(sigma=0.05)) 


#### 使用softmax cross entropy loss算法 
# Softmax和交叉熵损失函数
# softmax 回归实现  exp(Xi)/(sum(exp(Xi))) 归一化概率 使得 10类概率之和为1
# 交叉熵损失函数  将两个概率分布的负交叉熵作为目标值，最小化这个值等价于最大化这两个概率的相似度 
# 计算模型的预测能力 
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss() 

### 优化模型 
# 使用随机梯度下降算法(sgd)进行训练 
# 并且将学习率的超参数设置为 .1 
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1}) 

epochs = 10 ##训练
for e in range(epochs):#每一次训练整个训练集
    train_loss = 0.# 损失
    train_acc = 0. #准确度
    for i, (data, label) in enumerate(train_data): ##训练集里的 每一批次样本和标签
        data = data.as_in_context(mx.cpu()).reshape((-1, 784)) ## 28*28 转成 1*784
        label = label.as_in_context(mx.cpu()) 
        with autograd.record(): # 自动求微分
            output = net(data)  # 模型输出 向前传播 
            loss = softmax_cross_entropy(output, label)## 计算误差
        loss.backward()     # 向后传播
        trainer.step(data.shape[0]) # 优化模型参数 data.shape[0] = batch_size 
        # Provide stats on the improvement of the model over each epoch 
        train_loss += ndarray.mean(loss).asscalar() ## 当前的误差损失 均值
        train_acc += utils.accuracy(output, label)  #准确度
    test_acc = utils.evaluate_accuracy(test_data, net)#验证数据集的准确度
    print("遍历训练集次数 {}. 训练误差: {}. 训练准确度: {}. 测试准确度: {}.".format(
        e, train_loss/len(train_data),train_acc/len(train_data), test_acc)) 
