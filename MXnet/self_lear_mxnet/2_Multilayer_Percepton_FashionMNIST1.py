#!/usr/bin/env python
#-*- coding:utf-8 -*-
# 多层感知机
# 多层感知机与前面介绍的多类逻辑回归非常类似，
# 主要的区别是我们在输入层和输出层之间插入了一到多个隐含层。
# 服饰 FashionMNIST 识别  
from mxnet import ndarray as nd
from mxnet import autograd
import random
from mxnet import gluon

import matplotlib.pyplot as plt#画图

import sys
sys.path.append('..')
import utils #包含了自己定义的一些通用函数 如下载 载入数据集等
##########################################################
#### 准备输入数据 ###
#一个稍微复杂点的数据集，它跟MNIST非常像，但是内容不再是分类数字，而是服饰
## 准备 训练和测试数据集
batch_size = 256#每次训练 输入的图片数量
train_data, test_data = utils.load_data_fashion_mnist(batch_size)


###########################################################
### 定义模型 ##################
#@@@@初始化模型参数 权重和偏置@@@@
num_inputs = 28*28##输入为图像尺寸 28*28
num_outputs = 10#输出10个标签

num_hidden = 256#定义一个只有一个隐含层的模型，这个隐含层输出256个节点
weight_scale = .01#初始化权重参数的 均匀分布均值
#输入到隐含层 权重 + 偏置
W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale)
b1 = nd.zeros(num_hidden)
#隐含层到输出层 权重 + 偏置
W2 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale)
b2 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2]#所有参数
for param in params:#参数添加自动求导
    param.attach_grad()

##非线性激活函数
#为了让我们的模型可以拟合非线性函数，我们需要在层之间插入非线性的激活函数。
def relu(X):
    return nd.maximum(X, 0)
### 定义模型
def net(X):
    X = X.reshape((-1, num_inputs))
    h1 = relu(nd.dot(X, W1) + b1)# 隐含层输出 非线性激活
    output = nd.dot(h1, W2) + b2
    return output

##Softmax和交叉熵损失函数
## softmax 回归实现  exp(Xi)/(sum(exp(Xi))) 归一化概率 使得 10类概率之和为1
#交叉熵损失函数
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

## 开始训练
learning_rate = .5#学习率
epochs = 7        ##训练迭代训练集 次数
for epoch in range(epochs):##每迭代一次训练集
    train_loss = 0.##损失
    train_acc = 0. ##准确度
    for data, label in train_data:#训练集
        with autograd.record():#自动微分
            output = net(data)#模型输出 向前传播
            loss = softmax_cross_entropy(output, label)#计算损失
        loss.backward()#向后传播
        utils.SGD(params, learning_rate/batch_size)#随机梯度下降 训练更新参数 学习率递减

        train_loss += nd.mean(loss).asscalar()#损失
        train_acc += utils.accuracy(output, label)#准确度

    test_acc = utils.evaluate_accuracy(test_data, net)#测试集测试
    print("E次数 %d. 损失: %f, 训练准确度 %f,  测试准确度%f" % (
        epoch, train_loss/len(train_data),
        train_acc/len(train_data), test_acc))

