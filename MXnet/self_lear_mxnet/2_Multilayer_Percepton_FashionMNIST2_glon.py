#!/usr/bin/env python
#-*- coding:utf-8 -*-
# 多层感知机 使用glon库
# 多层感知机与前面介绍的多类逻辑回归非常类似，
# 主要的区别是我们在输入层和输出层之间插入了一到多个隐含层。
# 服饰 FashionMNIST 识别  

from mxnet import gluon
## 定义模型
net = gluon.nn.Sequential()#空模型
with net.name_scope():
    net.add(gluon.nn.Flatten())##数据变形
    net.add(gluon.nn.Dense(256, activation="relu"))##中间隐含层 非线性激活
    net.add(gluon.nn.Dense(10))#输出层
net.initialize()#模型初始化

##读取数据并训练
import sys
sys.path.append('..')
from mxnet import ndarray as nd
from mxnet import autograd
import utils#包含了自己定义的一些通用函数 如下载 载入数据集等

batch_size = 256#每次训练导入的图片数量
train_data, test_data = utils.load_data_fashion_mnist(batch_size)#载入数据
##Softmax和交叉熵损失函数
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()##损失函数
## 优化训练函数 随机梯度下降
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
## 开始训练
#learning_rate = .5#学习率
epochs = 7        ##训练迭代训练集 次数
for epoch in range(epochs):
    train_loss = 0.##损失
    train_acc = 0. ##准确度
    for data, label in train_data:#训练集
        with autograd.record():#自动微分
            output = net(data)#模型输出 向前传播
            loss = softmax_cross_entropy(output, label)#计算损失
        loss.backward()#向后传播
        trainer.step(batch_size)#随机梯度下降 训练更新参数 学习率递减

        train_loss += nd.mean(loss).asscalar()#损失
        train_acc += utils.accuracy(output, label)#准确度

    test_acc = utils.evaluate_accuracy(test_data, net)#测试集测试
    print("E次数 %d. 损失: %f, 训练准确度 %f,  测试准确度%f" % (
        epoch, train_loss/len(train_data),
        train_acc/len(train_data), test_acc))

