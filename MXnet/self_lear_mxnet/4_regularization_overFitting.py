#!/usr/bin/env python
#-*- coding:utf-8 -*-
#欠拟合和过拟合
#训练误差（模考成绩）和泛化误差（考试成绩）
#欠拟合：机器学习模型无法得到较低训练误差。
#过拟合：机器学习模型的训练误差远小于其在测试数据集上的误差。

#高维线性回归
# y = 0.05 + sum(0.01*xi) + noise # 这里噪音服从均值0和标准差为0.01的正态分布。

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import mxnet as mx

num_train = 20#训练集大小
num_test = 100#测试集大小
num_inputs = 200#输入神经元个数 xi的个数
#真实模型参数
true_w = nd.ones((num_inputs, 1)) * 0.01# 权重
true_b = 0.05#偏置

#生成 数据集
X = nd.random.normal(shape=(num_train + num_test, num_inputs))#输入
y = nd.dot(X, true_w) + true_b # y = 0.05 + sum(0.01*xi) 
y += .01 * nd.random.normal(shape=y.shape)#噪声 y = 0.05 + sum(0.01*xi) + noise 

X_train, X_test = X[:num_train, :], X[num_train:, :]# 0~19 行  20~99行
y_train, y_test = y[:num_train], y[num_train:]

# 不断读取数据块
import random
batch_size = 1
def data_iter(num_examples):
    idx = list(range(num_examples))
    random.shuffle(idx)#打乱
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i+batch_size,num_examples)])
        yield X.take(j), y.take(j)

#初始化模型参数
def init_params():
    w = nd.random_normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    for param in params:
        param.attach_grad()#自动求导 需要创建它们的梯度
    return params

# L2 范数正则化  损失函数中 加入 参数 平方项
def L2_penalty(w, b):
    return ((w**2).sum() + b**2) / 2 # 包括了 权重  和 偏置 也可以不加 偏置

# 定义训练和测试
import matplotlib as mpl#画图
mpl.rcParams['figure.dpi']= 120#图分辨率
import matplotlib.pyplot as plt
import numpy as np

#网络模型
def net(X, w, b):
    return nd.dot(X, w) + b

# 误差平方 损失
def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2 / 2

# 随机梯度下降算法 更新 参数
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size

# 求测试数据 误差
def test(net, params, X, y):
    return square_loss(net(X, *params), y).mean().asscalar()
    #return np.mean(square_loss(net(X, *params), y).asnumpy())

# 训练
def train(lambd):# 参数 平方 损失 比例lamdb
    epochs = 10#训练数据集次数
    learning_rate = 0.005#学习率  参数更新 步长
    w, b = params = init_params()#初始化参数
    train_loss = []#训练 损失
    test_loss = []#测试 损失
    for e in range(epochs):
        for data, label in data_iter(num_train):
            with autograd.record():#自动求导
                output = net(data, *params)#网络输出
                loss = square_loss(output, label) + lambd * L2_penalty(*params)#损失加入 参数平方项
            loss.backward()#向后传播计算 参数更新梯度
            sgd(params, learning_rate, batch_size)#更新参数
        train_loss.append(test(net, params, X_train, y_train))#训练损失
        test_loss.append(test(net, params, X_test, y_test))#测试损失
    plt.plot(train_loss)#画训练损失
    plt.plot(test_loss)#画测试损失
    plt.legend(['train', 'test'])#图例
    plt.show()#显示
    return 'learned w[:10]:', w[:10].T, 'learned b:', b#返回 训练得到的模型参数  权重 前 10个

# 不带有L2正则项
print train(0)

# 使用正则化
print train(5)









