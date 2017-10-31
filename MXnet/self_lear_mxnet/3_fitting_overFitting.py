#!/usr/bin/env python
#-*- coding:utf-8 -*-
#欠拟合和过拟合
#训练误差（模考成绩）和泛化误差（考试成绩）
#欠拟合：机器学习模型无法得到较低训练误差。
#过拟合：机器学习模型的训练误差远小于其在测试数据集上的误差。

## 一二次多项式拟合为例子
#y=1.2x−3.4x^2+5.6x^3+5.0+noise

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

num_train = 100
num_test = 100
true_w = [1.2, -3.4, 5.6]
true_b = 5.0

x = nd.random.normal(shape=(num_train + num_test, 1))#随机
X = nd.concat(x, nd.power(x, 2), nd.power(x, 3))#x  x^2  x^3
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_w[2] * X[:, 2] + true_b
y += .1 * nd.random.normal(shape=y.shape)#加入噪声

print('x:', x[:5], 'X:', X[:5], 'y:', y[:5])


### 训练
import matplotlib as mpl#画图
mpl.rcParams['figure.dpi']= 120#分辨率
import matplotlib.pyplot as plt#画图

def train(X_train, X_test, y_train, y_test):
    # 线性回归模型
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1))#全连接层 会根据输入数据维度 自动调整参数个数
    net.initialize()
    # 设一些默认参数
    learning_rate = 0.01#学习率
    epochs = 100#遍历训练集次数
    batch_size = min(10, y_train.shape[0])#每次训练样本数量
    dataset_train = gluon.data.ArrayDataset(X_train, y_train)#训练集
    data_iter_train = gluon.data.DataLoader(
        dataset_train, batch_size, shuffle=True)#每次训练的数据
    # 默认SGD和均方误差
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate})
    square_loss = gluon.loss.L2Loss()#差平方 损失
    # 保存训练和测试损失
    train_loss = []
    test_loss = []
    for e in range(epochs):#每次遍历训练集
        for data, label in data_iter_train:
            with autograd.record():#自动微分
                output = net(data)#向前传播 输出
                loss = square_loss(output, label)#损失
            loss.backward()#向后传播
            trainer.step(batch_size)#随机梯度下降 更新参数
        train_loss.append(square_loss(
            net(X_train), y_train).mean().asscalar())#训练时 的损失
        test_loss.append(square_loss(
            net(X_test), y_test).mean().asscalar())  #测试时的损失
    # 打印结果
    plt.plot(train_loss)#画图损失 变化
    plt.plot(test_loss)
    plt.legend(['train','test'])# 图例
    plt.show()#显示图
    return ('learned weight', net[0].weight.data(),#返回权重
            'learned bias', net[0].bias.data())#返回偏置
# 三阶多项式拟合（正常）
#我们先使用与数据生成函数同阶的三阶多项式拟合。
#实验表明这个模型的训练误差和在测试数据集的误差都较低。训练出的模型参数也接近真实值。
# 输入的 X为三维 模型会自动调整为三次多项式回归
print ("正常模型:")
print train(X[:num_train, :], X[num_train:, :], y[:num_train], y[num_train:])

#线性拟合（欠拟合）
#我们再试试线性拟合。很明显，该模型的训练误差很高。
#线性模型在非线性模型（例如三阶多项式）生成的数据集上容易欠拟合。
# 输入的x为一维 模型自动调整为 线性回归
print ("模型欠拟合:")
print train(x[:num_train, :], x[num_train:, :], y[:num_train], y[num_train:])


#训练量不足（过拟合）
#事实上，即便是使用与数据生成模型同阶的三阶多项式模型，如果训练量不足，该模型依然容易过拟合。
#让我们仅仅使用两个训练样本来训练。很显然，训练样本过少了，甚至少于模型参数的数量。
#这使模型显得过于复杂，以至于容易被训练数据集中的噪音影响。
#在机器学习过程中，即便训练误差很低，但是测试数据集上的误差很高。这是典型的过拟合现象。
#应对过拟合的方法，例如正则化
print ("模型过拟合:")
train(X[0:2, :], X[num_train:, :], y[0:2], y[num_train:])




