#!/usr/bin/env python
#-*- coding:utf-8 -*-
## y[i] = 2 * X[i][0] - 3.4 * X[i][1] + 4.2 + noise
# https://zh.gluon.ai/chapter_supervised-learning/linear-regression-scratch.html
## 测试 
## 所以我们的第一个教程是如何只利用ndarray和autograd来实现一个线性回归的训练。
# 使用高层抽象包gluon
## 两维 线性回归 带 偏置
# y = a*x1 + b*x2 + c

from mxnet import ndarray as nd
from mxnet import autograd
import random
from mxnet import gluon

###########################################
#### 准备输入数据 ###
num_inputs = 2##数据维度
num_examples = 1000##样例大小
## 真实的需要估计的参数
true_w = [2, -3.4]##权重
true_b = 4.2##偏置
## 产生假的数据 集
X = nd.random_normal(shape=(num_examples, num_inputs))## 1000*2
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b## 1000*1
y += .01 * nd.random_normal(shape=y.shape)##加入 噪声 服从均值0和标准差为0.01的正态分布
## 读取数据
'''
batch_size = 10
def data_iter():
    # 产生一个随机索引
    idx = list(range(num_examples))
    random.shuffle(idx)##打乱
    for i in range(0, num_examples, batch_size):##0 10 20 ...
        j = nd.array(idx[i:min(i+batch_size,num_examples)])##随机抽取10个样例
        yield nd.take(X, j), nd.take(y, j)##样例和标签 我们通过python的yield来构造一个迭代器。
'''
batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)## 按 batch_size 分割数据
##读取第一个随机数据块
for data, label in data_iter:
    print(data, label)# data 为10*2  label为10*1
    break#读取一个后就结束


###########################################################
### 定义模型 ##################
'''
###初始化模型参数
w = nd.random_normal(shape=(num_inputs, 1))## 2*1权重
b = nd.zeros((1,))##偏置
params = [w, b]
#之后训练时我们需要对这些参数求导来更新它们的值，所以我们需要创建它们的梯度。
for param in params:
    param.attach_grad()
### 定义模型
# 线性模型就是将输入和模型做乘法再加上偏移
def net(X):
    return nd.dot(X, w) + b
'''
'''
当我们手写模型的时候，我们需要先声明模型参数，然后再使用它们来构建模型。
但gluon提供大量提前定制好的层，使得我们只需要主要关注使用哪些层来构建模型。
例如线性模型就是使用对应的Dense层。
构建模型最简单的办法是利用Sequential来所有层串起来。
'''
## 首先我们定义一个空的模型
net = gluon.nn.Sequential()
#然后我们加入一个Dense层，它唯一必须要定义的参数就是输出节点的个数，在线性模型里面是1.
net.add(gluon.nn.Dense(1))#线性模型就是使用对应的Dense层
#注意这里我们并没有定义说这个层的输入节点是多少，这个在之后真正给数据的时候系统会自动赋值。
##初始化模型参数
net.initialize()

#############################################
### 损失函数
# 使用常见的平方误差来衡量预测目标和真实目标之间的差距
'''
def square_loss(yhat, y):
    # 注意这里我们把y变形成yhat的形状来避免自动广播
    return (yhat - y.reshape(yhat.shape)) ** 2
'''
square_loss = gluon.loss.L2Loss()#gluon提供了平方误差函数L2Loss
###############################################
### 优化模型 #################################
# 这里通过随机梯度下降来求解 
# 模型参数沿着梯度的反方向走特定距离，这个距离一般叫学习率
'''
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad
'''
'''
同样我们无需手动实现随机梯度下降，我们可以用创建一个Trainer的实例，并且将模型参数传递给它就行。
'''
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2})

#############################################
### 训练 #######################
epochs = 6##训练迭代次数
batch_size = 10##每次训练输入的样例个数
#learning_rate = .001##学习率
for e in range(epochs):
    total_loss = 0#总的loss
    for data, label in data_iter:
        with autograd.record():##自动微分
            output = net(data)#网络模型输出
            loss = square_loss(output, label)##平方误差 损失 
        loss.backward()## 反向传播
        #SGD(params, learning_rate)##更新梯度
        trainer.step(batch_size)#更新梯度
        total_loss += nd.sum(loss).asscalar()#总的loss
    print("Epoch %d, average loss: %f" % (e, total_loss/num_examples))

## 查看训练结果
dense = net[0]#我们先从net拿到需要的层dense，然后访问其权重和位移
print true_w, dense.weight.data()
print true_b, dense.bias.data()







