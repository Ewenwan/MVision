#!/usr/bin/env python
#-*- coding:utf-8 -*-
# 多类别逻辑回归
# 手写字体MNIST  多层感知器Multilayer Percepton (MLP)识别
# 多层神经网络

import mxnet as mx
import logging

##########################################################
#### 准备输入数据 ###
mnist = mx.test_utils.get_mnist()
# (batch_size, num_channels, width, height) 这里为灰度图像num_channels=1  width=height=28
batch_size = 100#每次训练载入 100张图片
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)# 随机打乱 shuffle=True
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)


data = mx.sym.var('data')
# Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
data = mx.sym.flatten(data=data)

###########################################################
### 定义模型 ##################
##第一层##
# The first fully-connected layer and the corresponding activation function
# #(batch_size,784) * (784,128) ——————> (batch_size,128)
fc1  = mx.sym.FullyConnected(data=data, num_hidden=128)#第一层全连接层
act1 = mx.sym.Activation(data=fc1, act_type="relu")    # 第一层激活函数
##第二层##
#(batch_size,128) * (128,64) ——————> (batch_size,64)
# The second fully-connected layer and the corresponding activation function
fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 64)#第二层全连接层
act2 = mx.sym.Activation(data=fc2, act_type="relu")     #第二层激活函数
##第三层输出层
# MNIST has 10 classes
#(batch_size,64) * (64,10) ——————> (batch_size,10)
fc3  = mx.sym.FullyConnected(data=act2, num_hidden=10)#全连接
# Softmax with cross entropy loss
# softmax 回归实现  exp(Xi)/(sum(exp(Xi))) 归一化概率 使得 10类概率之和为1
# 交叉熵损失函数 将两个概率分布的负交叉熵作为目标值 - nd.pick(nd.log(yhat), y)
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')#softmax输出 归一化输出类别概率


###############################################
### 优化模型 #################################
### 训练 #######################
logging.getLogger().setLevel(logging.DEBUG)  # 打印日志消息
# 在 CPU 上训练模型
mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())
mlp_model.fit(train_iter,  # 训练数据
              eval_data=val_iter,# 测试数据
              optimizer='sgd',   # 随机梯度下降SGD训练优化模型
              optimizer_params={'learning_rate':0.1},  # 学习率
              eval_metric='acc',  # 评估时 时打印 准确度
              batch_end_callback = mx.callback.Speedometer(batch_size, 100), 
              # 在每批数据训练完成后的回调函数 打印logging信息(每经过100个batch_size打印logging) 
              #ye就是 每 训练 10000 张照片 打印一次信息 
              num_epoch=10)  # 训练整个数据集 10次

## 查看测试结果
test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size)#测试数据集 按照 batch_size分割
prob = mlp_model.predict(test_iter)#预测
assert prob.shape == (10000, 10)

test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
# predict accuracy of mlp
acc = mx.metric.Accuracy()##准确度
mlp_model.score(test_iter, acc)
print(acc)
assert acc.get()[1] > 0.96




