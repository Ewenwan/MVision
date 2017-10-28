#!/usr/bin/env python
#-*- coding:utf-8 -*-
# https://mxnet.incubator.apache.org/tutorials/python/linear-regression.html
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt##画图
 
# 定义输入数据
X_data = np.linspace(-1, 1, 100)## 范围 -1，1，数量100 线性变化
noise = np.random.normal(0, 0.5, 100)## 噪声 范围 0，0.5 数量100
y_data = 5 * X_data + noise## y= a*X 插入噪声
 
# Plot 输入数据
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)##子图
ax.scatter(X_data, y_data)## 范围
 
# 定义mxnet变量
X = mx.symbol.Variable('data')
Y = mx.symbol.Variable('softmax_label')
 
# 定义网络
Y_ = mx.symbol.FullyConnected(data=X, num_hidden=1, name='pre')
loss = mx.symbol.LinearRegressionOutput(data=Y_, label=Y, name='loss')
 
# 定义优化模型
model = mx.model.FeedForward(
            ctx=mx.cpu(),
            symbol=loss,
            num_epoch=100,
            learning_rate=0.001,#学习率
            numpy_batch_size=1
        )
 
# 训练模型
model.fit(X=X_data, y=y_data)
 
# 预测
prediction = model.predict(X_data)
lines = ax.plot(X_data, prediction, 'r-', lw=5)
plt.show()

