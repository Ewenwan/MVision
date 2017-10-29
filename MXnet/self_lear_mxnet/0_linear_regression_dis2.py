#!/usr/bin/env python
#-*- coding:utf-8 -*-
## 两维 线性回归 不带 偏置
## y=a*x2 + 1*x1
# https://mxnet.incubator.apache.org/tutorials/python/linear-regression.html
## 测试 python 0_linear_regression_dis2.py
import mxnet as mx
import numpy as np
import logging
logging.getLogger().setLevel(logging.DEBUG)##日志记录调试信息
import matplotlib.pyplot as plt##画图
 
# 准备输入数据
#Training data 训练数据
train_data = np.random.uniform(0, 1, [100, 2])##100行2列 0~1均匀分布
train_label = np.array([train_data[i][0] + 2 * train_data[i][1] for i in range(100)])## y=y=a*x2+1*x1
batch_size = 1## 每次训练输入的训练数据数量

#Evaluation Data 验证评估数据
eval_data = np.array([[7,2],[6,10],[12,2]])# x1 + 2 * x2 
eval_label = np.array([11,26,16])# 7+2*2=11  6+10*2=26  12+2*2=16

## 定义 训练方式 batch_size每次训练输入的样例数量 shuffle为重新排列样例输入训练的序列
## 注意 label_name  与Y = mx.symbol.Variable('lin_reg_label')一致  默认为 softmax_label
train_iter = mx.io.NDArrayIter(train_data,train_label, batch_size, shuffle=True,label_name='lin_reg_label')
eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size, shuffle=False)

## 定义模型
X = mx.sym.Variable('data')
Y = mx.symbol.Variable('lin_reg_label')
fully_connected_layer  = mx.sym.FullyConnected(data=X, name='fc1', num_hidden = 1)##线性回归 相当于 全连接层
lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="lro")##计算损失函数 loss  l2 loss 差平方和
## SoftmaxOutput 交叉熵代价函数
model = mx.mod.Module(
    symbol = lro ,#损失函数
    data_names=['data'],## 数据
    label_names = ['lin_reg_label']#标签 线性回归 网络结构 network structure
)
#可视化模型
mx.viz.plot_network(symbol=lro)

##优化训练模型
model.fit(train_iter, eval_iter,
            optimizer_params={'learning_rate':0.005, 'momentum': 0.9},## 学习率 动量
            num_epoch=50,#训练次数
            eval_metric='mse',#校验数据所需要的评价指标 mean square error 误差平方均值
            batch_end_callback = mx.callback.Speedometer(batch_size, 2))
            # 在每批数据训练完成后的回调函数 打印logging信息(每经过2个batch_size打印logging) 


## mean squared error (MSE) 平均平方差
metric = mx.metric.MSE()#误差平方均值 评价指标
model.score(eval_iter, metric)#对验证集 进行预测后 使用评价指标进行评价

## 验证数据加入噪声
eval_data = np.array([[7,2],[6,10],[12,2]])
eval_label = np.array([11.1,26.1,16.1]) #增加0.1的噪声
eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size, shuffle=False)# 验证数据
model.score(eval_iter, metric)#再次预测并评价


## 验证模型
print model.predict(eval_iter).asnumpy()##输出预测结果





