#-*- coding: utf-8 -*-
# 论文
# https://arxiv.org/pdf/1602.07360.pdf
# 论文源码  caffe model
# https://github.com/DeepScale/SqueezeNet 
"""
2018/04/27
SqueezeNet的工作为以下几个方面：
   1. 提出了新的网络架构Fire Module，通过减少参数来进行模型压缩
   2. 使用其他方法对提出的SqeezeNet模型进行进一步压缩
   3. 对参数空间进行了探索，主要研究了压缩比和3∗3卷积比例的影响
   
Fire Module 结构
                                 ----->  1 * 1卷积 RELU -----|
             输入----->1 * 1卷积（全部） RELU                   ---> concat 通道扩展 -------> 输出
                                 ----->  3 * 3卷积 RELU  ----|
网络结构                                
0. conv1  7*7*3*96 7*7卷积 3通道输入 96通道输出 滑动步长2  relu激活                                 
1. 最大值池化 maxpool1 3*3池化核尺寸 滑动步长2           
2. fire2 squeeze层 16个输出通道， expand层  64个输出通道
3. fire3 squeeze层 16个输出通道， expand层  64个输出通道
4. fire4 squeeze层 32个输出通道， expand层  128个输出通道
5. maxpool4 最大值池化 maxpool1 3*3池化核尺寸 滑动步长2
6. fire5 squeeze层 32个输出通道， expand层  128个输出通道
7. fire6 squeeze层 48个输出通道， expand层  192个输出通道
8. fire7 squeeze层 48个输出通道， expand层  196个输出通道
9. fire8 squeeze层 64个输出通道， expand层  256个输出通道
10. maxpool8 最大值池化 maxpool1 3*3池化核尺寸 滑动步长2
11. fire9 squeeze层 64个输出通道， expand层  256个输出通道
12. 随机失活层 dropout 神经元以0.5的概率不输出
13. conv10 类似于全连接层 1*1的点卷积 将输出通道 固定为 1000类输出 + relu激活
14. avgpool10 13*13的均值池化核尺寸 13*13*1000 ---> 1*1*1000
15. softmax归一化分类概率输出 
"""

import tensorflow as tf
import numpy as np


class SqueezeNet(object):
    def __init__(self, inputs, nb_classes=1000, is_training=True):
        ######## 0. conv1  7*7*3*96 7*7卷积 3通道输入 96通道输出 滑动步长2  relu激活 ###############
        net = tf.layers.conv2d(inputs, 96, [7, 7], strides=[2, 2],
                                 padding="SAME", activation=tf.nn.relu,
                                 name="conv1")#### 224*224*3 >>>> 112*112*96
        ######## 1. 最大值池化 maxpool1 3*3池化核尺寸 滑动步长2
        net = tf.layers.max_pooling2d(net, [3, 3], strides=[2, 2],
                                      name="maxpool1")## 112*112*96 >>>> 56*56*96
        ######## 2. fire2 squeeze层 16个输出通道， expand层  64个输出通道 #########################
        net = self._fire(net, 16, 64, "fire2")#### 56*56*96  >>>> 56*56*128   64+64=128 
        ######## 3. fire3 squeeze层 16个输出通道， expand层  64个输出通道 #########################
        net = self._fire(net, 16, 64, "fire3")#### 56*56*128 >>>> 56*56*128
        ######## 4. fire4 squeeze层 32个输出通道， expand层  128个输出通道 #######################
        net = self._fire(net, 32, 128, "fire4")### 56*56*128 >>>> 56*56*256   128+128=256
        ######## 5. maxpool4 最大值池化 maxpool1 3*3池化核尺寸 滑动步长2
        net = tf.layers.max_pooling2d(net, [3, 3], strides=[2, 2],
                                      name="maxpool4")## 56*56*256 >>> 28*28*256 
        ######## 6. fire5 squeeze层 32个输出通道， expand层  128个输出通道 #######################
        net = self._fire(net, 32, 128, "fire5")### 28*28*256 >>> 28*28*256
        ######## 7. fire6 squeeze层 48个输出通道， expand层  192个输出通道 #########################
        net = self._fire(net, 48, 192, "fire6")### 28*28*256 >>> 28*28*384    192+192=384
        ######## 8. fire7 squeeze层 48个输出通道， expand层  196个输出通道 #########################
        net = self._fire(net, 48, 192, "fire7")### 28*28*584 >>> 28*28*384
        ######## 9. fire8 squeeze层 64个输出通道， expand层  256个输出通道 #########################
        net = self._fire(net, 64, 256, "fire8")### 28*28*584 >>> 28*28*512    256+256=512
        ######## 10. maxpool8 最大值池化 maxpool1 3*3池化核尺寸 滑动步长2
        net = tf.layers.max_pooling2d(net, [3, 3], strides=[2, 2],
                                      name="maxpool8")## 28*28*512 >>> 14*14*512
        ######## 11. fire9 squeeze层 64个输出通道， expand层  256个输出通道 #########################
        net = self._fire(net, 64, 256, "fire9")
        ######## 12. 随机失活层 dropout 神经元以0.5的概率不输出######################################
        net = tf.layers.dropout(net, 0.5, training=is_training)
        ######## 13. conv10 类似于全连接层 1*1的点卷积 将输出通道 固定为 1000类输出 + relu激活 ########
        net = tf.layers.conv2d(net, 1000, [1, 1], strides=[1, 1],
                               padding="SAME", activation=tf.nn.relu,
                               name="conv10")
        ######## 14. avgpool10 13*13的均值池化核尺寸 13*13*1000 ---> 1*1*1000
        net = tf.layers.average_pooling2d(net, [13, 13], strides=[1, 1],
                                          name="avgpool10")
        # squeeze the axis  1*1*1000 ---> 1*1000
        net = tf.squeeze(net, axis=[1, 2])

        self.logits = net#逻辑值
        ######### 15. softmax归一化分类概率输出 ###################################################
        self.prediction = tf.nn.softmax(net)# softmax归一化分类概率输出

    # Fire Module 结构
    #                             ----->  1 * 1卷积------|
    #         输入----->1 * 1卷积（全部）                   ---> concat 通道合并 -------> 输出
    #                             ----->  3 * 3卷积  ----|
    def _fire(self, inputs, squeeze_depth, expand_depth, scope):
        with tf.variable_scope(scope):
            # squeeze 层 1 * 1卷积 + relu
            squeeze = tf.layers.conv2d(inputs, squeeze_depth, [1, 1],
                                       strides=[1, 1], padding="SAME",
                                       activation=tf.nn.relu, name="squeeze")
            # expand  层 1*1 卷积 +relu    3*3卷积  + relu
            expand_1x1 = tf.layers.conv2d(squeeze, expand_depth, [1, 1],
                                          strides=[1, 1], padding="SAME",
                                          activation=tf.nn.relu, name="expand_1x1")
            expand_3x3 = tf.layers.conv2d(squeeze, expand_depth, [3, 3],
                                          strides=[1, 1], padding="SAME",
                                          activation=tf.nn.relu, name="expand_3x3")
            # 通道扩展 concat
            return tf.concat([expand_1x1, expand_3x3], axis=3)


if __name__ == "__main__":
    # 随机初始化测试数据  32张图片 224*224尺寸 3通道
    inputs = tf.random_normal([32, 224, 224, 3])
    # 经过 SqueezeNet网络得到输出
    net = SqueezeNet(inputs)
    # 打印预测结果 
    print(net.prediction)
