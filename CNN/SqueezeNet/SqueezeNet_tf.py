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
"""

import tensorflow as tf
import numpy as np


class SqueezeNet(object):
    def __init__(self, inputs, nb_classes=1000, is_training=True):
        # conv1
        net = tf.layers.conv2d(inputs, 96, [7, 7], strides=[2, 2],
                                 padding="SAME", activation=tf.nn.relu,
                                 name="conv1")
        # maxpool1
        net = tf.layers.max_pooling2d(net, [3, 3], strides=[2, 2],
                                      name="maxpool1")
        # fire2
        net = self._fire(net, 16, 64, "fire2")
        # fire3
        net = self._fire(net, 16, 64, "fire3")
        # fire4
        net = self._fire(net, 32, 128, "fire4")
        # maxpool4
        net = tf.layers.max_pooling2d(net, [3, 3], strides=[2, 2],
                                      name="maxpool4")
        # fire5
        net = self._fire(net, 32, 128, "fire5")
        # fire6
        net = self._fire(net, 48, 192, "fire6")
        # fire7
        net = self._fire(net, 48, 192, "fire7")
        # fire8
        net = self._fire(net, 64, 256, "fire8")
        # maxpool8
        net = tf.layers.max_pooling2d(net, [3, 3], strides=[2, 2],
                                      name="maxpool8")
        # fire9
        net = self._fire(net, 64, 256, "fire9")
        # dropout
        net = tf.layers.dropout(net, 0.5, training=is_training)
        # conv10
        net = tf.layers.conv2d(net, 1000, [1, 1], strides=[1, 1],
                               padding="SAME", activation=tf.nn.relu,
                               name="conv10")
        # avgpool10
        net = tf.layers.average_pooling2d(net, [13, 13], strides=[1, 1],
                                          name="avgpool10")
        # squeeze the axis
        net = tf.squeeze(net, axis=[1, 2])

        self.logits = net
        self.prediction = tf.nn.softmax(net)


    def _fire(self, inputs, squeeze_depth, expand_depth, scope):
        with tf.variable_scope(scope):
            squeeze = tf.layers.conv2d(inputs, squeeze_depth, [1, 1],
                                       strides=[1, 1], padding="SAME",
                                       activation=tf.nn.relu, name="squeeze")
            # squeeze
            expand_1x1 = tf.layers.conv2d(squeeze, expand_depth, [1, 1],
                                          strides=[1, 1], padding="SAME",
                                          activation=tf.nn.relu, name="expand_1x1")
            expand_3x3 = tf.layers.conv2d(squeeze, expand_depth, [3, 3],
                                          strides=[1, 1], padding="SAME",
                                          activation=tf.nn.relu, name="expand_3x3")
            return tf.concat([expand_1x1, expand_3x3], axis=3)


if __name__ == "__main__":
    inputs = tf.random_normal([32, 224, 224, 3])
    net = SqueezeNet(inputs)
    print(net.prediction)
