#-*- coding: utf-8 -*-
# yolo-v2 模型文件
# dark 19  passthrough 层 跨通道合并特征
"""
YOLOv2 implemented by Tensorflow, only for predicting
"""
import os

import numpy as np
import tensorflow as tf

######## basic layers #######

def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.1, name="leaky_relu")

# Conv2d
def conv2d(x, filters, size, pad=0, stride=1, batch_normalize=1,
           activation=leaky_relu, use_bias=False, name="conv2d"):
    if pad > 0:
        x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    out = tf.layers.conv2d(x, filters, size, strides=stride, padding="VALID",
                           activation=None, use_bias=use_bias, name=name)
    if batch_normalize == 1:
        out = tf.layers.batch_normalization(out, axis=-1, momentum=0.9,
                                            training=False, name=name+"_bn")
    if activation:
        out = activation(out)
    return out

# maxpool2d
def maxpool(x, size=2, stride=2, name="maxpool"):
    return tf.layers.max_pooling2d(x, size, stride)

# reorg layer
def reorg(x, stride):
    return tf.extract_image_patches(x, [1, stride, stride, 1],
                        [1, stride, stride, 1], [1,1,1,1], padding="VALID")


def darknet(images, n_last_channels=425):
    """Darknet19 for YOLOv2"""
    net = conv2d(images, 32, 3, 1, name="conv1")
    net = maxpool(net, name="pool1")
    net = conv2d(net, 64, 3, 1, name="conv2")
    net = maxpool(net, name="pool2")
    net = conv2d(net, 128, 3, 1, name="conv3_1")
    net = conv2d(net, 64, 1, name="conv3_2")
    net = conv2d(net, 128, 3, 1, name="conv3_3")
    net = maxpool(net, name="pool3")
    net = conv2d(net, 256, 3, 1, name="conv4_1")
    net = conv2d(net, 128, 1, name="conv4_2")
    net = conv2d(net, 256, 3, 1, name="conv4_3")
    net = maxpool(net, name="pool4")
    net = conv2d(net, 512, 3, 1, name="conv5_1")
    net = conv2d(net, 256, 1, name="conv5_2")
    net = conv2d(net, 512, 3, 1, name="conv5_3")
    net = conv2d(net, 256, 1, name="conv5_4")
    net = conv2d(net, 512, 3, 1, name="conv5_5")
    shortcut = net
    net = maxpool(net, name="pool5")
    net = conv2d(net, 1024, 3, 1, name="conv6_1")
    net = conv2d(net, 512, 1, name="conv6_2")
    net = conv2d(net, 1024, 3, 1, name="conv6_3")
    net = conv2d(net, 512, 1, name="conv6_4")
    net = conv2d(net, 1024, 3, 1, name="conv6_5")
    # ---------
    net = conv2d(net, 1024, 3, 1, name="conv7_1")
    net = conv2d(net, 1024, 3, 1, name="conv7_2")
    # shortcut
    shortcut = conv2d(shortcut, 64, 1, name="conv_shortcut")
    shortcut = reorg(shortcut, 2)
    net = tf.concat([shortcut, net], axis=-1)
    net = conv2d(net, 1024, 3, 1, name="conv8")
    # detection layer
    net = conv2d(net, n_last_channels, 1, batch_normalize=0,
                 activation=None, use_bias=True, name="conv_dec")
    return net



if __name__ == "__main__":
    x = tf.random_normal([1, 416, 416, 3])
    model = darknet(x)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./checkpoint_dir/yolo2_coco.ckpt")
        print(sess.run(model).shape)

© 2018 GitHub, Inc.
