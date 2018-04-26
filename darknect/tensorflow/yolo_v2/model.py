#-*- coding: utf-8 -*-
# yolo-v2 模型文件
# dark 19  passthrough 层 跨通道合并特征
"""
YOLOv2 implemented by Tensorflow, only for predicting

1. 3*3*3*32 卷积核  3通道输入 32通道输出 步长1 + 最大值池化 
2. 3*3*32*64 卷积核  32通道输入 64通道输出 步长1 + 最大值池化 
3. 3*3 1*1 3*3 卷积 + 最大值池化
4. 3*3 1*1 3*3 卷积 + 最大值池化
5. 3*3 1*1 3*3 1*1 3*3 卷积 + 最大值池化
6. 3*3 1*1 3*3 1*1 3*3 卷积
7.  3*3  3*3 卷积
7.5 passtrough 层 尺寸减半 通道数量变为4倍 跨层 通道合并concat
8. 3*3*(1024+64*4)*1024 卷积核  1280通道输入 1024通道输出 步长1 
9. 3*3*1024* n_last_channels 卷积核  1024通道输入 n_last_channels 通道输出 步长1 检测网络输出

"""
import os

import numpy as np
import tensorflow as tf

######## basic layers #######
# 激活函数  max(0,  0.1*x)
def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.1, name="leaky_relu")

# Leaky ReLU激活函数
#def leak_relu(x, alpha=0.1):
#return tf.maximum(alpha * x, x)


# Conv2d 2d 卷积 padding延拓 + 2d卷积 + 批规范化 + 激活输出
def conv2d(x, filters, size, pad=0, stride=1, batch_normalize=1,
           activation=leaky_relu, use_bias=False, name="conv2d"):
    # 对输入通道 延拓
    if pad > 0:
        x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    # 2d 卷积
    # conv = tf.nn.conv2d(x_pad, weight, strides=[1, stride, stride, 1], padding="VALID") 需要指定4维卷积和
    # tf.layers.conv2d 可以省略指定输入通道数量
    out = tf.layers.conv2d(x, filters, size, strides=stride, padding="VALID",
                           activation=None, use_bias=use_bias, name=name)
    # 批规范化
    if batch_normalize == 1:
        out = tf.layers.batch_normalization(out, axis=-1, momentum=0.9,
                                            training=False, name=name+"_bn")
    # 激活输出
    if activation:
        out = activation(out)
    return out

# 最大值池化层 maxpool2d
def maxpool(x, size=2, stride=2, name="maxpool"):
    return tf.layers.max_pooling2d(x, size, stride)

# passtrougt 层 reorg layer  
# 按行和按列隔行采样的方法，就可以得到4个新的特征图。
def reorg(x, stride):
    return tf.extract_image_patches(x, [1, stride, stride, 1],
                        [1, stride, stride, 1], [1,1,1,1], padding="VALID")

# 网络结构  输出通道数量 默认为 5*(5+80)=425  ,默认80类
def darknet(images, n_last_channels=425):
    """Darknet19 for YOLOv2"""
    ######### 1. 3*3*3*32 卷积核  3通道输入 32通道输出 步长1 + 最大值池化 ########################
    net = conv2d(images, 32, 3, 1, name="conv1")
    net = maxpool(net, name="pool1")# 2*2的池化核 步长 2  尺寸减半
    ######### 2. 3*3*32*64 卷积核  32通道输入 64通道输出 步长1 + 最大值池化 ######################
    net = conv2d(net, 64, 3, 1, name="conv2")
    net = maxpool(net, name="pool2")# 2*2的池化核 步长 2  尺寸减半
    ######### 3. 3*3 1*1 3*3 卷积 + 最大值池化 ##################################################
    net = conv2d(net, 128, 3, 1, name="conv3_1")# 3*3*64*128 卷积核  64通道输入  128通道输出 步长1
    net = conv2d(net, 64, 1, name="conv3_2")    # 1*1*128*64 卷积核  128通道输入 64通道输出 步长1 
    net = conv2d(net, 128, 3, 1, name="conv3_3")# 3*3*64*128 卷积核  64通道输入  128通道输出 步长1
    net = maxpool(net, name="pool3")# 2*2的池化核 步长 2  尺寸减半
    ######### 4. 3*3 1*1 3*3 卷积 + 最大值池化 ###################################################
    net = conv2d(net, 256, 3, 1, name="conv4_1")# 3*3*128*256 卷积核  128通道输入  256通道输出 步长1
    net = conv2d(net, 128, 1, name="conv4_2")   # 1*1*256*128 卷积核  256通道输入  128通道输出 步长1
    net = conv2d(net, 256, 3, 1, name="conv4_3")# 3*3*128*256 卷积核  128通道输入  256通道输出 步长1
    net = maxpool(net, name="pool4")# 2*2的池化核 步长 2  尺寸减半
    ######### 5. 3*3 1*1 3*3 1*1 3*3 卷积 + 最大值池化   ##########################################
    net = conv2d(net, 512, 3, 1, name="conv5_1")# 3*3*256*512 卷积核  256通道输入  512通道输出 步长1
    net = conv2d(net, 256, 1, name="conv5_2")   # 1*1*512*256 卷积核  512通道输入  256通道输出 步长1
    net = conv2d(net, 512, 3, 1, name="conv5_3")# 3*3*256*512 卷积核  256通道输入  512通道输出 步长1
    net = conv2d(net, 256, 1, name="conv5_4")   # 1*1*512*256 卷积核  512通道输入  256通道输出 步长1
    net = conv2d(net, 512, 3, 1, name="conv5_5")# 3*3*256*512 卷积核  256通道输入  512通道输出 步长1
    shortcut = net ######## 保存大尺寸的卷积特征图###########
    net = maxpool(net, name="pool5")# 2*2的池化核 步长 2  尺寸减半
    ######### 6. 3*3 1*1 3*3 1*1 3*3 卷积   ##########################################################
    net = conv2d(net, 1024, 3, 1, name="conv6_1")# 3*3*512*1024  卷积核  512通道输入   1024通道输出 步长1
    net = conv2d(net, 512, 1, name="conv6_2")    # 1*1*1024*512  卷积核  1024通道输入  512通道输出  步长1
    net = conv2d(net, 1024, 3, 1, name="conv6_3")# 3*3*512*1024  卷积核  512通道输入   1024通道输出 步长1
    net = conv2d(net, 512, 1, name="conv6_4")    # 1*1*1024*512  卷积核  1024通道输入  512通道输出  步长1
    net = conv2d(net, 1024, 3, 1, name="conv6_5")# 3*3*512*1024  卷积核  512通道输入   1024通道输出 步长1
    ######### 7.  3*3  3*3 卷积 ######################################################################
    net = conv2d(net, 1024, 3, 1, name="conv7_1")# 3*3*1024*1024 卷积核  1024通道输入  1024通道输出 步长 
    net = conv2d(net, 1024, 3, 1, name="conv7_2")# 3*3*1024*1024 卷积核  1024通道输入  1024通道输出 步长 
    # shortcut ？？？？？？？有点问题
    shortcut = conv2d(shortcut, 64, 1, name="conv_shortcut")# 26*26*512 特征图  再卷积 1*1*512*64  64个输出？？
    shortcut = reorg(shortcut, 2)# passtrough 层 尺寸减半 通道数量变为4倍
    net = tf.concat([shortcut, net], axis=-1)# 跨层 通道合并
    ######### 8. 3*3*(1024+64*4)*1024 卷积核  1280通道输入 1024通道输出 步长1 ##########################
    net = conv2d(net, 1024, 3, 1, name="conv8")
    ######### 9. 3*3*1024* n_last_channels 卷积核  1024通道输入 n_last_channels 通道输出 步长1##########
    ######### 检测网络输出
    net = conv2d(net, n_last_channels, 1, batch_normalize=0,
                 activation=None, use_bias=True, name="conv_dec")
    return net



if __name__ == "__main__":
    x = tf.random_normal([1, 416, 416, 3])#随机 0~1之间
    model = darknet(x)#检测

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./checkpoint_dir/yolo2_coco.ckpt")#载入网络参数
        print(sess.run(model).shape)#打印结果
