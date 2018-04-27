#-*- coding: utf-8 -*-
# 使用的函数
"""
激活函数 relu6 
批规范化BN 减均值 除以方差
2D卷积块  =  2D卷积层 + BN + RELU6 
点卷积块  = 1*1 PW点卷积 +  BN + RELU6
DW 深度拆解卷积 depthwise_conv2d  3*3*1*输入通道卷积
自适应残差深度可拆解模块  1x1 + 3x3DW(步长为2) + 1x1 卷积 
使用函数库实现 自适应残差深度可拆解模块
全剧均值 池化
展开成1维
0 填充
"""

import tensorflow as tf

weight_decay=1e-4#

################################################################################
# 激活函数 relu6 =  min(max(x, 0), 6)  最小值0 最大值6
# relu = max(x, 0) 最小值0
def relu(x, name='relu6'):
    return tf.nn.relu6(x, name)
    
################################################################################
# 批规范化BN 减均值 除以方差
def batch_norm(x, momentum=0.9, epsilon=1e-5, train=True, name='bn'):
    return tf.layers.batch_normalization(x,
                      momentum=momentum,
                      epsilon=epsilon,
                      scale=True,
                      training=train,
                      name=name)
                      
###################################### 
# 2D卷积层           输出通道数量 核尺寸    stride尺寸   初始权重方差
def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name='conv2d', bias=False):
    with tf.variable_scope(name):
        # 正态分布初始化权重  [核高 宽 输入通道 输出通道]
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        # 2d卷积
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        # 偏置
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv
        
################################################################################
# 2D卷积块  =  2D卷积层 + BN + RELU6 
def conv2d_block(input, out_dim, k, s, is_train, name):
    with tf.name_scope(name), tf.variable_scope(name):
        net = conv2d(input, out_dim, k, k, s, s, name='conv2d')#卷积
        net = batch_norm(net, train=is_train, name='bn')# 批规范化
        net = relu(net)#激活
        return net
        
########################### 
# 1*1 PW点卷积 1*1的卷积核
def conv_1x1(input, output_dim, name, bias=False):
    with tf.name_scope(name):
        return conv2d(input, output_dim, 1,1,1,1, stddev=0.02, name=name, bias=bias)
        
################################################################################       
# 点卷积块 = 1*1 PW点卷积 +  BN + RELU6
def pwise_block(input, output_dim, is_train, name, bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        out=conv_1x1(input, output_dim, bias=bias, name='pwb')
        out=batch_norm(out, train=is_train, name='bn')# 批规范化
        out=relu(out)# 激活
        return out
        
################################################################################
#####DW 深度拆解卷积 depthwise_conv2d  3*3*1*输入通道卷积#
def dwise_conv(input, k_h=3, k_w=3, channel_multiplier= 1, strides=[1,1,1,1],
               padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
    with tf.variable_scope(name):
        in_channel=input.get_shape().as_list()[-1]# 输入通道数量
        # 正态分布初始化权重  [核高 宽 输入通道 输出通道]
        w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
                        regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        # depthwise_conv2d  3*3*1*输入通道卷积 单个卷积核对所有输入通道卷积后合并
        conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None,name=None,data_format=None)
        # 偏置
        if bias:
            biases = tf.get_variable('bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv
        
################################################################################   
# 1. 步长为1结合x shortcut
# ___________________________________>
#  |                                     -->  f(x) + x  可能需要对 x做卷积调整 使得 通道一直 可以直接合并
#  x-----> 1x1 + 3x3DW + 1x1 卷积 ----->  
#      “扩张”→“卷积提特征”→ “压缩”
#  ResNet是：压缩”→“卷积提特征”→“扩张”，MobileNetV2则是Inverted residuals,即：“扩张”→“卷积提特征”→ “压缩”
# 
# 2. 步长为2时不结合x 
# x-----> 1x1 + 3x3DW(步长为2) + 1x1 卷积 ----->   输出
# 自适应残差深度可拆解模块
def res_block(input, expansion_ratio, output_dim, stride, is_train, name, bias=False, shortcut=True):
    with tf.name_scope(name), tf.variable_scope(name):
        ######## 1. pw 1*1 点卷积 #########################
        bottleneck_dim=round(expansion_ratio*input.get_shape().as_list()[-1])# 中间输出 通道数量随机
        net = conv_1x1(input, bottleneck_dim, name='pw', bias=bias)#点卷积
        net = batch_norm(net, train=is_train, name='pw_bn')# 批规范化
        net = relu(net)# 激活
        ######## 2. dw 深度拆解卷积 depthwise_conv2d  3*3*1*输入通道卷积 ##############
        net = dwise_conv(net, strides=[1, stride, stride, 1], name='dw', bias=bias)
        net = batch_norm(net, train=is_train, name='dw_bn')# 批规范化
        net = relu(net)# 激活
        ######## 3. 1*1 点卷积 pw & linear 无非线性激活relu ############
        net = conv_1x1(net, output_dim, name='pw_linear', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_linear_bn')

        # element wise add, only for stride==1
        if shortcut and stride == 1:# 需要叠加 x 即 需要残差结构
            in_dim = int(input.get_shape().as_list()[-1]) # 输入通道 数量
            if in_dim != output_dim:   # f(x) 和 x通道不一致  f(x) + w*x
                ins = conv_1x1(input, output_dim, name='ex_dim')
                net = ins + net # f(x) + w*x
            else: # f(x) 和 x通道一致
                net = input + net# f(x) + x
        # 不需要残差结构 直接输出 f(x)
        return net#
################################################################################   
# 使用函数库实现 自适应残差深度可拆解模块
def separable_conv(input, k_size, output_dim, stride, pad='SAME', channel_multiplier=1, name='sep_conv', bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        in_channel = input.get_shape().as_list()[-1]# 输入通道数量
        # dw 深度拆解卷积核参数
        dwise_filter = tf.get_variable('dw', [k_size, k_size, in_channel, channel_multiplier],
                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        # pw 1*1 点卷积核参数
        pwise_filter = tf.get_variable('pw', [1, 1, in_channel*channel_multiplier, output_dim],
                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        strides = [1,stride, stride,1]
        # 使用函数库实现 自适应残差深度可拆解模块
        conv=tf.nn.separable_conv2d(input,dwise_filter,pwise_filter,strides,padding=pad, name=name)
        # 偏置
        if bias:
            biases = tf.get_variable('bias', [output_dim],initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv
################################################################################   
# 全剧均值 池化
def global_avg(x):
    with tf.name_scope('global_avg'):
        net=tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
        return net
################################################################################           
# 展开成1维
def flatten(x):
    #flattened=tf.reshape(input,[x.get_shape().as_list()[0], -1])  # or, tf.layers.flatten(x)
    return tf.contrib.layers.flatten(x)
################################################################################   
# 0 填充
def pad2d(inputs, pad=(0, 0), mode='CONSTANT'):
    paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
    net = tf.pad(inputs, paddings, mode=mode)
    return net
