#-*- coding: utf-8 -*-
# MobileNet v2 模型结构
# 深度可分解卷积
"""
1. 2D卷积块  =  2D卷积层 + BN + RELU6  3*3*3*32 步长2 32个通道输出
2. 1个自适应残差深度可拆解模块  中间层扩张倍数为1为32*1  16个通道输出
3. 2个自适应残差深度可拆解模块  中间层扩张倍数为6为16*6  24个通道输出
4. 3个自适应残差深度可拆解模块  中间层扩张倍数为6为24*6  32个通道输出 
5. 4个自适应残差深度可拆解模块  中间层扩张倍数为6为32*6  64个通道输出
6. 3个自适应残差深度可拆解模块  中间层扩张倍数为6为64*6  96个通道输出
7. 3个自适应残差深度可拆解模块  中间层扩张倍数为6为96*6  160个通道输出 
8. 1个自适应残差深度可拆解模块  中间层扩张倍数为6为160*6  320个通道输出
9. 1个 1*1点卷积块 = 1*1 PW点卷积 +  BN + RELU6 1280个通道输出
10. 全局均值 池化 average_pooling2d
11. 1*1 PW点卷积 后 展开成1维
12. softmax 分类得到分类结果
"""
import tensorflow as tf
from ops import *

## 模型 结构
def mobilenetv2(inputs, num_classes, is_train=True, reuse=False):
    exp = 6  # 扩张倍数 expansion ratio
    with tf.variable_scope('mobilenetv2'):
        ####### 1. 2D卷积块  =  2D卷积层 + BN + RELU6  3*3*3*32 步长2 32个通道输出 ############
        net = conv2d_block(inputs, 32, 3, 2, is_train, name='conv1_1') # 步长2 size/2 尺寸减半
        ####### 2. 1个自适应残差深度可拆解模块  中间层扩张倍数为1为32*1  16个通道输出 ############
        net = res_block(net, 1, 16, 1, is_train, name='res2_1')
        ####### 3. 2个自适应残差深度可拆解模块  中间层扩张倍数为6为16*6  24个通道输出 ############
        net = res_block(net, exp, 24, 2, is_train, name='res3_1')  #  步长2 size/4 尺寸减半
        net = res_block(net, exp, 24, 1, is_train, name='res3_2')
        ####### 4. 3个自适应残差深度可拆解模块  中间层扩张倍数为6为24*6  32个通道输出 ############
        net = res_block(net, exp, 32, 2, is_train, name='res4_1')  # 步长2 size/8 尺寸减半
        net = res_block(net, exp, 32, 1, is_train, name='res4_2')
        net = res_block(net, exp, 32, 1, is_train, name='res4_3')
        ####### 5. 4个自适应残差深度可拆解模块  中间层扩张倍数为6为32*6  64个通道输出 ############
        net = res_block(net, exp, 64, 1, is_train, name='res5_1')
        net = res_block(net, exp, 64, 1, is_train, name='res5_2')
        net = res_block(net, exp, 64, 1, is_train, name='res5_3')
        net = res_block(net, exp, 64, 1, is_train, name='res5_4')
        ####### 6. 3个自适应残差深度可拆解模块  中间层扩张倍数为6为64*6  96个通道输出 ############
        net = res_block(net, exp, 96, 2, is_train, name='res6_1')  # 步长2 size/16 尺寸减半
        net = res_block(net, exp, 96, 1, is_train, name='res6_2')
        net = res_block(net, exp, 96, 1, is_train, name='res6_3')
        ####### 7. 3个自适应残差深度可拆解模块  中间层扩张倍数为6为96*6  160个通道输出 ###########
        net = res_block(net, exp, 160, 2, is_train, name='res7_1')  # 步长2 size/32 尺寸减半
        net = res_block(net, exp, 160, 1, is_train, name='res7_2')
        net = res_block(net, exp, 160, 1, is_train, name='res7_3')
        ####### 8. 1个自适应残差深度可拆解模块  中间层扩张倍数为6为160*6  320个通道输出 ##########
        net = res_block(net, exp, 320, 1, is_train, name='res8_1', shortcut=False)# 不进行残差合并 f(x)
        ####### 9. 1个 1*1点卷积块 = 1*1 PW点卷积 +  BN + RELU6 1280个通道输出 ################
        net = pwise_block(net, 1280, is_train, name='conv9_1')
        ####### 10. 全局均值 池化 average_pooling2d #########################################
        net = global_avg(net)
        ####### 11. 1*1 PW点卷积 后 展开成1维 ###############################################
        logits = flatten(conv_1x1(net, num_classes, name='logits'))
        ####### 12. softmax 分类得到分类结果  ###############################################
        pred = tf.nn.softmax(logits, name='prob')
        return logits, pred
