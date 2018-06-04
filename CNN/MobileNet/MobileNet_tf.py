#-*- coding: utf-8 -*-
# MobileNets模型结构
# 深度可分解卷积
# MobileNets总共28层（1 + 2 × 13 + 1 = 28）

# 核心思想：
"""
将标准卷积分解成一个深度卷积和一个点卷积（1 × 1卷积核）。深度卷积将每个卷积核应用到每一个通道，
而1 × 1卷积用来组合通道卷积的输出。后文证明，这种分解可以有效减少计算量，降低模型大小。

3 × 3 × 3 ×16 3*3的卷积 3通道输入  16通道输出
 ===== 3 × 3 × 1 × 3的深度卷积(3个3*3的卷积核，每一个卷积核对输入通道分别卷积后叠加输出) 输出3通道   1d卷积
 ===== + 1 × 1 × 3 ×16的1 ×1点卷积 1*1卷积核 3通道输入  16通道输出
参数数量 75/432 = 0.17

3*3*输入通道*输出通道 -> BN -> RELU
=======>
3*3*1*输入通道 -> BN -> RELU ->    1*1*输入通道*输出通道 -> BN -> RELU

"""

#网络结构：
"""
1. 普通3d卷积层 3*3*3*round(32 * width_multiplier) 3*3卷积核 步长2 3通道输入 输出通道数量 随机确定1~32个  卷积+NB+RELU
2. 13个 depthwise_separable_conv2d 层 3*3*1*输入通道 -> BN -> RELU ->  1*1*输入通道*输出通道 -> BN -> RELU
3. 均值池化层	 7*7核	+ squeeze 去掉维度为1的维
4. 全连接层 输出  -> [N, 1000]
5. softmax分类输出到 0~1之间
"""

import tensorflow as tf
from tensorflow.python.training import moving_averages

UPDATE_OPS_COLLECTION = "_update_ops_"
#### 使用GPU时指定 gpu设备
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

################################################################
# 创建变量 create variable 默认可优化训练
def create_variable(name, shape, initializer,
    dtype=tf.float32, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=dtype,
            initializer=initializer, trainable=trainable)
			
################################################################
# 批规范化 归一化层 BN层 减均值除方差 batchnorm layer
# s1 = W*x + b
# s2 = (s1 - s1均值)/s1方差
# s3 = beta * s2 + gamma
def bacthnorm(inputs, scope, epsilon=1e-05, momentum=0.99, is_training=True):
    inputs_shape = inputs.get_shape().as_list()# 输出 形状尺寸
    params_shape = inputs_shape[-1:]# 输入参数的长度
    axis = list(range(len(inputs_shape) - 1))

    with tf.variable_scope(scope):
        beta = create_variable("beta", params_shape,
                               initializer=tf.zeros_initializer())
        gamma = create_variable("gamma", params_shape,
                                initializer=tf.ones_initializer())
        # 均值 常量 不需要训练 for inference
        moving_mean = create_variable("moving_mean", params_shape,
                            initializer=tf.zeros_initializer(), trainable=False)
		# 方差 常量 不需要训练
        moving_variance = create_variable("moving_variance", params_shape,
                            initializer=tf.ones_initializer(), trainable=False)
    if is_training:
        mean, variance = tf.nn.moments(inputs, axes=axis)# 计算均值和方差
		# 移动平均求 均值和 方差  考虑上一次的量 xt = a * x_t-1 +(1-a)*x_now
        update_move_mean = moving_averages.assign_moving_average(moving_mean,
                                                mean, decay=momentum)
        update_move_variance = moving_averages.assign_moving_average(moving_variance,
                                                variance, decay=momentum)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_move_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_move_variance)
    else:
        mean, variance = moving_mean, moving_variance
    return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
	
################################################################
##### 实现 3*3*1*输入通道卷积
# 3*3*输入通道*输出通道 -> BN -> RELU
# =======>
# 3*3*1*输入通道 -> BN -> RELU ->    1*1*输入通道*输出通道 -> BN -> RELU
#########################
# depthwise conv2d layer
def depthwise_conv2d(inputs, scope, filter_size=3, channel_multiplier=1, strides=1):
    inputs_shape = inputs.get_shape().as_list()# 输入通道 形状尺寸 64*64* 512
    in_channels = inputs_shape[-1]#输入通道数量 最后一个参数 512
    with tf.variable_scope(scope):
        filter = create_variable("filter", shape=[filter_size, filter_size,
                                                  in_channels, channel_multiplier],
                       initializer=tf.truncated_normal_initializer(stddev=0.01))

    return tf.nn.depthwise_conv2d(inputs, filter, strides=[1, strides, strides, 1],
                                padding="SAME", rate=[1, 1])
								
								
#################################################################
# 正常的卷积层 conv2d layer 输出通道    核大小
def conv2d(inputs, scope, num_filters, filter_size=1, strides=1):
    inputs_shape = inputs.get_shape().as_list()# 输入通道 形状尺寸 64*64* 512
    in_channels = inputs_shape[-1]#输入通道数量 最后一个参数 512
    with tf.variable_scope(scope):
        filter = create_variable("filter", shape=[filter_size, filter_size,
                                                  in_channels, num_filters],
                        initializer=tf.truncated_normal_initializer(stddev=0.01))
    return tf.nn.conv2d(inputs, filter, strides=[1, strides, strides, 1],
                        padding="SAME")

################################################################
# 均值池化层 avg pool layer
def avg_pool(inputs, pool_size, scope):
    with tf.variable_scope(scope):
        return tf.nn.avg_pool(inputs, [1, pool_size, pool_size, 1],
                strides=[1, pool_size, pool_size, 1], padding="VALID")

################################################################
# 全连接层 fully connected layer
def fc(inputs, n_out, scope, use_bias=True):
    inputs_shape = inputs.get_shape().as_list()# 输入通道 形状尺寸 1*1* 512 输入时已经被展开了
    n_in = inputs_shape[-1]#输入通道数量 最后一个参数 512
    with tf.variable_scope(scope):
        weight = create_variable("weight", shape=[n_in, n_out],
                    initializer=tf.random_normal_initializer(stddev=0.01))
        if use_bias:#带偏置 与输出通道数量 同维度
            bias = create_variable("bias", shape=[n_out,],
                                   initializer=tf.zeros_initializer())
            return tf.nn.xw_plus_b(inputs, weight, bias)#带偏置 相乘
        return tf.matmul(inputs, weight)#不带偏置 相乘

################################################################
##### MobileNet模型结构定义 ####################################
class MobileNet(object):
    def __init__(self, inputs, num_classes=1000, is_training=True,
                 width_multiplier=1, scope="MobileNet"):
        """
        The implement of MobileNet(ref:https://arxiv.org/abs/1704.04861)
        :param inputs:      输入数据 4-D Tensor of [batch_size, height, width, channels]
        :param num_classes: 类别数量 ImageNet 1000 类物体 number of classes
        :param is_training: 训练模型 Boolean, whether or not the model is training
        :param width_multiplier: 宽度乘数 0~1 改变网络输入输出通道数量 float, controls the size of model
        :param scope: Optional scope for variables
        """
        self.inputs = inputs#输入数据
        self.num_classes = num_classes#类别数量
        self.is_training = is_training#训练标志
        self.width_multiplier = width_multiplier# 模型 输入输出通道数量 宽度乘数 因子

        # 定义模型结构 construct model
        with tf.variable_scope(scope):
            ######## 1. 普通3d卷积层  随机输出 通道数量 round(32 * width_multiplier) 步长2 卷积+NB+RELU
			###############  3*3*3*round(32 * width_multiplier) 3*3卷积核 3通道输入 输出通道数量 随机确定1~32个
            net = conv2d(inputs, "conv_1", round(32 * width_multiplier), filter_size=3,
                         strides=2)  # ->[N, 112, 112, 32]
            net = tf.nn.relu(bacthnorm(net, "conv_1/bn", is_training=self.is_training))# NB+RELU
			######## 2. 13个 depthwise_separable_conv2d 层 ####################
			###############  3*3*1*输入通道 -> BN -> RELU ->    1*1*输入通道*输出通道 -> BN -> RELU
			###################### a.  MobileNet 核心模块 64输出 卷积步长1 尺寸不变
            net = self._depthwise_separable_conv2d(net, 64, self.width_multiplier,
                                "ds_conv_2") # ->[N, 112, 112, 64]
			###################### b.  MobileNet 核心模块 128输出 卷积步长2 尺寸减半
            net = self._depthwise_separable_conv2d(net, 128, self.width_multiplier,
                                "ds_conv_3", downsample=True) # ->[N, 56, 56, 128]
			###################### c.  MobileNet 核心模块 128输出 卷积步长1 尺寸不变					
            net = self._depthwise_separable_conv2d(net, 128, self.width_multiplier,
                                "ds_conv_4") # ->[N, 56, 56, 128]
			###################### d.  MobileNet 核心模块 256 输出 卷积步长2 尺寸减半					
            net = self._depthwise_separable_conv2d(net, 256, self.width_multiplier,
                                "ds_conv_5", downsample=True) # ->[N, 28, 28, 256]
			###################### e.  MobileNet 核心模块 256输出 卷积步长1 尺寸不变					
            net = self._depthwise_separable_conv2d(net, 256, self.width_multiplier,
                                "ds_conv_6") # ->[N, 28, 28, 256]
			###################### f.  MobileNet 核心模块 512 输出 卷积步长2 尺寸减半							
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_7", downsample=True) # ->[N, 14, 14, 512]
			###################### g.  MobileNet 核心模块 512输出 卷积步长1 尺寸不变					
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_8") # ->[N, 14, 14, 512]
			###################### h.  MobileNet 核心模块 512输出 卷积步长1 尺寸不变					
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_9")  # ->[N, 14, 14, 512]
			###################### i.  MobileNet 核心模块 512输出 卷积步长1 尺寸不变					
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_10")  # ->[N, 14, 14, 512]
			###################### j.  MobileNet 核心模块 512输出 卷积步长1 尺寸不变					
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_11")  # ->[N, 14, 14, 512]
			###################### k.  MobileNet 核心模块 512输出 卷积步长1 尺寸不变					
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_12")  # ->[N, 14, 14, 512]
			###################### l.  MobileNet 核心模块 1024输出 卷积步长2 尺寸减半					
            net = self._depthwise_separable_conv2d(net, 1024, self.width_multiplier,
                                "ds_conv_13", downsample=True) # ->[N, 7, 7, 1024]
			###################### m.  MobileNet 核心模块 1024输出 卷积步长1 尺寸不变					
            net = self._depthwise_separable_conv2d(net, 1024, self.width_multiplier,
                                "ds_conv_14") # ->[N, 7, 7, 1024]
			######### 3. 均值池化层	 7*7核	+ squeeze 去掉维度为1的维
            net = avg_pool(net, 7, "avg_pool_15")# ->[N, 1, 1, 1024]
            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")# 去掉维度为1的维[N, 1, 1, 1024] => [N,1024]
			######### 4. 全连接层 输出  -> [N, 1000]
            self.logits = fc(net, self.num_classes, "fc_16")# -> [N, 1000]
			######### 5. softmax分类输出到 0~1之间
            self.predictions = tf.nn.softmax(self.logits)
			
    ###############################################################################
	######## MobileNet 核心模块 
	######## 3*3*1*输入通道 -> BN -> RELU ->    1*1*输入通道*输出通道 -> BN -> RELU
    def _depthwise_separable_conv2d(self, inputs, num_filters, width_multiplier,
                                    scope, downsample=False):
        """depthwise separable convolution 2D function"""
        num_filters = round(num_filters * width_multiplier)#输出通道数量
        strides = 2 if downsample else 1#下采样 确定卷积步长

        with tf.variable_scope(scope):
            ####### 1. 3*3*1*输入通道 卷积 depthwise conv2d
            dw_conv = depthwise_conv2d(inputs, "depthwise_conv", strides=strides)
            ####### 2. BN 批规范化 batchnorm
            bn = bacthnorm(dw_conv, "dw_bn", is_training=self.is_training)
            ####### 3. relu激活输出
            relu = tf.nn.relu(bn)
            ####### 4. 普通卷积 1*1*输入通道*输出通道 点卷积 1*1卷积核 pointwise conv2d (1x1)
            pw_conv = conv2d(relu, "pointwise_conv", num_filters)
            ####### 5. BN 批规范化 batchnorm
            bn = bacthnorm(pw_conv, "pw_bn", is_training=self.is_training)
			####### 6. relu激活输出
            return tf.nn.relu(bn)

if __name__ == "__main__":
    # test data
    inputs = tf.random_normal(shape=[4, 224, 224, 3])# 4张图片 224*224 大小 3通道
    mobileNet = MobileNet(inputs)# 网络模型输出
    writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph())#
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        pred = sess.run(mobileNet.predictions)#预测输出
        print(pred.shape)#打印 



