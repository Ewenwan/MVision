#-*- coding:utf-8 -*-
# 各种网络层实现 
#
"""
1. 卷积操作
2. 卷积层  卷积 + BN + RELU + droupout + maxpooling
3. 分组卷积层 每组输入通道数量平分  输出通道数量平分 卷积后 各通道 concat通道扩展合并 + BN + relu激活
4. 通道重排 channel_shuffle 分组再分组 取每个组中的一部分重新排序
5. 逐通道卷积操作 逐通道卷积 每个卷积核 只和输入数据的一个通道卷积
6. 逐通道卷积层  逐通道卷积 + BN 批规范化 + 激活
7. ShuffleNet 核心模块 1x1分组点卷积 + 通道重排 + 3x3DW + 1x1分组点卷积
8. 全连接层之前 最后的卷积之后 摊平操作 (N,H,W,C)----> (N,D)   D = H*W*C
9. 全连接 操作 (N,D)*(D,output_dim) + Baise --> (N,output_dim)
10. 全连接层 全链接 + BN + 激活 + 随机失活dropout
11. 最大值池化
12. 均值池化
13. 权重参数初始化  可带有  L2 正则项
14. 参数记录 均值 方差  最大值 最小值 直方图
"""
import tensorflow as tf
import numpy as np


################################################################################################ 
# 1.  卷积操作  Convolution layer Methods########################################################
def __conv2d_p(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    """
    Convolution 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C). 批大小 图像尺寸 通道数量
    :param w: (tf.tensor) pretrained weights (if None, it means no pretrained weights)
    :param num_filters: (integer) No. of filters (This is the output depth) 输出通道大小 卷积核数量
    :param kernel_size: (integer tuple) The size of the convolving kernel.  卷积核大小
    :param padding: (string) The amount of padding required.                填充
    :param stride: (integer tuple) The stride required.                     卷积步长
    :param initializer: (tf.contrib initializer)  normal or Xavier normal are recommended. 初始化器
    :param l2_strength:(weight decay) (float) L2 regularization parameter.  L2正则化系数
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)#偏置
    :return out: The output of the layer. (N, H', W', num_filters)
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]# 卷积步长
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]# 卷积核尺寸 

        with tf.name_scope('layer_weights'):# 初始化权重
            if w == None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)# 初始化
            __variable_summaries(w)# 记录参数
        with tf.name_scope('layer_biases'):# 初始化偏置
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
            __variable_summaries(bias)# 记录参数
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(x, w, stride, padding)# 卷积
            out = tf.nn.bias_add(conv, bias)# 添加偏置
    return out
###########################################################################################
# 2. 卷积层  卷积 + BN + RELU + droupout + maxpooling########################################
def conv2d(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,
           activation=None, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1,
           is_training=True):
    """
    This block is responsible for a convolution 2D layer followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param max_pool_enabled:  (boolean) for enabling max-pooling 2x2 to decrease width and height by a factor of 2.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return: The output tensor of the layer (N, H', W', C').
    """
    with tf.variable_scope(name) as scope:
        # 2d卷积
        conv_o_b = __conv2d_p('conv', x=x, w=w, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                              padding=padding,
                              initializer=initializer, l2_strength=l2_strength, bias=bias)
        # BN + 激活
        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=1e-5)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)
                
        # droupout 随机失活
        def dropout_with_keep():
            return tf.nn.dropout(conv_a, dropout_keep_prob)
        # 全部 激活
        def dropout_no_keep():
            return tf.nn.dropout(conv_a, 1.0)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr
        # 最大值池化
        if max_pool_enabled:
            conv_o = max_pool_2d(conv_o_dr)

    return conv_o
  
#######################################################################################
# 3. 分组卷积层 ###########################################################################
## 分组卷积 每组输入通道数量平分  输出通道数量平分 卷积后 各通道 concat通道扩展合并 + BN + relu激活
def grouped_conv2d(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                   initializer=tf.contrib.layers.xavier_initializer(), num_groups=1, l2_strength=0.0, bias=0.0,
                   activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
                   is_training=True):
    with tf.variable_scope(name) as scope:
        sz = x.get_shape()[3].value // num_groups# 每组 通道数量 = 通道总数量/ 分组数
        # 分组卷积 每组输入通道数量平分  输出通道数量平分
        conv_side_layers = [
            conv2d(name + "_" + str(i), x[:, :, :, i * sz:i * sz + sz], w, num_filters // num_groups, kernel_size,
                   padding,
                   stride,
                   initializer,
                   l2_strength, bias, activation=None,
                   batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=dropout_keep_prob,
                   is_training=is_training) for i in range(num_groups)]
        conv_g = tf.concat(conv_side_layers, axis=-1)# 组 间 通道扩展合并
        # BN 批规范化 + 激活
        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_g, training=is_training, epsilon=1e-5)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_g
            else:
                conv_a = activation(conv_g)

        return conv_a
      
##########################################################################
# 4. 通道重排 channel_shuffle ########################################
##  |||||| |||||| ||||||  分组
##  || || ||   || || ||    || || || 分组再分组
## 取每个组中的一部分重新排序
def channel_shuffle(name, x, num_groups):
    with tf.variable_scope(name) as scope:
        n, h, w, c = x.shape.as_list()# 批数量  特征图尺寸  通道总数量
        x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])# 分组 再 分组
        x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
        output = tf.reshape(x_transposed, [-1, h, w, c])
        return output

########################################################################    
# 5. 逐通道卷积操作 depthwise_conv####################################### 
# 逐通道卷积 每个卷积核 只和输入数据的一个通道卷积
def __depthwise_conv2d_p(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                         initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]# 卷积步长
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], 1]
        #  初始化权重
        with tf.name_scope('layer_weights'):
            if w is None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            __variable_summaries(w)
        # 初始化偏置
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [x.shape[-1]], initializer=tf.constant_initializer(bias))
            __variable_summaries(bias)
        # 逐通道卷积 + 偏置
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.depthwise_conv2d(x, w, stride, padding)# 逐通道卷积
            out = tf.nn.bias_add(conv, bias)# 偏置
    return out
  
###############################################################################
# 6. 逐通道卷积层 ########################################################
# 逐通道卷积 + BN 批规范化 + 激活
def depthwise_conv2d(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                     initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0, activation=None,
                     batchnorm_enabled=False, is_training=True):
    with tf.variable_scope(name) as scope:
        # 逐通道卷积操作 DW CONV
        conv_o_b = __depthwise_conv2d_p(name='conv', x=x, w=w, kernel_size=kernel_size, padding=padding,
                                        stride=stride, initializer=initializer, l2_strength=l2_strength, bias=bias)
        # BN 批规范化 + 激活
        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=1e-5)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)
    return conv_a


############################################################################################################
# 7. ShuffleNet unit methods 核心模块
# 
'''
ShuffleNet 普通点卷积 变为 分组点卷积+通道重排   逐通道卷积
普通点卷积时间还是较长

    版本1：
    ___________________________________________________________>
    |                                                            ADD -->  f(x) + x
    x-----> 1x1分组点卷积 + 通道重排 + 3x3DW + 1x1分组点卷积 ----->
    
    
    版本2：（特征图降采样）
    _____________________3*3AvgPool_________________________________>
    |                                                               concat -->  f(x) 链接  x
    x-----> 1x1分组点卷积 + 通道重排 + 3x3DW步长2 + 1x1分组点卷积 ----->  
'''    
############################################################################################################
def shufflenet_unit(name, x, w=None, num_groups=1, group_conv_bottleneck=True, num_filters=16, stride=(1, 1),
                    l2_strength=0.0, bias=0.0, batchnorm_enabled=True, is_training=True, fusion='add'):
    # Paper parameters. If you want to change them feel free to pass them as method parameters.
    activation = tf.nn.relu# relu 激活 max(0,x) 

    with tf.variable_scope(name) as scope:
        residual = x# 残差模块的 x
        bottleneck_filters = (num_filters // 4) if fusion == 'add' else (num_filters - residual.get_shape()[
            3].value) // 4
        # 分组卷积 ##############
        if group_conv_bottleneck:
            # 1*1 分组点卷积
            bottleneck = grouped_conv2d('Gbottleneck', x=x, w=None, num_filters=bottleneck_filters, kernel_size=(1, 1),
                                        padding='VALID',
                                        num_groups=num_groups, l2_strength=l2_strength, bias=bias,
                                        activation=activation,
                                        batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            # 通道重排
            shuffled = channel_shuffle('channel_shuffle', bottleneck, num_groups)
        else:
            # 普通 点卷积
            bottleneck = conv2d('bottleneck', x=x, w=None, num_filters=bottleneck_filters, kernel_size=(1, 1),
                                padding='VALID', l2_strength=l2_strength, bias=bias, activation=activation,
                                batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            shuffled = bottleneck
        # 填充
        padded = tf.pad(shuffled, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        # 逐通道卷积层
        depthwise = depthwise_conv2d('depthwise', x=padded, w=None, stride=stride, l2_strength=l2_strength,
                                     padding='VALID', bias=bias,
                                     activation=None, batchnorm_enabled=batchnorm_enabled, is_training=is_training)
        # 逐通道卷积层 步长为2 下采样模式
        if stride == (2, 2):
            # 残差通道也需要降采样 使用 3*3均值池化核 2*2步长下采样
            residual_pooled = avg_pool_2d(residual, size=(3, 3), stride=stride, padding='SAME')
        else:
            # 非下采样模式 特征图尺寸不变
            residual_pooled = residual
            
        # 再次通过 1*1分组点卷积 +  和残差通路 通道扩展合并 + relu激活
        if fusion == 'concat':
            group_conv1x1 = grouped_conv2d('Gconv1x1', x=depthwise, w=None,
                                           num_filters=num_filters - residual.get_shape()[3].value,
                                           kernel_size=(1, 1),
                                           padding='VALID',
                                           num_groups=num_groups, l2_strength=l2_strength, bias=bias,
                                           activation=None,
                                           batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            return activation(tf.concat([residual_pooled, group_conv1x1], axis=-1))
        # 再次通过 1*1分组点卷积 +  和残差通路 通道叠加合并 + relu激活
        elif fusion == 'add':
            group_conv1x1 = grouped_conv2d('Gconv1x1', x=depthwise, w=None,
                                           num_filters=num_filters,
                                           kernel_size=(1, 1),
                                           padding='VALID',
                                           num_groups=num_groups, l2_strength=l2_strength, bias=bias,
                                           activation=None,
                                           batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            # 通道叠加合并是 需要保证 x和F(x)的通道数量一致
            residual_match = residual_pooled
            # This is used if the number of filters of the residual block is different from that
            # of the group convolution.
            if num_filters != residual_pooled.get_shape()[3].value:
                # 对X 再次卷积 使得其通道数量和 F(x)的通道数量一致
                residual_match = conv2d('residual_match', x=residual_pooled, w=None, num_filters=num_filters,
                                        kernel_size=(1, 1),
                                        padding='VALID', l2_strength=l2_strength, bias=bias, activation=None,
                                        batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            return activation(group_conv1x1 + residual_match)
        else:
            raise ValueError("Specify whether the fusion is \'concat\' or \'add\'")


############################################################################################################
# Fully Connected layer Methods 全连接层 ########################
################################################################

#########################################################
# 8. 全连接层之前 最后的卷积之后 摊平 (N,H,W,C)----> (N,D)   D = H*W*C
###### 卷积后得到 (N,H,W,C) ----> (N,D) ----> 全链接 (N,D)*(D,output_dim) ----> (N,output_dim)
def flatten(x):
    """
    Flatten a (N,H,W,C) input into (N,D) output. Used for fully connected layers after conolution layers
    :param x: (tf.tensor) representing input
    :return: flattened output
    """
    all_dims_exc_first = np.prod([v.value for v in x.get_shape()[1:]])# D = H*W*C
    o = tf.reshape(x, [-1, all_dims_exc_first])
    return o
  
###################################################################
# 9. 全连接 操作 (N,D)*(D,output_dim) + Baise --> (N,output_dim) ## 
def __dense_p(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
              bias=0.0):
    """
    Fully connected layer
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H)
    """
    n_in = x.get_shape()[-1].value# 最后一个参数为 输入通道数量 (N,D) 也就是D
    with tf.variable_scope(name):
        if w == None:
            w = __variable_with_weight_decay([n_in, output_dim], initializer, l2_strength)
        __variable_summaries(w)
        if isinstance(bias, float):
            bias = tf.get_variable("layer_biases", [output_dim], tf.float32, tf.constant_initializer(bias))
        __variable_summaries(bias)
        output = tf.nn.bias_add(tf.matmul(x, w), bias)
        return output

##############################################################
# 10. 全连接层 全链接 + BN + 激活 + 随机失活dropout
def dense(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
          bias=0.0,
          activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
          is_training=True
          ):
    """
    This block is responsible for a fully connected followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return out: The output of the layer. (N, H)
    """
    with tf.variable_scope(name) as scope:
        # 全链接  操作 (N,D)*(D,output_dim) + Baise --> (N,output_dim)
        dense_o_b = __dense_p(name='dense', x=x, w=w, output_dim=output_dim, initializer=initializer,
                              l2_strength=l2_strength,
                              bias=bias)
        # BN 批规范化 + relu激活
        if batchnorm_enabled:
            dense_o_bn = tf.layers.batch_normalization(dense_o_b, training=is_training, epsilon=1e-5)
            if not activation:
                dense_a = dense_o_bn
            else:
                dense_a = activation(dense_o_bn)
        else:
            if not activation:
                dense_a = dense_o_b
            else:
                dense_a = activation(dense_o_b)
        # 随机失活
        def dropout_with_keep():
            return tf.nn.dropout(dense_a, dropout_keep_prob)
        def dropout_no_keep():
            return tf.nn.dropout(dense_a, 1.0)
        if dropout_keep_prob != -1:
            dense_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            dense_o_dr = dense_a
        dense_o = dense_o_dr
    return dense_o
  

############################################################################################################
# Pooling Methods 池化方法  最大值池化   均值池化
###########################################################################
# 11. 最大值池化
def max_pool_2d(x, size=(2, 2), stride=(2, 2), name='pooling'):
    """
    Max pooling 2D Wrapper
    :param x: (tf.tensor) The input to the layer (N,H,W,C).
    :param size: (tuple) This specifies the size of the filter as well as the stride.
    :param name: (string) Scope name.
    :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C).
    """
    size_x, size_y = size
    stride_x, stride_y = stride
    return tf.nn.max_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, stride_x, stride_y, 1], padding='VALID',
                          name=name)
###############################
# 12. 均值池化##############
def avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID'):
    """
        Average pooling 2D Wrapper
        :param x: (tf.tensor) The input to the layer (N,H,W,C).
        :param size: (tuple) This specifies the size of the filter as well as the stride.
        :param name: (string) Scope name.
        :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C).
    """
    size_x, size_y = size
    stride_x, stride_y = stride
    return tf.nn.avg_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, stride_x, stride_y, 1], padding=padding,
                          name=name)


############################################################################################################
# 13. 权重参数初始化  可带有  L2 正则项
#######################################
def __variable_with_weight_decay(kernel_shape, initializer, wd):
    """
    Create a variable with L2 Regularization (Weight Decay)
    :param kernel_shape: the size of the convolving weight kernel.
    :param initializer: The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param wd:(weight decay) L2 regularization parameter.
    :return: The weights of the kernel initialized. The L2 loss is added to the loss collection.
    """
    w = tf.get_variable('weights', kernel_shape, tf.float32, initializer=initializer)

    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='w_loss')
        tf.add_to_collection(collection_name, weight_decay)
    return w
  
#######################################################################
# 14. 参数记录 均值 方差  最大值 最小值 直方图
# Summaries for variables
def __variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :param var: variable to be summarized
    :return: None
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)#均值
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):#方差
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))#最大值
        tf.summary.scalar('min', tf.reduce_min(var))#最小值
        tf.summary.histogram('histogram', var)#直方图
