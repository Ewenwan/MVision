#-*- coding: utf-8 -*-
"""
残差网络 f(x) + W*x
50个卷积层
ResNet50
2018/04/22

1  64个输出  7*7卷积核 步长 2 224*224图像输入 大小减半（+BN + RELU + MaxPol）
2  3个 3*3卷积核  64输出的残差模块 256/4 = 64     且第一个残差块的第一个卷积步长为1
3  4个 3*3卷积核  128输出的残差模块 512/4 = 128   且第一个残差块的第一个卷积步长为2      
4  6个 3*3卷积核  256输出的残差模块 1024/4 = 256  且第一个残差块的第一个卷积步长为2  
5  3个  3*3卷积核 512输出的残差模块 2048/4 = 512  且第一个残差块的第一个卷积步长为2  
6  均值池化 
7  全连接层 输出 1000  类
8  softmax 类别预测

实际中，考虑计算的成本，对残差块做了计算优化，即将2个3x3的卷积层替换为 1x1 + 3x3 + 1x1 。
新结构中的中间3x3的卷积层首先在一个降维1x1卷积层下减少了计算，然后在另一个1x1的卷积层下做了还原，既保持了精度又减少了计算量。
运行
python3 ResNet50.py

# python3.4 对应的1.4版本 tensorflow 安装
sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp34-cp34m-linux_x86_64.whl


"""

import tensorflow as tf
from tensorflow.python.training import moving_averages# 移动平均
####### 全连接层 卷积层初始化############################
fc_initializer = tf.contrib.layers.xavier_initializer
conv2d_initializer = tf.contrib.layers.xavier_initializer_conv2d

######## 创建变量 create weight variable########################
def create_var(name, shape, initializer, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=tf.float32,
                           initializer=initializer, trainable=trainable)
####### 2d卷积层 conv2d layer##################################
####### 卷积核3维 加一个输出数量 [filter_w, filter_h, input_chanels, output_chanels]
def conv2d(x, num_outputs, kernel_size, stride=1, scope="conv2d"):
    num_inputs = x.get_shape()[-1]#输入通道数
    with tf.variable_scope(scope):
        kernel = create_var("kernel", [kernel_size, kernel_size,
                                       num_inputs, num_outputs],
                            conv2d_initializer())
        return tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1],
                            padding="SAME")
###### 全连接层 fully connected layer###########################
def fc(x, num_outputs, scope="fc"):
    num_inputs = x.get_shape()[-1]# 输入通道数
    with tf.variable_scope(scope):
        weight = create_var("weight", [num_inputs, num_outputs],
                            fc_initializer())
        bias = create_var("bias", [num_outputs,],
                          tf.zeros_initializer())
        return tf.nn.xw_plus_b(x, weight, bias)
####### 批规范化 （去均值 除以方差 零均值 1方差处理）batch norm layer#########
def batch_norm(x, decay=0.999, epsilon=1e-03, is_training=True,
               scope="scope"):
    x_shape = x.get_shape()
    num_inputs = x_shape[-1]#输入通道数
    reduce_dims = list(range(len(x_shape) - 1))
    with tf.variable_scope(scope):
        # 版本问题 beta = create_var("beta", [num_inputs,], initializer=tf.zeros_initializer())
        beta = create_var("beta", [num_inputs,], initializer=tf.zeros_initializer())
        gamma = create_var("gamma", [num_inputs,], initializer=tf.ones_initializer())
        # 移动均值for inference
        moving_mean = create_var("moving_mean", [num_inputs,], initializer=tf.zeros_initializer(), trainable=False)
        # 方差
        moving_variance = create_var("moving_variance", [num_inputs], initializer=tf.ones_initializer(), trainable=False)
    if is_training:
        mean, variance = tf.nn.moments(x, axes=reduce_dims)
        update_move_mean = moving_averages.assign_moving_average(moving_mean,
                                                mean, decay=decay)
        update_move_variance = moving_averages.assign_moving_average(moving_variance,
                                                variance, decay=decay)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_variance)
    else:
        mean, variance = moving_mean, moving_variance
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)


############# 均值池化层 avg pool layer###################
def avg_pool(x, pool_size, scope):
    with tf.variable_scope(scope):
        return tf.nn.avg_pool(x, [1, pool_size, pool_size, 1],
                strides=[1, pool_size, pool_size, 1], padding="VALID")

############# 最大值池化层 max pool layer######################
def max_pool(x, pool_size, stride, scope):
    with tf.variable_scope(scope):
        return tf.nn.max_pool(x, [1, pool_size, pool_size, 1],
                              [1, stride, stride, 1], padding="SAME")
################ 残差网络 1000 类#######################
class ResNet50(object):
    def __init__(self, inputs, num_classes=1000, is_training=True,
                 scope="resnet50"):
        self.inputs =inputs# 输入数量
        self.is_training = is_training
        self.num_classes = num_classes# 类别数量

        with tf.variable_scope(scope):
            # 定义模型结构 construct the model
            # 64个输出  7*7卷积核 步长 2 224*224图像输入 大小减半
            net = conv2d(inputs, 64, 7, 2, scope="conv1") # -> [batch, 112, 112, 64]
            # 先 批规范化 在 relu激活
            net = tf.nn.relu(batch_norm(net, is_training=self.is_training, scope="bn1"))
            # 最大值池化 步长2 大小减半
            net = max_pool(net, 3, 2, scope="maxpool1")  # -> [batch, 56, 56, 64]
            # 3个 3*3卷积核 64输出的残差模块 256/4 = 64  且第一个残差块的第一个卷积步长为1
            net = self._block(net, 256, 3, init_stride=1, is_training=self.is_training,scope="block2") # -> [batch, 56, 56, 256]
            # 4个 3*3卷积核 128输出的残差模块 512/4 = 128 且第一个残差块的第一个卷积步长为2             
            net = self._block(net, 512, 4, is_training=self.is_training, scope="block3")# -> [batch, 28, 28, 512]
            # 6个 3*3卷积核 256输出的残差模块 1024/4 = 256且第一个残差块的第一个卷积2                                                    
            net = self._block(net, 1024, 6, is_training=self.is_training, scope="block4")# -> [batch, 14, 14, 1024]
            # 3个 3*3卷积核 512输出的残差模块 2048/4 = 512 且第一个残差块的第一个卷积步长为2                                              
            net = self._block(net, 2048, 3, is_training=self.is_training, scope="block5")# -> [batch, 7, 7, 2048]
            # 均值池化                                          
            net = avg_pool(net, 7, scope="avgpool5") # -> [batch, 1, 1, 2048]  
            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze") # -> [batch, 2048]
            # 全连接层
            self.logits = fc(net, self.num_classes, "fc6")       # -> [batch, num_classes]
            # 预测输出
            self.predictions = tf.nn.softmax(self.logits)

    # 残差块集合 默认卷积步长为2
    def _block(self, x, n_out, n, init_stride=2, is_training=True, scope="block"):
        with tf.variable_scope(scope):
            h_out = n_out // 4 #上面的调用 该参数都为实际输出通道的 4被所有这里除以4
            # 第一个残差模型（会涉及到 不同残差集合块 不同通道的合并） f(x) + x
            # 这里第一个残差块的 卷积步长 与后面的残差块的卷积步长可能不一样 所以独立出来
            out = self._bottleneck(x, h_out, n_out, stride=init_stride, is_training=is_training, scope="bottlencek1")
            for i in range(1, n):#1....(n-1)个残差块
                out = self._bottleneck(out, h_out, n_out, is_training=is_training, scope=("bottlencek%s" % (i + 1)))
            return out
    '''
    实际中，考虑计算的成本，对残差块做了计算优化，即将2个3x3的卷积层替换为 1x1 + 3x3 + 1x1 。
    新结构中的中间3x3的卷积层首先在一个降维1x1卷积层下减少了计算，然后在另一个1x1的卷积层下做了还原，既保持了精度又减少了计算量。
    '''
    # 残差模块 f(x) + x
    def _bottleneck(self, x, h_out, n_out, stride=None, is_training=True, scope="bottleneck"):
        """ A residual bottleneck unit"""
        n_in = x.get_shape()[-1]#输入通道
        if stride is None:
            stride = 1 if n_in == n_out else 2# 步长大小

        with tf.variable_scope(scope):
            # 经过两个3×3卷积(= 1*1  + 3*3  + 1*1)形成 f(x) 后与 x相加  
            # 第一个卷积(2d卷积 + 批规范化 + relu激活) 1*1的卷积核
            h = conv2d(x, h_out, 1, stride=stride, scope="conv_1")
            h = batch_norm(h, is_training=is_training, scope="bn_1")
            h = tf.nn.relu(h)
            #  第二个卷积(2d卷积 + 批规范化 + relu激活) 3*3的卷积核
            h = conv2d(h, h_out, 3, stride=1, scope="conv_2")
            h = batch_norm(h, is_training=is_training, scope="bn_2")
            h = tf.nn.relu(h)
            #  第三个卷积 1*1
            h = conv2d(h, n_out, 1, stride=1, scope="conv_3")
            h = batch_norm(h, is_training=is_training, scope="bn_3")

            if n_in != n_out:# 当 x和 f(x)通道不一致时 对x再次进行卷积 输出和f(x)一致的通道数
                shortcut = conv2d(x, n_out, 1, stride=stride, scope="conv_4")
                shortcut = batch_norm(shortcut, is_training=is_training, scope="bn_4")
            else:
                shortcut = x
            return tf.nn.relu(shortcut + h)# f(x) + w*x

if __name__ == "__main__":
    # 32个图像  224*224*3 3个通道 
    x = tf.random_normal([32, 224, 224, 3])
    resnet50 = ResNet50(x)# 32个图像每个图像 输出类别个数个 预测概率
    print(resnet50.logits)
