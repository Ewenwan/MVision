#-*- coding:utf-8 -*-
# 轻量级网络--ShuffleNet 分组点卷积+通道重排+逐通道卷积
"""
0. 图像预处理 减去均值 乘以归一化系数
1. conv1 3*3*3*24 卷积 步长2 BN RELU
2. 最大值池化 3*3 步长2
3. 一次 步长为2 非分组点卷积 concate通道扩展合并模块， 再进行3次步长为1的 add通道叠加模块
4. 一次 步长为2 分组点卷积   concate通道扩展合并模块， 再进行7次步长为1的 add通道叠加模块
5. 一次 步长为2 分组点卷积   concate通道扩展合并模块， 再进行3次步长为1的 add通道叠加模块
6. 全局均值池化层 7*7 池化核 步长1
7. 1*1点卷积 输出 类别数量个 卷积特征图
8. 摊平 到 一维
"""
# layers 
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
#                ShuffleNet核心模块  2D卷积   最大值池化    均值池化    全链接层    全链接之前的摊平层    
from layers import shufflenet_unit, conv2d, max_pool_2d, avg_pool_2d, dense, flatten


class ShuffleNet:
    """ShuffleNet is implemented here!"""
    MEAN = [103.94, 116.78, 123.68]# 个通道 像素值 减去的值 均值
    NORMALIZER = 0.017# 归一化比例

    def __init__(self, args):
        self.args = args
        self.X = None
        self.y = None
        self.logits = None
        self.is_training = None
        self.loss = None
        self.regularization_loss = None
        self.cross_entropy_loss = None
        self.train_op = None
        self.accuracy = None
        self.y_out_argmax = None
        self.summaries_merged = None

        # A number stands for the num_groups
        # Output channels for conv1 layer
        self.output_channels = {'1': [144, 288, 576], '2': [200, 400, 800], '3': [240, 480, 960], '4': [272, 544, 1088],
                                '8': [384, 768, 1536], 'conv1': 24}

        self.__build()
    # 初始化输入
    def __init_input(self):
        batch_size = self.args.batch_size if self.args.train_or_test == 'train' else 1
        with tf.variable_scope('input'):
            # 输入图片 Input images 图片数量*长*宽*通道数量
            self.X = tf.placeholder(tf.float32,
                                    [batch_size, self.args.img_height, self.args.img_width,
                                     self.args.num_channels])
            # 数据对应得标签
            self.y = tf.placeholder(tf.int32, [batch_size])
            # is_training is for batch normalization and dropout, if they exist
            self.is_training = tf.placeholder(tf.bool)
    # 改变 图像大小 
    def __resize(self, x):#双三次插值
        return tf.image.resize_bicubic(x, [224, 224])
  
    # 先进行一次 步长为2 的下采样 concate合并模块， 再进行多次步长为1的 add通道叠加模块
    def __stage(self, x, stage=2, repeat=3):
        if 2 <= stage <= 4:
            stage_layer = shufflenet_unit('stage' + str(stage) + '_0', x=x, w=None,
                                          num_groups=self.args.num_groups,
                                          group_conv_bottleneck=not (stage == 2),# stage = 2 时 先不进行分组点卷积
                                          num_filters=
                                          self.output_channels[str(self.args.num_groups)][
                                              stage - 2],
                                          stride=(2, 2),# concate通道扩展合并
                                          fusion='concat', l2_strength=self.args.l2_strength,
                                          bias=self.args.bias,
                                          batchnorm_enabled=self.args.batchnorm_enabled,
                                          is_training=self.is_training)
            for i in range(1, repeat + 1):
                stage_layer = shufflenet_unit('stage' + str(stage) + '_' + str(i),
                                              x=stage_layer, w=None,
                                              num_groups=self.args.num_groups,
                                              group_conv_bottleneck=True,# 分组点卷积
                                              num_filters=self.output_channels[
                                                  str(self.args.num_groups)][stage - 2],
                                              stride=(1, 1),#ADD 通道叠加
                                              fusion='add',
                                              l2_strength=self.args.l2_strength,
                                              bias=self.args.bias,
                                              batchnorm_enabled=self.args.batchnorm_enabled,
                                              is_training=self.is_training)
            return stage_layer
        else:
            raise ValueError("Stage should be from 2 -> 4")
    # 输出
    def __init_output(self):
        with tf.variable_scope('output'):
            # Losses
            self.regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.cross_entropy_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y, name='loss'))
            self.loss = self.regularization_loss + self.cross_entropy_loss

            # Optimizer
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
                self.train_op = self.optimizer.minimize(self.loss)
                # This is for debugging NaNs. Check TensorFlow documentation.
                self.check_op = tf.add_check_numerics_ops()

            # Output and Metrics
            self.y_out_softmax = tf.nn.softmax(self.logits)# softmax 归一化分类
            self.y_out_argmax = tf.argmax(self.y_out_softmax, axis=-1, output_type=tf.int32)# 最大值得到分类结果
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_out_argmax), tf.float32))#准确度
        # 记录参数
        with tf.name_scope('train-summary-per-iteration'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('acc', self.accuracy)
            self.summaries_merged = tf.summary.merge_all()
    
    def __build(self):
        self.__init_global_epoch()
        self.__init_global_step()
        self.__init_input()
        # 0. 图像预处理 减去均值 乘以归一化系数##################################
        with tf.name_scope('Preprocessing'):
            # 分割成三通道
            red, green, blue = tf.split(self.X, num_or_size_splits=3, axis=3)
            # 每个通道 减去均值 乘以归一化系数 后再concat/merge 通道扩展合并
            preprocessed_input = tf.concat([
                tf.subtract(blue, ShuffleNet.MEAN[0]) * ShuffleNet.NORMALIZER,
                tf.subtract(green, ShuffleNet.MEAN[1]) * ShuffleNet.NORMALIZER,
                tf.subtract(red, ShuffleNet.MEAN[2]) * ShuffleNet.NORMALIZER,
            ], 3)
        # 1. conv1 3*3*3*24 卷积 步长 2 BN RELU #########################################################
        ######## 周围填充 
        x_padded = tf.pad(preprocessed_input, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        ######## conv
        conv1 = conv2d('conv1', x=x_padded, w=None, num_filters=self.output_channels['conv1'], kernel_size=(3, 3),
                       stride=(2, 2), l2_strength=self.args.l2_strength, bias=self.args.bias,
                       batchnorm_enabled=self.args.batchnorm_enabled, is_training=self.is_training,
                       activation=tf.nn.relu, padding='VALID')
        # 2. 最大值池化 3*3 步长2 ##################################################
        padded = tf.pad(conv1, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT")
        max_pool = max_pool_2d(padded, size=(3, 3), stride=(2, 2), name='max_pool')
        # 3. 一次 步长为2 非分组点卷积 concate通道扩展合并模块， 再进行3次步长为1的 add通道叠加模块
        stage2 = self.__stage(max_pool, stage=2, repeat=3)
        # 4. 一次 步长为2 分组点卷积   concate通道扩展合并模块， 再进行7次步长为1的 add通道叠加模块
        stage3 = self.__stage(stage2, stage=3, repeat=7)
        # 5. 一次 步长为2 分组点卷积   concate通道扩展合并模块， 再进行3次步长为1的 add通道叠加模块
        stage4 = self.__stage(stage3, stage=4, repeat=3)
        # 6. 全局均值池化层 7*7 池化核 步长1
        global_pool = avg_pool_2d(stage4, size=(7, 7), stride=(1, 1), name='global_pool', padding='VALID')
        # 7. 1*1点卷积 输出 类别数量个 卷积特征图
        logits_unflattened = conv2d('fc', global_pool, w=None, num_filters=self.args.num_classes,
                                    kernel_size=(1, 1),# 1*1点卷积
                                    l2_strength=self.args.l2_strength,
                                    bias=self.args.bias,
                                    is_training=self.is_training)
        # 8. 摊平 到 一维
        self.logits = flatten(logits_unflattened)
        # 9. 计算误差 
        self.__init_output()

    def __init_global_epoch(self):
        """
        Create a global epoch tensor to totally save the process of the training
        :return:
        """
        with tf.variable_scope('global_epoch'):
            self.global_epoch_tensor = tf.Variable(-1, trainable=False, name='global_epoch')
            self.global_epoch_input = tf.placeholder('int32', None, name='global_epoch_input')
            self.global_epoch_assign_op = self.global_epoch_tensor.assign(self.global_epoch_input)

    def __init_global_step(self):
        """
        Create a global step variable to be a reference to the number of iterations
        :return:
        """
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)
