#-*- coding:utf-8 -*-
# 轻量级网络--ShuffleNet 分组点卷积+通道重排+逐通道卷积
"""

"""

import tensorflow as tf
#                ShuffleNet核心模块  2D卷积   最大值池化    均值池化    全链接层    全链接之前的摊平层    
from layers import shufflenet_unit, conv2d, max_pool_2d, avg_pool_2d, dense, flatten


class ShuffleNet:
    """ShuffleNet is implemented here!"""
    MEAN = [103.94, 116.78, 123.68]
    NORMALIZER = 0.017

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

    def __init_input(self):
        batch_size = self.args.batch_size if self.args.train_or_test == 'train' else 1
        with tf.variable_scope('input'):
            # Input images
            self.X = tf.placeholder(tf.float32,
                                    [batch_size, self.args.img_height, self.args.img_width,
                                     self.args.num_channels])
            # Classification supervision, it's an argmax. Feel free to change it to one-hot,
            # but don't forget to change the loss from sparse as well
            self.y = tf.placeholder(tf.int32, [batch_size])
            # is_training is for batch normalization and dropout, if they exist
            self.is_training = tf.placeholder(tf.bool)

    def __resize(self, x):
        return tf.image.resize_bicubic(x, [224, 224])

    def __stage(self, x, stage=2, repeat=3):
        if 2 <= stage <= 4:
            stage_layer = shufflenet_unit('stage' + str(stage) + '_0', x=x, w=None,
                                          num_groups=self.args.num_groups,
                                          group_conv_bottleneck=not (stage == 2),
                                          num_filters=
                                          self.output_channels[str(self.args.num_groups)][
                                              stage - 2],
                                          stride=(2, 2),
                                          fusion='concat', l2_strength=self.args.l2_strength,
                                          bias=self.args.bias,
                                          batchnorm_enabled=self.args.batchnorm_enabled,
                                          is_training=self.is_training)
            for i in range(1, repeat + 1):
                stage_layer = shufflenet_unit('stage' + str(stage) + '_' + str(i),
                                              x=stage_layer, w=None,
                                              num_groups=self.args.num_groups,
                                              group_conv_bottleneck=True,
                                              num_filters=self.output_channels[
                                                  str(self.args.num_groups)][stage - 2],
                                              stride=(1, 1),
                                              fusion='add',
                                              l2_strength=self.args.l2_strength,
                                              bias=self.args.bias,
                                              batchnorm_enabled=self.args.batchnorm_enabled,
                                              is_training=self.is_training)
            return stage_layer
        else:
            raise ValueError("Stage should be from 2 -> 4")

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
            self.y_out_softmax = tf.nn.softmax(self.logits)
            self.y_out_argmax = tf.argmax(self.y_out_softmax, axis=-1, output_type=tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_out_argmax), tf.float32))

        with tf.name_scope('train-summary-per-iteration'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('acc', self.accuracy)
            self.summaries_merged = tf.summary.merge_all()

    def __build(self):
        self.__init_global_epoch()
        self.__init_global_step()
        self.__init_input()

        with tf.name_scope('Preprocessing'):
            red, green, blue = tf.split(self.X, num_or_size_splits=3, axis=3)
            preprocessed_input = tf.concat([
                tf.subtract(blue, ShuffleNet.MEAN[0]) * ShuffleNet.NORMALIZER,
                tf.subtract(green, ShuffleNet.MEAN[1]) * ShuffleNet.NORMALIZER,
                tf.subtract(red, ShuffleNet.MEAN[2]) * ShuffleNet.NORMALIZER,
            ], 3)
        x_padded = tf.pad(preprocessed_input, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        conv1 = conv2d('conv1', x=x_padded, w=None, num_filters=self.output_channels['conv1'], kernel_size=(3, 3),
                       stride=(2, 2), l2_strength=self.args.l2_strength, bias=self.args.bias,
                       batchnorm_enabled=self.args.batchnorm_enabled, is_training=self.is_training,
                       activation=tf.nn.relu, padding='VALID')
        padded = tf.pad(conv1, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT")
        max_pool = max_pool_2d(padded, size=(3, 3), stride=(2, 2), name='max_pool')
        stage2 = self.__stage(max_pool, stage=2, repeat=3)
        stage3 = self.__stage(stage2, stage=3, repeat=7)
        stage4 = self.__stage(stage3, stage=4, repeat=3)
        global_pool = avg_pool_2d(stage4, size=(7, 7), stride=(1, 1), name='global_pool', padding='VALID')

        logits_unflattened = conv2d('fc', global_pool, w=None, num_filters=self.args.num_classes,
                                    kernel_size=(1, 1),
                                    l2_strength=self.args.l2_strength,
                                    bias=self.args.bias,
                                    is_training=self.is_training)
        self.logits = flatten(logits_unflattened)

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
