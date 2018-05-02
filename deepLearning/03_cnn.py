"""
Convolution neural network
author: Ye Hu
2016/12/15
redit：wanyouwen 2018/05/02
"""
import numpy as np
import tensorflow as tf
import input_data
from logisticRegression import LogisticRegression
from mlp import HiddenLayer

class ConvLayer(object):
    """
    A convolution layer
    """
    def __init__(self, inpt, filter_shape, strides=(1, 1, 1, 1),
                 padding="SAME", activation=tf.nn.relu, bias_setting=True):
        """
        inpt: tf.Tensor, shape [n_examples, witdth, height, channels]
        filter_shape: list or tuple, [witdth, height. channels, filter_nums]
        strides: list or tuple, the step of filter
        padding:
        activation:
        bias_setting:
        """
        self.input = inpt
        # initializes the filter
        self.W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), dtype=tf.float32)
        if bias_setting:
            self.b = tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1),
                                 dtype=tf.float32)
        else:
            self.b = None
        conv_output = tf.nn.conv2d(self.input, filter=self.W, strides=strides,
                                   padding=padding)
        conv_output = conv_output + self.b if self.b is not None else conv_output
        # the output
        self.output = conv_output if activation is None else activation(conv_output)
        # the params
        self.params = [self.W, self.b] if self.b is not None else [self.W, ]


class MaxPoolLayer(object):
    """pool layer"""
    def __init__(self, inpt, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME"):
        """
        """
        self.input = inpt
        # the output
        self.output = tf.nn.max_pool(self.input, ksize=ksize, strides=strides, padding=padding)
        self.params = []


class FlattenLayer(object):
    """Flatten layer"""
    def __init__(self, inpt, shape):
        self.input = inpt
        self.output = tf.reshape(self.input, shape=shape)
        self.params = []

class DropoutLayer(object):
    """Dropout layer"""
    def __init__(self, inpt, keep_prob):
        """
        keep_prob: float (0, 1]
        """
        self.keep_prob = tf.placeholder(tf.float32)
        self.input = inpt
        self.output = tf.nn.dropout(self.input, keep_prob=self.keep_prob)
        self.train_dicts = {self.keep_prob: keep_prob}
        self.pred_dicts = {self.keep_prob: 1.0}

if __name__ == "__main__":
    # mnist examples
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # define input and output placehoders
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    # reshape
    inpt = tf.reshape(x, shape=[-1, 28, 28, 1])

    # create network
    # params for training
    # conv and pool layer0
    layer0_conv = ConvLayer(inpt, filter_shape=[5, 5, 1, 32], strides=[1, 1, 1, 1], activation=tf.nn.relu,
                            padding="SAME")              # [?, 28, 28, 32]
    layer0_pool = MaxPoolLayer(layer0_conv.output, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1])                       # [?, 14, 14, 32]
    # conv and pool layer1
    layer1_conv = ConvLayer(layer0_pool.output, filter_shape=[5, 5, 32, 64], strides=[1, 1, 1, 1],
                            activation=tf.nn.relu, padding="SAME")  # [?, 14, 14, 64]
    layer1_pool = MaxPoolLayer(layer1_conv.output, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1])              # [?, 7, 7, 64]
    # flatten layer
    layer2_flatten = FlattenLayer(layer1_pool.output, shape=[-1, 7*7*64])
    # fully-connected layer
    layer3_fullyconn = HiddenLayer(layer2_flatten.output, n_in=7*7*64, n_out=256, activation=tf.nn.relu)
    # dropout layer
    layer3_dropout = DropoutLayer(layer3_fullyconn.output, keep_prob=0.5)
    # the output layer
    layer4_output = LogisticRegression(layer3_dropout.output, n_in=256, n_out=10)

    # params for training
    params = layer0_conv.params + layer1_conv.params + layer3_fullyconn.params + layer4_output.params
    # train dicts for dropout
    train_dicts = layer3_dropout.train_dicts
    # prediction dicts for dropout
    pred_dicts = layer3_dropout.pred_dicts

    # get cost
    cost = layer4_output.cost(y_)
    # accuracy
    accuracy = layer4_output.accuarcy(y_)
    predictor = layer4_output.y_pred
    # 定义训练器
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(
        cost, var_list=params)

    # 初始化所有变量
    init = tf.global_variables_initializer()

    # 定义训练参数
    training_epochs = 10
    batch_size = 100
    display_step = 1

    # 开始训练
    print("Start to train...")
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0.0
            batch_num = int(mnist.train.num_examples / batch_size)
            for i in range(batch_num):
                x_batch, y_batch = mnist.train.next_batch(batch_size)
                # 训练
                train_dicts.update({x: x_batch, y_: y_batch})

                sess.run(train_op, feed_dict=train_dicts)
                # 计算cost
                pred_dicts.update({x: x_batch, y_: y_batch})
                avg_cost += sess.run(cost, feed_dict=pred_dicts) / batch_num
            # 输出
            if epoch % display_step == 0:
                pred_dicts.update({x: mnist.validation.images,
                                   y_: mnist.validation.labels})
                val_acc = sess.run(accuracy, feed_dict=pred_dicts)
                print("Epoch {0} cost: {1}, validation accuacy: {2}".format(epoch,
                                                                            avg_cost, val_acc))

        print("Finished!")
        test_x = mnist.test.images[:10]
        test_y = mnist.test.labels[:10]
        print("Ture lables:")
        print("  ", np.argmax(test_y, 1))
        print("Prediction:")
        pred_dicts.update({x: test_x})
        print("  ", sess.run(predictor, feed_dict=pred_dicts))
        tf.scan()





