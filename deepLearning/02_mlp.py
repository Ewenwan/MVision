"""
Multi-Layer Perceptron Class
author: Ye Hu
2016/12/15
redit：wanyouwen 2018/05/02
"""
import numpy as np
import tensorflow as tf
import input_data
from logisticRegression import LogisticRegression

class HiddenLayer(object):
    """Typical hidden layer of MLP"""
    def __init__(self, inpt, n_in, n_out, W=None, b=None,
                 activation=tf.nn.sigmoid):
        """
        inpt: tf.Tensor, shape [n_examples, n_in]
        n_in: int, the dimensionality of input
        n_out: int, number of hidden units
        W, b: tf.Tensor, weight and bias
        activation: tf.op, activation function
        """
        if W is None:
            bound_val = 4.0*np.sqrt(6.0/(n_in + n_out))
            W = tf.Variable(tf.random_uniform([n_in, n_out], minval=-bound_val, maxval=bound_val),
                            dtype=tf.float32, name="W")
        if b is None:
            b = tf.Variable(tf.zeros([n_out,]), dtype=tf.float32, name="b")

        self.W = W
        self.b = b
        # the output
        sum_W = tf.matmul(inpt, self.W) + self.b
        self.output = activation(sum_W) if activation is not None else sum_W
        # params
        self.params = [self.W, self.b]


class MLP(object):
    """Multi-layer perceptron class"""
    def __init__(self, inpt, n_in, n_hidden, n_out):
        """
        inpt: tf.Tensor, shape [n_examples, n_in]
        n_in: int, the dimensionality of input
        n_hidden: int, number of hidden units
        n_out: int, number of output units
        """
        # hidden layer
        self.hiddenLayer = HiddenLayer(inpt, n_in=n_in, n_out=n_hidden)
        # output layer (logistic layer)
        self.outputLayer = LogisticRegression(self.hiddenLayer.output, n_in=n_hidden,
                                              n_out=n_out)
        # L1 norm
        self.L1 = tf.reduce_sum(tf.abs(self.hiddenLayer.W)) + \
                  tf.reduce_sum(tf.abs(self.outputLayer.W))
        # L2 norm
        self.L2 = tf.reduce_sum(tf.square(self.hiddenLayer.W)) + \
                  tf.reduce_sum(tf.square(self.outputLayer.W))
        # cross_entropy cost function
        self.cost = self.outputLayer.cost
        # accuracy function
        self.accuracy = self.outputLayer.accuarcy

        # params
        self.params = self.hiddenLayer.params + self.outputLayer.params
        # keep track of input
        self.input = inpt


if __name__ == "__main__":
    # mnist examples
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # define input and output placehoders
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    # create mlp model
    mlp_classifier = MLP(inpt=x, n_in=784, n_hidden=500, n_out=10)
    # get cost
    l2_reg = 0.0001
    cost = mlp_classifier.cost(y_) + l2_reg*mlp_classifier.L2
    # accuracy
    accuracy = mlp_classifier.accuracy(y_)
    predictor = mlp_classifier.outputLayer.y_pred
    # 定义训练器
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(
        cost, var_list=mlp_classifier.params)

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
                sess.run(train_op, feed_dict={x: x_batch, y_: y_batch})
                # 计算cost
                avg_cost += sess.run(cost, feed_dict={x: x_batch, y_: y_batch}) / batch_num
            # 输出
            if epoch % display_step == 0:
                val_acc = sess.run(accuracy, feed_dict={x: mnist.validation.images,
                                                       y_: mnist.validation.labels})
                print("Epoch {0} cost: {1}, validation accuacy: {2}".format(epoch,
                                                                            avg_cost, val_acc))

        print("Finished!")
        test_x = mnist.test.images[:10]
        test_y = mnist.test.labels[:10]
        print("Ture lables:")
        print("  ", np.argmax(test_y, 1))
        print("Prediction:")
        print("  ", sess.run(predictor, feed_dict={x: test_x}))




