"""
Stacked Denoising Autoencoders (SDA)
author: Ye Hu
2016/12/16
redit：wanyouwen 2018/05/02
"""
import timeit
import numpy as np
import tensorflow as tf
import input_data

from logisticRegression import LogisticRegression
from mlp import HiddenLayer
from da import DA

class SdA(object):
    """
    Stacked denoising autoencoder class
    the model is constructed by stacking several dAs
    the dA layers are used to initialize the network, after pre-training,
    the SdA is similar to a normal MLP
    """
    def __init__(self, n_in=784, n_out=10, hidden_layers_sizes=(500, 500),
                 corruption_levels=(0.1, 0.1)):
        """
        :param n_in: int, the dimension of input
        :param n_out: int, the dimension of output
        :param hidden_layers_sizes: list or tuple, the hidden layer sizes
        :param corruption_levels: list or tuple, the corruption lever for each layer
        """
        assert len(hidden_layers_sizes) >= 1
        assert len(hidden_layers_sizes) == len(corruption_levels)
        self.corruption_levels = corruption_levels
        self.n_layers = len(hidden_layers_sizes)
        # define the layers
        self.layers = []   # the normal layers
        self.dA_layers = []  # the dA layers
        self.params = []     # params
        # define the input and output
        self.x = tf.placeholder(tf.float32, shape=[None, n_in])
        self.y = tf.placeholder(tf.float32, shape=[None, n_out])
        # construct the layers
        for i in range(self.n_layers):
            if i == 0:  # the input layer
                input_size = n_in
                layer_input = self.x
            else:
                input_size = hidden_layers_sizes[i-1]
                layer_input = self.layers[i-1].output
            # create the sigmoid layer
            sigmoid_layer = HiddenLayer(inpt=layer_input, n_in=input_size,
                                         n_out=hidden_layers_sizes[i], activation=tf.nn.sigmoid)
            self.layers.append(sigmoid_layer)
            # create the da layer
            dA_layer = DA(inpt=layer_input, n_hidden=hidden_layers_sizes[i], n_visiable=input_size,
                          W=sigmoid_layer.W, bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)

            # collect the params
            self.params.extend(sigmoid_layer.params)

        # add the output layer
        self.output_layer = LogisticRegression(inpt=self.layers[-1].output, n_in=hidden_layers_sizes[-1],
                                               n_out=n_out)
        self.params.extend(self.output_layer.params)

        # the finetuning cost
        self.finetune_cost = self.output_layer.cost(self.y)
        # the accuracy
        self.accuracy = self.output_layer.accuarcy(self.y)

    def pretrain(self, sess, X_train, pretraining_epochs=10, batch_size=100, learning_rate=0.001,
                 display_step=1):
        """
        Pretrain the layers
        :param sess: tf.Session
        :param X_train: the input of the train set
        :param batch_size: int
        :param learning_rate: float
        """
        print('Starting pretraining...')
        start_time = timeit.default_timer()
        batch_num = int(X_train.train.num_examples / batch_size)
        for i in range(self.n_layers):
            # pretraining layer by layer
            cost = self.dA_layers[i].get_cost(corruption_level=self.corruption_levels[i])
            params = self.dA_layers[i].params
            train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, var_list=params)
            for epoch in range(pretraining_epochs):
                avg_cost = 0.0
                for j in range(batch_num):
                    x_batch, _ = X_train.train.next_batch(batch_size)
                    # 训练
                    sess.run(train_op, feed_dict={self.x: x_batch})
                    # 计算cost
                    avg_cost += sess.run(cost, feed_dict={self.x: x_batch,}) / batch_num
                # 输出
                if epoch % display_step == 0:
                    print("Pretraing layer {0} Epoch {1} cost: {2}".format(i, epoch, avg_cost))

        end_time = timeit.default_timer()
        print("The pretraining process ran for {0}m".format((end_time - start_time) / 60))

    def finetuning(self, sess, trainSet, training_epochs=10, batch_size=100, learning_rate=0.1,
                   display_step=1):
        """Finetuing the network"""
        print("Start finetuning...")
        start_time = timeit.default_timer()
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            self.finetune_cost, var_list=self.params)
        for epoch in range(training_epochs):
            avg_cost = 0.0
            batch_num = int(trainSet.train.num_examples / batch_size)
            for i in range(batch_num):
                x_batch, y_batch = trainSet.train.next_batch(batch_size)
                # 训练
                sess.run(train_op, feed_dict={self.x: x_batch, self.y: y_batch})
                # 计算cost
                avg_cost += sess.run(self.finetune_cost, feed_dict=
                {self.x: x_batch, self.y: y_batch}) / batch_num
            # 输出
            if epoch % display_step == 0:
                val_acc = sess.run(self.accuracy, feed_dict={self.x: trainSet.validation.images,
                                                       self.y: trainSet.validation.labels})
                print("  Epoch {0} cost: {1}, validation accuacy: {2}".format(epoch, avg_cost, val_acc))

        end_time = timeit.default_timer()
        print("The finetuning process ran for {0}m".format((end_time - start_time) / 60))


if __name__ == "__main__":
    # mnist examples
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    sda = SdA(n_in=784, n_out=10, hidden_layers_sizes=[500, 500, 500], corruption_levels=[0.1, 0.2, 0.2])
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    # set random_seed
    tf.set_random_seed(seed=1111)
    sda.pretrain(sess, X_train=mnist)
    sda.finetuning(sess, trainSet=mnist)

