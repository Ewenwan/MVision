"""
Denoising Autoencoder (DA)
author: Ye Hu
2016/12/16
redit：wanyouwen 2018/05/02
"""
import os
import timeit

import numpy as np
import tensorflow as tf
from PIL import Image

import input_data
from utils import tile_raster_images



class DA(object):
    """A denoising autoencoder class (using tied weight)"""
    def __init__(self, inpt, n_visiable=784, n_hidden=500, W=None, bhid=None,
                 bvis=None, activation=tf.nn.sigmoid):
        """
        inpt: tf.Tensor, the input
        :param n_visiable: int, number of hidden units
        :param n_hidden: int, number of visable units
        :param W, bhid, bvis: tf.Tensor, the weight, bias tensor
        """
        self.n_visiable = n_visiable
        self.n_hidden = n_hidden
        # initialize the weight and bias if not given
        if W is None:
            bound = -4*np.sqrt(6.0 / (self.n_hidden + self.n_visiable))
            W = tf.Variable(tf.random_uniform([self.n_visiable, self.n_hidden], minval=-bound,
                                              maxval=bound), dtype=tf.float32)
        if bhid is None:
            bhid = tf.Variable(tf.zeros([n_hidden,]), dtype=tf.float32)
        if bvis is None:
            bvis = tf.Variable(tf.zeros([n_visiable,]), dtype=tf.float32)
        self.W = W
        self.b = bhid
        # reconstruct params
        self.b_prime = bvis
        self.W_prime = tf.transpose(self.W)
        # keep track of input and params
        self.input = inpt
        self.params = [self.W, self.b, self.b_prime]
        # activation
        self.activation = activation

    def get_encode_values(self, inpt):
        """Compute the encode values"""
        return self.activation(tf.matmul(inpt, self.W) + self.b)

    def get_decode_values(self, encode_input):
        """Get the reconstructed values"""
        return self.activation(tf.matmul(encode_input, self.W_prime) + self.b_prime)

    def get_corrupted_input(self, inpt, corruption_level):
        """
        Randomly zero the element of input
        corruption_level: float, (0,1]
        """
        # the shape of input
        input_shape = tf.shape(inpt)
        # the probablity for corruption
        probs = tf.tile(tf.log([[corruption_level, 1-corruption_level]]),
                        multiples=[input_shape[0], 1])
        return tf.mul(tf.cast(tf.multinomial(probs, num_samples=input_shape[1]),
                              dtype=tf.float32), inpt)

    def get_cost(self, corruption_level=0.3):
        """Get the cost for training"""
        corrupted_input = self.get_corrupted_input(self.input, corruption_level)
        encode_output = self.get_encode_values(corrupted_input)
        decode_output = self.get_decode_values(encode_output)
        # use cross_entropy
        cross = tf.mul(self.input, tf.log(decode_output)) + \
                tf.mul(1.0-self.input, tf.log(1.0-decode_output))
        cost = -tf.reduce_mean(tf.reduce_sum(cross, axis=1))
        return cost

if __name__ == "__main__":
    # mnist examples
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # define input
    x = tf.placeholder(tf.float32, shape=[None, 784])
    # set random_seed
    tf.set_random_seed(seed=99999)
    # the DA model
    da = DA(x, n_visiable=784, n_hidden=500)
    # corruption level
    corruption_level = 0.0
    learning_rate = 0.1
    cost = da.get_cost(corruption_level)
    params = da.params
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, var_list=params)
    init = tf.global_variables_initializer()

    output_folder = "dA_plots"
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    training_epochs = 10
    batch_size = 100
    display_step = 1
    print("Start training...")
    start_time = timeit.default_timer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0.0
            batch_num = int(mnist.train.num_examples / batch_size)
            for i in range(batch_num):
                x_batch, _ = mnist.train.next_batch(batch_size)
                # 训练
                sess.run(train_op, feed_dict={x: x_batch})
                # 计算cost
                avg_cost += sess.run(cost, feed_dict={x: x_batch,}) / batch_num
            # 输出
            if epoch % display_step == 0:
                print("Epoch {0} cost: {1}".format(epoch, avg_cost))

        end_time = timeit.default_timer()
        training_time = end_time - start_time
        print("Finished!")
        print("  The {0}%% corruption code ran for {1}.".format(corruption_level*100, training_time/60,))
        W_value = sess.run(da.W_prime)
        image = Image.fromarray(tile_raster_images(
            X=W_value,
            img_shape=(28, 28), tile_shape=(10, 10),
            tile_spacing=(1, 1)))
        image.save('filters_corruption_{0}.png'.format(int(corruption_level*100)))




