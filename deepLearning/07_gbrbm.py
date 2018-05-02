"""
Restricted Boltzmann Machines (RBM)
author: Ye Hu
2016/12/18
redit：wanyouwen 2018/05/02
"""
import os
import timeit
import numpy as np
import tensorflow as tf
from PIL import Image
from utils import tile_raster_images
import input_data
from rbm import RBM


class GBRBM(RBM):
    """
    Gaussian-binary Restricted Boltzmann Machines
    Note we assume that the standard deviation is a constant (not training parameter)
    You better normalize you data with range of [0, 1.0].
    """
    def __init__(self, inpt=None, n_visiable=784, n_hidden=500, sigma=1.0, W=None,
                 hbias=None, vbias=None, sample_visible=True):
        """
        :param inpt: Tensor, the input tensor [None, n_visiable]
        :param n_visiable: int, number of visiable units
        :param n_hidden: int, number of hidden units
        :param sigma: float, the standard deviation (note we use the same σ for all visible units)
        :param W, hbias, vbias: Tensor, the parameters of RBM (tf.Variable)
        :param sample_visble: bool, if True, do gaussian sampling.
        """
        super(GBRBM, self).__init__(inpt, n_visiable, n_hidden, W, hbias, vbias)
        self.sigma = sigma
        self.sample_visible = sample_visible
    
    @staticmethod
    def sample_gaussian(x, sigma):
        return x + tf.random_normal(tf.shape(x), mean=0.0, stddev=sigma)

    def propdown(self, h):
        """Compute the mean for visible units given hidden units"""
        return tf.matmul(h, tf.transpose(self.W)) + self.vbias
    
    def sample_v_given_h(self, h0_sample):
        """Sampling the visiable units given hidden sample"""
        v1_mean = self.propdown(h0_sample)
        v1_sample = v1_mean
        if self.sample_visible:
            v1_sample = GBRBM.sample_gaussian(v1_mean, self.sigma)
        return (v1_mean, v1_sample)
    
    def propup(self, v):
        """Compute the sigmoid activation for hidden units given visible units"""
        return tf.nn.sigmoid(tf.matmul(v, self.W) / self.sigma**2 + self.hbias)
    
    def free_energy(self, v_sample):
        """Compute the free energy"""
        wx_b = tf.matmul(v_sample, self.W) / self.sigma**2 + self.hbias
        vbias_term = tf.reduce_sum(0.5 * tf.square(v_sample - self.vbias) / self.sigma**2, axis=1)
        hidden_term = tf.reduce_sum(tf.log(1.0 + tf.exp(wx_b)), axis=1)
        return -hidden_term + vbias_term
    
    def get_reconstruction_cost(self):
        """Compute the mse of the original input and the reconstruction"""
        activation_h = self.propup(self.input)
        activation_v = self.propdown(activation_h)
        mse = tf.reduce_mean(tf.reduce_sum(tf.square(self.input - activation_v), axis=1))
        return mse  
        
    

if __name__ == "__main__":
    # mnist examples
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # define input
    x = tf.placeholder(tf.float32, shape=[None, 784])
    # set random_seed
    tf.set_random_seed(seed=99999)
    np.random.seed(123)
    # the rbm model
    n_visiable, n_hidden = 784, 500
    rbm = GBRBM(x, n_visiable=n_visiable, n_hidden=n_hidden)
    
    learning_rate = 0.01
    batch_size = 50
    cost = rbm.get_reconstruction_cost()
    # Create the persistent variable
    #persistent_chain = tf.Variable(tf.zeros([batch_size, n_hidden]), dtype=tf.float32)
    persistent_chain = None
    train_ops = rbm.get_train_ops(learning_rate=learning_rate, k=1, persistent=persistent_chain)
    init = tf.global_variables_initializer()

    output_folder = "rbm_plots"
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    training_epochs = 15
    display_step = 1
    print("Start training...")
   
    with tf.Session() as sess:
        start_time = timeit.default_timer()
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0.0
            batch_num = int(mnist.train.num_examples / batch_size)
            for i in range(batch_num):
                x_batch, _ = mnist.train.next_batch(batch_size)
                # 训练
                sess.run(train_ops, feed_dict={x: x_batch})
                # 计算cost
                avg_cost += sess.run(cost, feed_dict={x: x_batch,}) / batch_num
            # 输出
            if epoch % display_step == 0:
                print("Epoch {0} cost: {1}".format(epoch, avg_cost))
            # Construct image from the weight matrix
            image = Image.fromarray(
            tile_raster_images(
                X=sess.run(tf.transpose(rbm.W)),
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)))
            image.save("test_filters_at_epoch_{0}.png".format(epoch))

        end_time = timeit.default_timer()
        training_time = end_time - start_time
        print("Finished!")
        print("  The training ran for {0} minutes.".format(training_time/60,))
        
        # Randomly select the 'n_chains' examples
        n_chains = 20
        n_batch = 10
        n_samples = n_batch*2
        number_test_examples = mnist.test.num_examples
        test_indexs = np.random.randint(number_test_examples - n_chains*n_batch)
        test_samples = mnist.test.images[test_indexs:test_indexs+n_chains*n_batch]
        image_data = np.zeros((29*(n_samples+1)+1, 29*(n_chains)-1),
                          dtype="uint8")
        # Add the original images
        for i in range(n_batch):
            image_data[2*i*29:2*i*29+28,:] = tile_raster_images(X=test_samples[i*n_batch:(i+1)*n_chains],
                                            img_shape=(28, 28),
                                            tile_shape=(1, n_chains),
                                            tile_spacing=(1, 1))
            samples = sess.run(rbm.reconstruct(x), feed_dict={x:test_samples[i*n_batch:(i+1)*n_chains]})
            image_data[(2*i+1)*29:(2*i+1)*29+28,:] = tile_raster_images(X=samples,
                                            img_shape=(28, 28),
                                            tile_shape=(1, n_chains),
                                            tile_spacing=(1, 1))
        
        image = Image.fromarray(image_data)
        image.save("original_and_reconstruct.png")


    
