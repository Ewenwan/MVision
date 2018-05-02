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

class RBM(object):
    """A Restricted Boltzmann Machines class"""
    def __init__(self, inpt=None, n_visiable=784, n_hidden=500, W=None,
                 hbias=None, vbias=None):
        """
        :param inpt: Tensor, the input tensor [None, n_visiable]
        :param n_visiable: int, number of visiable units
        :param n_hidden: int, number of hidden units
        :param W, hbias, vbias: Tensor, the parameters of RBM (tf.Variable)
        """
        self.n_visiable = n_visiable
        self.n_hidden = n_hidden
        # Optionally initialize input
        if inpt is None:
            inpt = tf.placeholder(dtype=tf.float32, shape=[None, self.n_visiable])
        self.input = inpt
        # Initialize the parameters if not given
        if W is None:
            bounds = -4.0 * np.sqrt(6.0 / (self.n_visiable + self.n_hidden))
            W = tf.Variable(tf.random_uniform([self.n_visiable, self.n_hidden], minval=-bounds,
                                              maxval=bounds), dtype=tf.float32)
        if hbias is None:
            hbias = tf.Variable(tf.zeros([self.n_hidden,]), dtype=tf.float32)
        if vbias is None:
            vbias = tf.Variable(tf.zeros([self.n_visiable,]), dtype=tf.float32)
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        # keep track of parameters for training (DBN)
        self.params = [self.W, self.hbias, self.vbias]
    
    def propup(self, v):
        """Compute the sigmoid activation for hidden units given visible units"""
        return tf.nn.sigmoid(tf.matmul(v, self.W) + self.hbias)

    def propdown(self, h):
        """Compute the sigmoid activation for visible units given hidden units"""
        return tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.vbias)
    
    def sample_prob(self, prob):
        """Do sampling with the given probability (you can use binomial in Theano)"""
        return tf.nn.relu(tf.sign(prob - tf.random_uniform(tf.shape(prob))))
    
    def sample_h_given_v(self, v0_sample):
        """Sampling the hidden units given visiable sample"""
        h1_mean = self.propup(v0_sample)
        h1_sample = self.sample_prob(h1_mean)
        return (h1_mean, h1_sample)
    
    def sample_v_given_h(self, h0_sample):
        """Sampling the visiable units given hidden sample"""
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.sample_prob(v1_mean)
        return (v1_mean, v1_sample)
    
    def gibbs_vhv(self, v0_sample):
        """Implement one step of Gibbs sampling from the visiable state"""
        h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return (h1_mean, h1_sample, v1_mean, v1_sample)

    def gibbs_hvh(self, h0_sample):
        """Implement one step of Gibbs sampling from the hidden state"""
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return (v1_mean, v1_sample, h1_mean, h1_sample)

    def free_energy(self, v_sample):
        """Compute the free energy"""
        wx_b = tf.matmul(v_sample, self.W) + self.hbias
        vbias_term = tf.matmul(v_sample, tf.expand_dims(self.vbias, axis=1))
        hidden_term = tf.reduce_sum(tf.log(1.0 + tf.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def get_train_ops(self, learning_rate=0.1, k=1, persistent=None):
        """
        Get the training opts by CD-k
        :params learning_rate: float
        :params k: int, the number of Gibbs step (Note k=1 has been shown work surprisingly well)
        :params persistent: Tensor, PCD-k (TO DO:)
        """
        # Compute the positive phase
        ph_mean, ph_sample = self.sample_h_given_v(self.input)
        # The old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        # Use tf.while_loop to do the CD-k
        cond = lambda i, nv_mean, nv_sample, nh_mean, nh_sample: i < k
        body = lambda i, nv_mean, nv_sample, nh_mean, nh_sample: (i+1, ) + self.gibbs_hvh(nh_sample)
        i, nv_mean, nv_sample, nh_mean, nh_sample = tf.while_loop(cond, body, loop_vars=[tf.constant(0), tf.zeros(tf.shape(self.input)), 
                                                            tf.zeros(tf.shape(self.input)), tf.zeros(tf.shape(chain_start)), chain_start])
        """
        # Compute the update values for each parameter
        update_W = self.W + learning_rate * (tf.matmul(tf.transpose(self.input), ph_mean) - 
                                tf.matmul(tf.transpose(nv_sample), nh_mean)) / tf.to_float(tf.shape(self.input)[0])  # use probability
        update_vbias = self.vbias + learning_rate * (tf.reduce_mean(self.input - nv_sample, axis=0))   # use binary value
        update_hbias = self.hbias + learning_rate * (tf.reduce_mean(ph_mean - nh_mean, axis=0))       # use probability
        # Assign the parameters new values
        new_W = tf.assign(self.W, update_W)
        new_vbias = tf.assign(self.vbias, update_vbias)
        new_hbias = tf.assign(self.hbias, update_hbias)
        """
        chain_end = tf.stop_gradient(nv_sample)   # do not compute the gradients
        cost = tf.reduce_mean(self.free_energy(self.input)) - tf.reduce_mean(self.free_energy(chain_end))
        # Compute the gradients
        gparams = tf.gradients(ys=[cost], xs=self.params)
        new_params = []
        for gparam, param in zip(gparams, self.params):
            new_params.append(tf.assign(param, param - gparam*learning_rate))

        if persistent is not None:
            new_persistent = [tf.assign(persistent, nh_sample)]
        else:
            new_persistent = []
        return new_params + new_persistent  # use for training

    def get_reconstruction_cost(self):
        """Compute the cross-entropy of the original input and the reconstruction"""
        activation_h = self.propup(self.input)
        activation_v = self.propdown(activation_h)
        # Do this to not get Nan
        activation_v_clip = tf.clip_by_value(activation_v, clip_value_min=1e-30, clip_value_max=1.0)
        reduce_activation_v_clip = tf.clip_by_value(1.0 - activation_v, clip_value_min=1e-30, clip_value_max=1.0)
        cross_entropy = -tf.reduce_mean(tf.reduce_sum(self.input*(tf.log(activation_v_clip)) + 
                                    (1.0 - self.input)*(tf.log(reduce_activation_v_clip)), axis=1))
        return cross_entropy   
    def reconstruct(self, v):
        """Reconstruct the original input by RBM"""
        h = self.propup(v)
        return self.propdown(h) 

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
    rbm = RBM(x, n_visiable=n_visiable, n_hidden=n_hidden)
    
    learning_rate = 0.1
    batch_size = 20
    cost = rbm.get_reconstruction_cost()
    # Create the persistent variable
    persistent_chain = tf.Variable(tf.zeros([batch_size, n_hidden]), dtype=tf.float32)
    train_ops = rbm.get_train_ops(learning_rate=learning_rate, k=15, persistent=persistent_chain)
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
            image.save("new_filters_at_epoch_{0}.png".format(epoch))

        end_time = timeit.default_timer()
        training_time = end_time - start_time
        print("Finished!")
        print("  The training ran for {0} minutes.".format(training_time/60,))
        # Reconstruct the image by sampling
        print("...Sampling from the RBM")
        # the 
        n_chains = 20
        n_samples = 10
        number_test_examples = mnist.test.num_examples
        # Randomly select the 'n_chains' examples
        test_indexs = np.random.randint(number_test_examples - n_chains)
        test_samples = mnist.test.images[test_indexs:test_indexs+n_chains]
        # Create the persistent variable saving the visiable state
        persistent_v_chain = tf.Variable(tf.to_float(test_samples), dtype=tf.float32)
        # The step of Gibbs
        step_every = 1000
        # Inplement the Gibbs
        cond = lambda i, h_mean, h_sample, v_mean, v_sample: i < step_every
        body = lambda i, h_mean, h_sample, v_mean, v_sample: (i+1, ) + rbm.gibbs_vhv(v_sample)
        i, h_mean, h_sample, v_mean, v_sample = tf.while_loop(cond, body, loop_vars=[tf.constant(0), tf.zeros([n_chains, n_hidden]), 
                                                            tf.zeros([n_chains, n_hidden]), tf.zeros(tf.shape(persistent_v_chain)), persistent_v_chain])
        # Update the persistent_v_chain
        new_persistent_v_chain = tf.assign(persistent_v_chain, v_sample)
        # Store the image by sampling
        image_data = np.zeros((29*(n_samples+1)+1, 29*(n_chains)-1),
                          dtype="uint8")
        # Add the original images
        image_data[0:28,:] = tile_raster_images(X=test_samples,
                                            img_shape=(28, 28),
                                            tile_shape=(1, n_chains),
                                            tile_spacing=(1, 1))
        # Initialize the variable
        sess.run(tf.variables_initializer(var_list=[persistent_v_chain]))
        # Do successive sampling
        for idx in range(1, n_samples+1):
            sample = sess.run(v_mean)
            sess.run(new_persistent_v_chain)
            print("...plotting sample", idx)
            image_data[idx*29:idx*29+28,:] = tile_raster_images(X=sample,
                                            img_shape=(28, 28),
                                            tile_shape=(1, n_chains),
                                            tile_spacing=(1, 1))
        image = Image.fromarray(image_data)
        image.save("new_original_and_{0}samples.png".format(n_samples))
