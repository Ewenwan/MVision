"""
Logistic Regression
author: Ye Hu
2016/12/14  update 2017/02/16
redit：wanyouwen 2018/05/02
"""
import numpy as np
import tensorflow as tf
import input_data

class LogisticRegression(object):
    """Multi-class logistic regression class"""
    def __init__(self, inpt, n_in, n_out):
        """
        inpt: tf.Tensor, (one minibatch) [None, n_in]
        n_in: int, number of input units
        n_out: int, number of output units
        """
        # weight
        self.W = tf.Variable(tf.zeros([n_in, n_out], dtype=tf.float32))
        # bias
        self.b = tf.Variable(tf.zeros([n_out,]), dtype=tf.float32)
        # activation output
        self.output = tf.nn.softmax(tf.matmul(inpt, self.W) + self.b)
        # prediction
        self.y_pred = tf.argmax(self.output, axis=1)
        # keep track of variables
        self.params = [self.W, self.b]

    def cost(self, y):
        """
        y: tf.Tensor, the target of the input
        """
        # cross_entropy
        return -tf.reduce_mean(tf.reduce_sum(y * tf.log(self.output), axis=1))

    def accuarcy(self, y):
        """errors"""
        correct_pred = tf.equal(self.y_pred, tf.argmax(y, axis=1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))




if __name__ == "__main__":
    # Load mnist dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # Define placeholder for input and target
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # Construct model
    classifier = LogisticRegression(x, n_in=784, n_out=10)
    cost = classifier.cost(y_)
    accuracy = classifier.accuarcy(y_)
    predictor = classifier.y_pred
    # Define the train operation
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(
        cost, var_list=classifier.params)

    # Initialize all variables
    init = tf.global_variables_initializer()

    # Training settings
    training_epochs = 50
    batch_size = 100
    display_step = 5

    # Train loop
    print("Start to train...")
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0.0
            batch_num = int(mnist.train.num_examples/batch_size)
            for i in range(batch_num):
                x_batch, y_batch = mnist.train.next_batch(batch_size)
                # Run train op
                c, _ = sess.run([cost, train_op], feed_dict={x: x_batch, y_: y_batch})
                # Sum up cost
                avg_cost += c/batch_num

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

