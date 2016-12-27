from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

class SoftmaxReg(object):
    def __init__(self):
        self.input_data = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.x = tf.placeholder(tf.float32,shape=[None,784])
        self.y = tf.placeholder(tf.float32,shape=[None,10])
        self.y_ = tf.placeholder(tf.float32, shape = [None, 10])

    def init_variables(self):
        self.W = tf.Variable(tf.zeros([784,10])).initialized_value()
        self.b = tf.Variable(tf.zeros([10])).initialized_value()
        return self

    def fit(self):
        self.y = tf.matmul(self.x, self.W) + self.b
        return self

    def cross_entropy_loss(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_))
        return loss

    def train_step(self):
        step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy_loss())
        return step


softmax = SoftmaxReg().init_variables().fit()
sess = tf.InteractiveSession()
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(softmax.cross_entropy_loss())
for _ in range(1000):
    batch_xs, batch_ys = softmax.input_data.train.next_batch(100)
    sess.run(train_step, feed_dict={softmax.x: batch_xs, softmax.y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(softmax.y, 1), tf.argmax(softmax.y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={softmax.x: softmax.input_data.test.images,
                                      softmax.y_: softmax.input_data.test.labels}))


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                 reduction_indices=[1]))
#
# can be numerically unstable.
#
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'y', and then average across the batch.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                  y_: mnist.test.labels}))