from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

class DeepCNN(object):
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

    def first_conv_layer(self):
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        x_image = tf.reshape(self.x, [-1,28,28,1])

        h_conv1 = tf.nn.relu(self.conv2d(x_image,W_conv1)+b_conv1)

        h_pool1 = self.max_pool_2x2(h_conv1)

        return h_pool1


    def second_conv_layer(self,h_pool1):
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        return h_pool2

    def fully_connected_layer(self, h_pool2):

        W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    def dropout(self, h_fc1):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        return h_fc1_drop

    def readout_layer(self,h_fc1_drop):
        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        return y_conv


    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shap):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)