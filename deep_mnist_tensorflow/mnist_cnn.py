from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

class DeepCNN(object):
    def __init__(self):
        self.input_data = input_data
        self.x = tf.placeholder(tf.float32,shape=[None,784])
        self.y = tf.placeholder(tf.float32,shape=[None,10])

    def init_variables(self):
        self.w = tf.Variable(tf.zeros([784,10]))
        self.b = tf.Variable(tf.zeros([10]))
        return self

    def softmax_regression =


