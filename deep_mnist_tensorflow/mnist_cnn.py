from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


class DeepCNN(object):
    def __init__(self):
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)
        self.y_conv = self.cnn()
        self.loss = self.cross_entropy_loss()

    def cross_entropy_loss(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_conv, self.y_))
        return loss

    def first_conv_layer(self):
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)

        h_pool1 = self.max_pool_2x2(h_conv1)

        return h_pool1

    def second_conv_layer(self, h_pool1):
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        return h_pool2

    def fully_connected_layer(self, h_pool2):
        W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        return h_fc1

    def dropout(self, h_fc1):
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        return h_fc1_drop

    def readout_layer(self, h_fc1_drop):
        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        return y_conv

    def cnn(self):
        h_pool1 = self.first_conv_layer()
        h_pool2 = self.second_conv_layer(h_pool1)
        h_fc1 = self.fully_connected_layer(h_pool2)
        h_fc1_drop = self.dropout(h_fc1)
        y_conv = self.readout_layer(h_fc1_drop)

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
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


cnn = DeepCNN()
train_step = tf.train.AdamOptimizer(1e-4).minimize(cnn.loss)
correct_prediction = tf.equal(tf.argmax(cnn.y_conv, 1), tf.argmax(cnn.y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(200):
    batch = cnn.mnist.train.next_batch(50)
    if i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict={cnn.x: batch[0], cnn.y_: batch[1], cnn.keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(feed_dict={cnn.x: batch[0], cnn.y_: batch[1], cnn.keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(
    feed_dict={cnn.x: cnn.mnist.test.images, cnn.y_: cnn.mnist.test.labels, cnn.keep_prob: 1.0}))
