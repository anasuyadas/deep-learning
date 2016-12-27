from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


class SoftmaxReg(object):
    def __init__(self):
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.init_variables()
        self.y = tf.matmul(self.x, self.W) + self.b
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.loss = self.cross_entropy_loss()

    def init_variables(self):
        self.W = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))
        return self

    def cross_entropy_loss(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_))
        return loss


softmax = SoftmaxReg()
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(softmax.loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = softmax.mnist.train.next_batch(100)
    sess.run(fetches=[train_step, softmax.loss], feed_dict={softmax.x: batch_xs, softmax.y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(softmax.y, 1), tf.argmax(softmax.y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={softmax.x: softmax.mnist.test.images,
                                    softmax.y_: softmax.mnist.test.labels}))
