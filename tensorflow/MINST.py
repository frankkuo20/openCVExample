# http://ithelp.ithome.com.tw/users/20103494/ironman/1231
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


x = tf.placeholder(tf.float32, [None, 784])  # 28*28
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W)+b)

y_ = tf.placeholder(tf.float32, [None, 10])  # right answer


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

# 梯度下降法gradient descent algorithm
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


init = tf.global_variables_initializer()

sess = tf.Session()

writer = tf.summary.FileWriter("logs/", graph=sess.graph)

sess.run(init)
for i in range(500):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


