from __future__ import print_function
import cv2
import tensorflow as tf
import numpy as np


IMG_SIZE = 128  # 图像大小
LABEL_CNT = 2  # 标签类别的数量
P_KEEP_INPUT = 0.8  # 输入dropout层保持比例
P_KEEP_HIDDEN = 0.5  # 隐层dropout的保持比例


# 获取并初始化权重
def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


X = tf.placeholder("float", [None, IMG_SIZE, IMG_SIZE, 3])
Y = tf.placeholder("float", [None, 2])

w = init_weights([3, 3, 3, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([3, 3, 128, 128])
w5 = init_weights([4 * 4 * 128, 625])
w_o = init_weights([625, 2])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")


# 简单的卷积模型
def simple_model(X, w, w_2, w_3, w_4, w_5, w_o, p_keep_input, p_keep_hidden):
    # batchsize * 128 * 128 * 3
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    # 2x2 max_pooling
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # dropout
    l1 = tf.nn.dropout(l1, p_keep_input)  # 64 * 64 * 32

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w_2, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_hidden)  # 32 * 32 * 64

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w_3, strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.nn.dropout(l3, p_keep_hidden)  # 16 * 16 * 128

    l4a = tf.nn.relu(tf.nn.conv2d(l3, w_4, strides=[1, 1, 1, 1], padding='SAME'))
    l4 = tf.nn.max_pool(l4a, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')  # 4 * 4 * 128
    l4 = tf.reshape(l4, [-1, w_5.get_shape().as_list()[0]])

    l5 = tf.nn.relu(tf.matmul(l4, w_5))
    l5 = tf.nn.dropout(l5, p_keep_hidden)

    return tf.matmul(l5, w_o)


# y_pred是预测tensor
y_pred = simple_model(X, w, w2, w3, w4, w5, w_o, p_keep_input, p_keep_hidden)




# # image = cv2.imread('0_train/neg/IMG_3101.jpg', 1)
# image = cv2.imread('0_train/pos/IMG_3114.jpg', 1)
# image = cv2.resize(image, (128, 128))
# sess = tf.Session()
# saver = tf.train.Saver()
# saver.restore(sess, './0_train/graph.ckpt-30')
# result = sess.run(y_pred, feed_dict={X: [image], p_keep_hidden: 1.0, p_keep_input: 1.0})
# print(result)
# print(result.argmax())

# cap = cv2.VideoCapture("video.mp4")
cap = cv2.VideoCapture(0)
count = 0
while True:
    _, image = cap.read()
    cv2.imshow('test', image)
    image = cv2.resize(image, (128, 128))

    if count % 5 == 0:

        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, './0_train/graph.ckpt-5')
        result = sess.run(y_pred, feed_dict={X: [image], p_keep_hidden: 1.0, p_keep_input: 1.0})
        print(result)
        print(count, result.argmax())


    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

