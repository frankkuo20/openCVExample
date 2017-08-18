import tensorflow as tf
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

file_path = 'data/pi3box.jpg'

image = cv2.imread(file_path, 1)
cv2.namedWindow('image', 0)
cv2.imshow('image', image)

x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    x = tf.transpose(x, perm=[1, 0, 2])
    session.run(model)
    result = session.run(x)

cv2.namedWindow('result', 0)
cv2.imshow('result', result)
cv2.waitKey(0)



