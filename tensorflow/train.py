import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# step 1
filenames = ['data/pi3box.jpg', 'data/pi3box2.jpg',
             'data/pi3box.jpg', 'data/pi3box2.jpg']

# step 2
filename_queue = tf.train.string_input_producer(filenames)

# step 3: read, decode and resize images
reader = tf.WholeFileReader()
filename, content = reader.read(filename_queue)

image = tf.image.decode_jpeg(content, channels=3)

image = tf.cast(image, tf.float32)
image = tf.image.resize_images(image, [224, 224])


x = tf.placeholder(tf.float32, [None, 224*224])  # 28*28
y_ = tf.placeholder(tf.float32, [None, 10])  # right answer

W = tf.Variable(tf.zeros([224*224, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W)+b)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

# 梯度下降法gradient descent algorithm
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

# image.set_shape((224, 224, 3))
# batch_size = 50
# num_preprocess_threads = 1
# min_queue_examples = 256
# images = tf.train.shuffle_batch(
#     [image],
#     batch_size=batch_size,
#     num_threads=num_preprocess_threads,
#     capacity=min_queue_examples + 3 * batch_size,
#     min_after_dequeue=min_queue_examples
# )

with tf.Session() as sess:
    # Required to get the filename matching to run.
    sess.run(init)
    for i in range(1000):
        image_batch = tf.train.batch([image], batch_size=8)
        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(train_step, feed_dict={x: image_batch, y_: threads})
        # Get an image tensor and print its value.
        image_tensor = sess.run([image])
        print(image_tensor)

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)





'''
image = tf.cast(image, tf.float32)
resized_image = tf.image.resize_images(image, [224, 224])

# step 4: Batching
image_batch = tf.train.batch([resized_image], batch_size=8)
'''


