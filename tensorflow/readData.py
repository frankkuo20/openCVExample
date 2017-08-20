import tensorflow as tf
import numpy as np
import os


# http://www.foxhunter.xyz/2017/05/05/%E7%9E%8E%E7%8E%A9tensorflow%EF%BC%9A%E7%AE%80%E5%8D%95%E7%9A%84cnn%E8%AF%86%E5%88%AB%E8%89%B2%E6%83%85%E5%9B%BE%E7%89%87/
#https://www.zhihu.com/question/51693189

def getFiles(fileDir):
    '''
    
    :param fileDir: 
    :return: imageList, labelList
    '''
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(fileDir):
        name = file.split('.')
        if name[0]=='cat':
            cats.append(fileDir+file)
            label_cats.append(0)
        else:
            dogs.append(fileDir+file)
            label_dogs.append(1)

    print('count cats: {}, dogs: {} '.format(len(cats), len(dogs)))

    imageList = np.hstack((cats, dogs))
    labelList = np.hstack((label_cats, label_dogs))

    temp = np.array([imageList, labelList])
    temp = temp.transpose()
    np.random.shuffle(temp)

    imageList = list(temp[:, 0])
    labelList = list(temp[:, 1])
    labelList = [int(float(i)) for i in labelList]

    return imageList, labelList



def getBatch(image, label, image_W, image_H, batch_size, capacity):
    '''
    
    :param image: 
    :param label: 
    :param image_w: 
    :param image_H: 
    :param batch_size: 
    :param capacity: 
    :return: 
    '''
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])

    # 1: output a grayscale image.
    # 3: output an RGB image.
    image = tf.image.decode_jpeg(image_contents, channels=1)

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

    # image_batch = tf.reshape(image_batch, [batch_size, 2, -1])
    image_batch = tf.reshape(image_batch, [batch_size, -1])

    # label_batch = tf.reshape(label_batch, [batch_size])
    print(label_batch)

    label_batch = tf.reshape(label_batch, [-1, 2])
    print(label_batch)
    return image_batch, label_batch



if __name__=='__main__':
    BATCH_SIZE = 200
    CAPACITY = 256
    IMG_W = 208
    IMG_H = 208
    train_dir = 'data/'

    imageList, labelList = getFiles(train_dir)

    image_batch, label_batch = getBatch(imageList, labelList,
                                        IMG_W,
                                        IMG_H,
                                        BATCH_SIZE,
                                        CAPACITY)
    # ? 43264
    x = tf.placeholder(tf.float32, [None, IMG_W * IMG_H])

    # 43264 2

    W = tf.Variable(tf.zeros([IMG_W * IMG_H, 2]))

    b = tf.Variable(tf.zeros([2]))

    y = tf.nn.softmax(tf.matmul(x, W)+b)
    print(y)

    y_ = tf.placeholder(tf.int64, [None, 2])  # right answer

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=tf.squeeze(y_)))

    # 梯度下降法gradient descent algorithm
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # # Operation comparing prediction with true label
    # correct_prediction = tf.equal(tf.argmax(cross_entropy, 1), y_)
    #
    # # Operation calculating the accuracy of our predictions
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        i = 0
        try:
            while not coord.should_stop() and i < 2 :


                img, label = sess.run([image_batch, label_batch])

                sess.run(train_step, feed_dict={x: img, y_: label})

                # for j in np.arange(BATCH_SIZE):
                #
                #     sess.run(train_step, feed_dict={x: img, y_: label[j]})
                #     if i % 100 == 0:
                #         train_accuracy = sess.run(accuracy, feed_dict={x: image_batch, y_: label_batch})
                #         print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))

                # sess.run(train_step, feed_dict={x: img, y_: label})
                # if i % 100 == 0:
                #     train_accuracy = sess.run(accuracy, feed_dict={x: image_batch, y_: label_batch})
                #     print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))

                # img, label = sess.run([image_batch, label_batch])
                # for j in np.arange(BATCH_SIZE):
                #     print(label)



                i += 1


        except tf.errors.OutOfRangeError:
            print('done')
        finally:
            coord.request_stop()

        coord.join(threads)
















