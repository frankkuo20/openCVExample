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
    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)




    label_batch = tf.reshape(label_batch, [-1, 2])

    return image_batch, label_batch



IMG_SIZE = 128  # 图像大小
LABEL_CNT = 2  # 标签类别的数量
P_KEEP_INPUT = 0.8  # 输入dropout层保持比例
P_KEEP_HIDDEN = 0.5  # 隐层dropout的保持比例

BATCH_SIZE = 200
min_after_dequeue = 1000
num_threads = 4
CAPACITY = min_after_dequeue + (num_threads + 3) * BATCH_SIZE


train_dir = 'data/'

imageList, labelList = getFiles(train_dir)

image_batch, label_batch = getBatch(imageList, labelList,
                                    IMG_SIZE,
                                    IMG_SIZE,
                                    BATCH_SIZE,
                                    CAPACITY)

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

# 定义损失函数为交叉熵。
# 注意simple_model最后返回的不包含softmax操作，
# softmax_cross_entropy_with_logits会自动做softmax。
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_pred))
# 定义精度
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# rmsprop方式对最小化损失函数
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

disp_step = 5
save_step = 20
max_step = 1000  # 最大迭代次数
step = 0
saver = tf.train.Saver()  # 用来保存模型的


with tf.Session() as sess:
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())

    # start_queue_runnes读取数据，具体用法参见官网
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:

        # 获取训练数据成功，并且没有到达最大训练次数
        while not coord.should_stop() and step < max_step:
            step += 1
            # 运行tensor，获取数据
            print(step)
            imgs, labels = sess.run([image_batch, label_batch])
            # 训练。训练时dropout层要有值。
            sess.run(train_op, feed_dict={X: imgs, Y: labels, p_keep_hidden: P_KEEP_HIDDEN, p_keep_input: P_KEEP_INPUT})
            if epoch % disp_step == 0:
                print('in')
                # 输出当前batch的精度。预测时keep的取值均为1
                acc = sess.run(accuracy, feed_dict={X: imgs, Y: labels, p_keep_hidden: 1.0, p_keep_input: 1.0})
                print('%s accuracy is %.2f' % (step, acc))
            if step % save_step == 0:
                # 保存当前模型
                save_path = saver.save(sess, './0_train/graph.ckpt', global_step=step)
                print("save graph to %s" % save_path)

    except tf.errors.OutOfRangeError:
        print("reach epoch limit")
    finally:
        coord.request_stop()
    coord.join(threads)
    save_path = saver.save(sess, './0_train/graph.ckpt', global_step=epoch)

print("training is done")















