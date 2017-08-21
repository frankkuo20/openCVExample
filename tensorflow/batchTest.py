import tensorflow as tf

IMG_SIZE = 128

X = tf.placeholder("float", [None, IMG_SIZE, IMG_SIZE, 3])
Y = tf.placeholder("float", [None, 2])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

# y_pred是预测tensor
y_pred = simple_model(X, w, w2, w3, w4, w5, w_o, p_keep_input, p_keep_hidden)

# 定义损失函数为交叉熵。
# 注意simple_model最后返回的不包含softmax操作，
# softmax_cross_entropy_with_logits会自动做softmax。
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_pred))
# 定义精度
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    example, label = read_and_decode(filename_queue)
    min_after_dequeue = 1000
    num_threads = 4
    capacity = min_after_dequeue + (num_threads + 3) * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity, num_threads = num_threads,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch


# 每batch随机取500张
test_img_batch, test_label_batch = input_pipeline(["./0_train/test.bin"], 500)
with tf.Session() as sess:
    # 加载模型。模型的文件名称看下本地情况
    saver.restore(sess, './0_train/graph.ckpt-1000')

    coord_test = tf.train.Coordinator()
    threads_test = tf.train.start_queue_runners(coord=coord_test)
    test_imgs, test_labels = sess.run([test_img_batch, test_label_batch])
    # 预测阶段，keep取值均为1
    acc = sess.run(accuracy, feed_dict={X: test_imgs, Y: test_labels, p_keep_hidden: 1.0, p_keep_input: 1.0})
    print("predict accuracy is %.2f" % acc)
    coord_test.request_stop()
    coord_test.join(threads_test)



