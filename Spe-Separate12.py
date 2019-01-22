# coding=utf-8
# https://www.zhihu.com/question/52200883/answer/229750418
# python2.7: TensorFlow-0.8.0


import tensorflow as tf
import numpy as np
import time

print "参数定义"
x = tf.placeholder(tf.int32, [None, 1000])
embedding = tf.get_variable('embedding', shape=[100, 25])
x_embedding = tf.nn.embedding_lookup(embedding,x)

source_sentence_length = tf.placeholder(tf.int32, [None])

encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=25)

encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, x_embedding,
                              sequence_length=source_sentence_length, dtype=tf.float32)

print "开始训练"
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    X_batch = np.random.randint(0, 100, size=[50, 1000])
    print "time0"
    time0 = time.time()

    for i in range(1):            #[10]*5: [10, 10, 10, 10, 10]
        encoder_outputs.eval(feed_dict={x: X_batch, source_sentence_length: [10]*50})
    print "time1"
    time1 = time.time()
    print('sequence_length_10, time: %.9f' % (time1-time0))
    print "time2"
    # sequence_length_10, time: 64.168038845

    time2 = time.time()
    for i in range(1):
        encoder_outputs.eval(feed_dict={x: X_batch, source_sentence_length: [100]*50})
    print "time3"
    time3 = time.time()
    print('sequence_length_100, time: %.9f' % (time3-time2))
    # sequence_length_1000, time: 63.966410875



