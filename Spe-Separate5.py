#coding=UTF-8
# 运行环境： python2.7 , tensorflow-v0.8


from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.contrib import rnn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import xlrd


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# RNN学习时使用的参数
learning_rate = 0.001        #学习率
training_iters = 600         #训练次数
batch_size =128              #每轮训练数据大小

# 神经网络的参数
n_inputs = 28        # 输入层的n,28步
n_steps = 28        # 28长度,每一步
n_hidden = 128      # 隐藏层神经元个数
n_classes = 10      # 输出的数量,10类


# placeholder（type,strucuct)第一个参数是要保存的数据的数据类型,后面的参数是数据的结构，如1×2的矩阵
# 它在使用的时候和variable不同的是在session运行阶段，利用feed_dict的字典结构给placeholdr变量数据
x = tf.placeholder( tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder( tf.float32, [None, n_classes])
# tf.float32
# 随机初始化每一层的权值和偏置
weights = {              #字典
    'hidden': tf.Variable(tf.random_normal([n_inputs, n_hidden])),
    'out':    tf.Variable(tf.random_normal([n_hidden, n_classes]))
          }             #字典：键对值
# _biases = {
#     'hidden': tf.Variable(tf.constant(0.1, shape=[n_hidden, 1])),
#     'out':    tf.Variable(tf.constant(0.1, shape=[n_classes,1]))
#          }              #biases是一维数组

biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out':    tf.Variable(tf.random_normal([n_classes]))
         }              #biases是一维数组

'''
构建RNN
'''
def RNN(X, weights, biases):
    '''' 把输入的 X 转换成 X ==> (128 batch * 28 steps, 28 inputs) '''''
    # x1 = X   #放在这里不行
    X = tf.reshape(X, [batch_size*n_steps, n_inputs])   #128*28*28
    x1 = X
    # 进入隐藏层
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['hidden']) + biases['hidden']
    # X_in == > (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden])


    # 这里采用基本的 LSTM 循环网络单元： basic LSTM Cell
    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0,state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # 初始化为零值， lstm 单元由两个部分组成： (c_state, h_state)

    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # dynamic_rnn 接收张量(batch, steps, inputs)或者(steps, batch, inputs)作为 X_in
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, sequence_length=None,
                                     initial_state =init_state, time_major = False)

    outputs = tf.transpose(outputs, [1, 0, 2])
    x1 = outputs[n_steps-1,:,:]

    results = tf.matmul(outputs[n_steps-1,:,:], weights['out']) + biases['out']
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    return results ,x1

'''
#预测值，调用RNN()
'''
# 定义损失函数和优化器，优化器采用AdamOptimizer
pred ,x_new = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 定义模型预测结果及准确率计算方法：
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

arg_max_pre = tf.argmax(pred, 1)
arg_max_y   = tf.argmax(y, 1)

'''time计时'''
start_time = time.time()

''''
开始运行
'''
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    step = 0
    # 持续迭代
    while step * batch_size < training_iters:
        # 随机抽出这一次迭代训练时用的数据
        #batch_xs: batch*784, batch_ys: batch*classes
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)   #通过next_batch()就可以一个一个batch的拿数据
        # 对数据进行处理，使得其符合输入
        print "x:", batch_xs
        #batch_xs: batch*step*input
        batch_xs = batch_xs.reshape((batch_size, n_steps, n_inputs)) #这里没有加”[]“号
        # print "x:", type(batch_xs)
        print "y:", batch_ys

        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, })    #字典：采用键对值

        cost_1=sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, })
        predict = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys, })
        print "predict:",predict
        print "cost:",cost_1

        x_now = sess.run(x_new, feed_dict={x: batch_xs, y: batch_ys, })
        print " x_now.shape :", x_now


        # acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,  })
        # Acc = acc
        # print "Acc:",Acc

        print "step * batch_size:", step * batch_size
        step += 1

    print ("Optimization Finished!")

print "训练耗时：", (time.time() - start_time)

    # '''# 载入测试集进行测试'''
    # # Calculate accuracy for 128 mnist test images
    # test_data, test_label = mnist.test.next_batch(batch_size)  # 通过next_batch()就可以一个一个batch的拿数据
    # test_data= test_data.reshape((batch_size, n_steps, n_input))

    # batch_size = 1000
    # test_len = batch_size
    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    # test_label = mnist.test.labels[:test_len]
    # print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,})






