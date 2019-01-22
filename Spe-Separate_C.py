#coding=UTF-8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import time
# import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)   #默认与Mnist-11在一个文件夹: /home/chuwei/PycharmProjects/MNIST/MNIST_data/


# RNN学习时使用的参数
learning_rate = 0.001        #学习率
training_iters = 4000         #训练次数
batch_size =128              #每轮训练数据大小

# 神经网络的参数
n_input = 28        # 输入层的n,28步
n_steps = 28        # 28长度,每一步
n_hidden = 128      # 隐藏层神经元个数
n_classes = 10      # 输出的数量，因为是分类问题，0~9个数字，这里一共有10个


x = tf.placeholder("float", [None, n_steps, n_input])

# tensorflow里的LSTM需要两倍于n_hidden的长度的状态，一个state和一个cell
istate = tf.placeholder("float", [None, 2 * n_hidden])

y = tf.placeholder("float", [None, n_classes])

# 随机初始化每一层的权值和偏置
weights = {                  #字典
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),  # 从正态分布中输出随机值
    'out':    tf.Variable(tf.random_normal([n_hidden, n_classes]))
          }             #字典：键对值
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out':    tf.Variable(tf.random_normal([n_classes]))
         }              #biases是一维数组

'''
构建RNN
'''
def RNN(_X,_istate, _weights, _biases):
    # 规整输入的数据,命名可以以下划线作为开端
    # _X: batch_xs: batch*step*input
    # x1 = _X   #不能放在这里打印
    # n_steps* batch_size* n_input
    _X = tf.reshape(_X, [-1, batch_size, n_input])  # n_steps* batch_size* n_input
    print _X
    _X = tf.transpose(_X, [1, 0, 2])      # 改变顺序 n_steps and batch_size
    x1 = _X
    # (n_steps * batch_size, n_input)
    _X = tf.reshape(_X, [-1, n_input])  # (n_steps*batch_size, n_input) ,-1表示根据N_input来调整。
    print  _X
    # 输入层到隐含层，第一次是直接运算：n_steps * batch_size, n_hidden)
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # 之后使用LSTM
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # 初始化为零值， lstm 单元由两个部分组成： (c_state, h_state)
    # init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # tf.split(dimension, num_split, input)：dimension的意思就是输入张量的哪一个维度，
    # 如果是0就表示对第0维度进行切割。num_split就是切割的数量，如果是2就表示输入张量被切成2份，
    # 每一份是一个列表。

    # 分割后得到list：n_steps* batch_size* n_hidden
    _X = tf.split(0, n_steps, _X)  # 得到list

    # _X = tf.reshape(_X,[n_steps, n_hidden, -1])  #array
    # _X = list(_X)   # array到list的转换

    # 开始跑RNN那部分
    outputs, states = tf.nn.rnn(lstm_cell, _X, sequence_length=None, initial_state=_istate)
    # outputs, states = tf.nn.dynamic_rnn(lstm_cell, _X, initial_state =_istate, time_major = False)
    # tf.nn.dynamic_rnn()  #dynamic-rnn可以允许不同batch的sequence length不同，但rnn不能。

    # 输出层：outputs[-1]: batch*hidden , results: batch*classes
    results = tf.matmul(outputs[-1], _weights['out']) + _biases['out']

    return results , x1

'''
#预测值，调用RNN()
'''
pred ,x_new= RNN(x,istate, weights, biases)

#定义损失函数和优化方法，其中损失函数为softmax交叉熵，优化方法为Adam
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))  # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer

#进行模型的评估，argmax是取出取值最大的那一个的标签作为输出
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  #判断预测值索引与实际值索引是否相等, http://blog.csdn.net/qq575379110/article/details/70538051
# tf.equal()返回 True or False
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  #tf.cast:将correct_pred的数据格式转化成float32

# tf.argmax(pred, 0)    #表示每列最大的索引位置
arg_max_pre = tf.argmax(pred, 1)
arg_max_y   = tf.argmax(y, 1)

# 初始化
init = tf.initialize_all_variables()
# 即initialize_variables(all_variables())

print "'''开始计时'''"
start_time_0 = time.time()   #计时
''''
开始运行
'''
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # 持续迭代
    while step * batch_size < training_iters:
        start_time = time.time()  # 计时
        # 随机抽出这一次迭代训练时用的数据
        #batch_xs: batch*784, batch_ys: batch*classes
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)   #通过next_batch()就可以一个一个batch的拿数据
        # 对数据进行处理，使得其符合输入
        print "xx:", batch_xs.shape
        #batch_xs: batch*step*input
        batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
        print "x:", batch_xs.shape
        print "y:", batch_ys.shape

        # 迭代
        #给x、y、istate赋值，并运行
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,     #字典：采用键对值
                                       istate: np.zeros((batch_size, 2 * n_hidden))})
        cor=sess.run(correct_pred, feed_dict={x: batch_xs, y: batch_ys,
                                       istate: np.zeros((batch_size, 2 * n_hidden))})
        # print "correct_pred:",cor
        # arg_pre = sess.run(arg_max_pre, feed_dict={x: batch_xs, y: batch_ys,
        #                                            istate: np.zeros((batch_size, 2 * n_hidden))})
        # arg_y = sess.run(arg_max_y, feed_dict={x: batch_xs, y: batch_ys,
        #                                        istate: np.zeros((batch_size, 2 * n_hidden))})
        # print "arg_pre:\n ", arg_pre   # batch_size 个数
        # print "arg_y：\n ",  arg_y     # batch_size 个数
        # print "arg_pre.shape:\n ", arg_pre.shape
        predict = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys,
                                            istate: np.zeros((batch_size, 2 * n_hidden))})
        print "prediction:",predict.shape


        x_now = sess.run(x_new, feed_dict={x: batch_xs, y: batch_ys,
                                          istate: np.zeros((batch_size, 2 * n_hidden))})
        # print " x_now.shape :", np.array(x_now).shape
        print " x_now.shape :", x_now.shape

        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                          istate: np.zeros((batch_size, 2 * n_hidden))})
        Acc = acc
        # list1.append(step * batch_size)  # 添加到列表中,画图用
        # list2.append(Acc)
        print (step * batch_size)
        print "Acc:",Acc

        step = step+1
        print "训练耗时：", (time.time() - start_time)
    print ("Optimization Finished!")

print "训练全部耗时：", (time.time() - start_time_0)
print "平均每批batch耗时：", (time.time() - start_time_0)/(step-1)



