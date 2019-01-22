#coding=UTF-8
# 运行环境： python2.7 , tensorflow-v0.8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import time


'''读取Excel文件'''
data1 = xlrd.open_workbook('./语音图片1/TIMIT-sa1-syn.xls')
data2 = xlrd.open_workbook('./语音图片1/TIMIT-sa1.xls')
print "data:", data1
table1 = data1.sheets()[0]    #选择工作表sheet1
table2 = data2.sheets()[0]
print "table:", np.array(table1)
'''计算行列数'''
rows_num = table1.nrows
cols_num = table1.ncols
print "行数列数：", rows_num, cols_num
# 行数列数： 513 91

'''定义函数：将Excel转换为array格式'''
def datas_array(_rows_num,_cols_num,table):
    A = np.array([])   #定义空数组
    i=0
    while i < _rows_num:
        value = table.row_values(i)[:]
        A = np.append(A,value)
        i=i+1
    A = np.reshape(A,[_rows_num,_cols_num])
    print "A:", A.shape
    return A

'''调用函数，获取x,y的数据'''
batch_xs = datas_array(rows_num,cols_num,table1)
batch_ys = datas_array(1,cols_num,table2)
print "batch_xs:", batch_xs.shape
print "batch_ys:", batch_ys.shape



'''初始化神经网络参数'''
# RNN学习时使用的参数
learning_rate = 0.1        #学习率
training_iters = 10         #训练次数
batch_size =1               #每轮训练数据大小

# 神经网络的参数
n_inputs = cols_num        # 输入层的n,91步
n_steps =  rows_num      # 513长度,每一步
n_hidden = 20      # 隐藏层神经元个数
n_classes = 91       # 输出的数量，因为是分类问题，0~9个数字，这里一共有10个

# batch_xs = np.reshape(batch_xs, [batch_size,rows_num, cols_num])
# batch_ys = np.reshape(batch_ys, [1, cols_num])
# print "xs:",batch_xs.shape
# print "ys:",batch_ys.shape

x = tf.placeholder("float", [None, n_steps, n_inputs])
y = tf.placeholder("float", [None, n_classes])
# y = tf.placeholder("float", [None, n_steps, n_inputs])

# tensorflow里的LSTM需要两倍于n_hidden的长度的状态，一个state和一个cell
# istate = tf.placeholder("float", [None, 2 * n_hidden])


'''随机初始化每一层的权值和偏置'''
weights = {                  #字典
    'hidden': tf.Variable(tf.random_normal([n_inputs, n_hidden])),  # 从正态分布中输出随机值
    'out':    tf.Variable(tf.random_normal([n_hidden, n_classes]))
          }             #字典：键对值
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out':    tf.Variable(tf.random_normal([n_classes]))
         }              #biases是一维数组

'''
构建RNN
'''
def RNN(_X, _weights, _biases):
    ''''输入-batch_xs: batch*steps*inputs'''''
    # x1 = _X         #放在这里不行，报错 Placeholder:0 is both fed and fetched.
    _X = tf.reshape(_X, [-1, n_inputs])  # (n_steps* batch_size)* n_input

    # 输入层到隐含层，第一次是直接运算：n_steps * batch_size, n_hidden)
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']   # 3584*128
    x1 = _X
    # 自己添加的代码 #128*28*128,作为函数tf.nn.dynamic_rnn()中_X的输入：batch*steps*hidden
    _X = tf.reshape(_X, [-1, n_steps, n_hidden])

    # cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
    # 之后使用LSTM
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # 初始化为零值， lstm 单元由两个部分组成： (c_state, h_state)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    '''开始RNN'''
    # dynamic_rnn 接收张量(batch, steps, inputs)或者(steps, batch, inputs)作为 X_in，不能是list
    # dynamic_rnn可以允许不同batch的sequence length不同，但rnn不能。
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, _X,sequence_length=None,
                                            initial_state =init_state, time_major = False)
    # final_state[0]是cell state; final_state[1]是输出h_state
    # x1 = "outputs:", final_state
    #单步调试发现，outputs:128*28*128, final_state:128*256

    outputs = tf.transpose(outputs,[1,0,2])   # steps*batch*hidden:28*128*128
    x1 = outputs[n_steps-1,:,:]
    # x1 = final_state
    # 输出层：outputs[-1]: batch*hidden , results: batch*classes: 128*10

    results = tf.matmul(outputs[n_steps-1,:,:], _weights['out']) + _biases['out']
    # results = tf.matmul(final_state[1], _weights['out']) + _biases['out']

    return results , x1

'''
#预测值，调用RNN()
'''
# pred ,x_new= RNN(x,istate, weights, biases)
pred ,x_new= RNN(x,weights, biases)
#定义损失函数和优化方法，其中损失函数为softmax交叉熵，优化方法为Adam
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  # Softmax loss
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

''''开始运行'''
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # 持续迭代
    # while step * batch_size < training_iters:
    while step  < training_iters:
        start_time = time.time()   #计时

        # if step * batch_size==6:
        #     learning_rate = 0.001
        # print "learning_rate:", learning_rate

        batch_xs = datas_array(rows_num, cols_num, table1)
        batch_ys = datas_array(1, cols_num, table2)
        # batch_ys = datas_array(1, cols_num, table2)
        # batch_xs = batch_xs*(10**5)
        # batch_ys = batch_ys*(10**5)

        batch_xs = np.reshape(batch_xs, [batch_size,rows_num, cols_num])
        batch_ys = np.reshape(batch_ys, [1, cols_num])

        # print "x:", batch_xs.shape
        # print "y:", batch_ys.shape

        #给x、y、istate赋值，并运行。 --#字典：采用键对值
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,})

        cost_1=sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,})
        print "cost:", cost_1
        print "y:", batch_ys

        predict = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys, })
        print "prediction:",predict

        x_now = sess.run(x_new, feed_dict={x: batch_xs, y: batch_ys, })
        # print " x_now.shape :", np.array(x_now).shape
        print " x_now.shape :", x_now

        # acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, })
        # Acc = acc

        print "step * batch_size:", (step * batch_size)
        # print "Acc:",Acc

        print "训练耗时：", (time.time() - start_time)

        step = step+1

    print ("Optimization Finished!")

print "训练全部耗时：", (time.time() - start_time_0)
print "平均每批batch耗时：", (time.time() - start_time_0)/(step-1)



