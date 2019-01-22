#coding=UTF-8
# 运行环境： python2.7 , tensorflow-v0.8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import xlwt
import time
from scipy import signal


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
# batch_xs = datas_array(rows_num,cols_num,table1)
# batch_ys = datas_array(1,cols_num,table2)
# print "batch_xs:", batch_xs.shape
# print "batch_ys:", batch_ys.shape


'''初始化神经网络参数'''
# RNN学习时使用的参数
learning_rate = 0.001       #学习率
training_iters = 100        #训练次数
batch_size =cols_num
#每轮训练数据大小

# 神经网络的参数
n_inputs = rows_num        # 输入层的n,91步
n_steps =   1              # 513长度,每一步
n_hidden = 30             # 隐藏层神经元个数
n_classes = rows_num       # 输出的数量，因为是分类问题，0~9个数字，这里一共有10个

# batch_xs = np.reshape(batch_xs, [batch_size,rows_num, cols_num])
# batch_ys = np.reshape(batch_ys, [1, cols_num])
# print "xs:",batch_xs.shape
# print "ys:",batch_ys.shape

x = tf.placeholder("float", [None, cols_num*rows_num])
y = tf.placeholder("float", [None, cols_num*rows_num])
# x = tf.placeholder("float", [None, n_steps, n_inputs])
# y = tf.placeholder("float", [None, n_classes])
# y = tf.placeholder("float", [None, n_steps, n_inputs])
# sequence_length = tf.placeholder(tf.int32, [None])

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
    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    # 初始化为零值， lstm 单元由两个部分组成： (c_state, h_state)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    '''开始RNN'''
    # dynamic_rnn 接收张量(batch, steps, inputs)或者(steps, batch, inputs)作为 X_in，不能是list
    # dynamic_rnn可以允许不同batch的sequence length不同，但rnn不能。
    # outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, _X,sequence_length=[1]*cols_num,
    #                                         initial_state =init_state, time_major = False)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, _X,sequence_length=None,
                                            initial_state =init_state, time_major = False)
    # final_state[0]是cell state; final_state[1]是输出h_state
    # x1 = "outputs:", final_state
    #单步调试发现，outputs:128*28*128, final_state:128*256
    # X2 = sequence_length
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
pred ,x_new= RNN(x,weights, biases)
#定义损失函数和优化方法，其中损失函数为softmax交叉熵，优化方法为Adam
pred = tf.reshape(pred, [1, rows_num*cols_num])   #将数据变成一行
y = tf.reshape(y, [1, rows_num*cols_num])         #将数据变成一行

# cost = (tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer

# error = tf.square(tf.sub(y, pred))  #subtract
# losses = tf.rnn.sequence_loss_by_example(
#             [tf.reshape(pred, [-1], name='reshape_pred')],
#             [tf.reshape(y, [-1], name='reshape_target')],
#             [tf.ones([batch_size * n_steps], dtype=tf.float32)],
#             average_across_timesteps=True,
#             softmax_loss_function=error,
#             name='losses')


# 初始化
init = tf.initialize_all_variables()

print "'''开始计时'''"
start_time_0 = time.time()   #计时

''''开始运行'''
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # 持续迭代
    # while step * batch_size < training_iters:
    while step  <= training_iters:
        start_time = time.time()   #计时

        # if step * batch_size==6:
        #     learning_rate = 0.001
        # print "learning_rate:", learning_rate

        # 随机抽出这一次迭代训练时用的数据
        #batch_xs: batch*784, batch_ys: batch*classes
        # batch_xs, batch_ys = mnist.train.next_batch(batch_size)   #通过next_batch()就可以一个一个batch的拿数据

        '''调用函数，得到xs,ys'''
        batch_xs = datas_array(rows_num, cols_num, table1)
        batch_ys = datas_array(rows_num, cols_num, table2)
        # batch_ys = datas_array(1, cols_num, table2)
        # batch_xs = batch_xs*(10**5)
        # batch_ys = batch_ys*(10**5)

        # batch_xs = np.reshape(batch_xs, [batch_size, 1, cols_num])
        # batch_ys = np.reshape(batch_ys, [batch_size, 1, cols_num])

        batch_xs = np.reshape(batch_xs, [1, rows_num*cols_num])
        batch_ys = np.reshape(batch_ys, [1, rows_num*cols_num])

        # print "x:", batch_xs.shape
        # print "y:", batch_ys

        #给x、y、istate赋值，并运行。 --#字典：采用键对值
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,})

        cost_1=sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,})
        print "cost:", cost_1
        print "y:", batch_ys

        predict = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys, })
        print "prediction:",predict

        x_now = sess.run(x_new, feed_dict={x: batch_xs, y: batch_ys, })
        print " x_now.shape :", x_now

        # acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, })
        # Acc = acc

        print "step :", (step)
        # print "Acc:",Acc

        print "训练耗时：", (time.time() - start_time)

        step = step+1

    print ("Optimization Finished!")

print "训练全部耗时：", (time.time() - start_time_0)
print "平均每批batch耗时：", (time.time() - start_time_0)/(step-1)



'''载入测试集进行测试'''
# # print "'''载入测试集进行测试'''"
# # '''读取Excel文件'''
# data1 = xlrd.open_workbook('./语音图片1/TIMIT-sa2-syn.xls')
# data2 = xlrd.open_workbook('./语音图片1/TIMIT-sa2.xls')
# print "data:", data1
# table1 = data1.sheets()[0]    #选择工作表sheet1
# table2 = data2.sheets()[0]
# print "table:", np.array(table1)
# '''计算行列数'''
# rows_num = table1.nrows
# cols_num = table1.ncols
# print "行数列数：", rows_num, cols_num
# '''调用函数，得到xs,ys'''
# batch_xs = datas_array(rows_num, cols_num, table1)
# batch_ys = datas_array(rows_num, cols_num, table2)
#
# batch_xs = np.reshape(batch_xs, [1, rows_num * cols_num])
# batch_ys = np.reshape(batch_ys, [1, rows_num * cols_num])
# print "x:", batch_xs.shape
# print "y:", batch_ys.shape
#
# with tf.Session() as sess:
#     sess.run(init)
#     print "'''测试开始计时'''"
#     start_time = time.time()
#     predict = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys, })
#     print "prediction:", predict
#     print "测试耗时：", (time.time() - start_time)



'''保存数据'''
# predict = np.reshape(predict, [rows_num , cols_num])
# print "predict：", predict
# #使用workbook方法，创建一个新的工作簿book
# book=xlwt.Workbook(encoding='utf-8')
# #添加一个sheet
# sheet=book.add_sheet('Sheet_1', cell_overwrite_ok=True)
# for i,row in enumerate(predict):
#     for j,col in enumerate(row):
#         sheet.write(i,j,col)
# print "'''开始保存数据为.xls'''"
# book.save('./语音图片1/data.xls')
# print "'''数据保存完成'''"
