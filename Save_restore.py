# coding=utf-8
#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

# This shows how to save/restore your model (trained variables).
# To see how it works, please stop this program during training and resart.
# This network is the same as 3_net.py

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden): # this network is the same as the previous one except with an extra hidden layer + dropout
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w_h = init_weights([784, 625])
w_h2 = init_weights([625, 625])
w_o = init_weights([625, 10])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)


ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

global_step = tf.Variable(0, name='global_step', trainable=False)

# Call this after declaring all tf.Variables.
saver = tf.train.Saver()

# This variable won't be stored, since it is declared after tf.train.Saver()
non_storable_variable = tf.Variable(777)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    #加载模型
    # ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    # if ckpt and ckpt.model_checkpoint_path:
    #     print(ckpt.model_checkpoint_path)
    #     saver.restore(sess, ckpt.model_checkpoint_path)   # 加载所有的参数

    start = global_step.eval() # 得到初始值  get last global_step
    print("Start from:", type(start))

    for i in range(start, 2):
        #以128作为batch_size
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})

        # 更新计数器
        global_step.assign(i)  # 更新计数器  set and update(eval) global_step with index, i
        # 存储模型
        saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)  # 存储训练的模型
        #第几次训练及准确率
        print i, np.mean(np.argmax(teY, axis=1) == sess.run(predict_op,
                         feed_dict={X: teX,  p_keep_input: 1.0, p_keep_hidden: 1.0}))

'''''  
加载模型
'''''
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)

    print ckpt
    # model_checkpoint_path: "./ckpt_dir/model.ckpt-0"
    print ckpt.model_checkpoint_path
    # all_model_checkpoint_paths: "./ckpt_dir/model.ckpt-0"

    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        # ./ ckpt_dir / model.ckpt - 0
        # 加载所有的参数
        saver.restore(sess, ckpt.model_checkpoint_path)  # 加载所有的参数
        # 从这里开始就可以直接使用模型进行预测，或者接着继续训练了


# eval()函数常见作用有：
# 1、计算字符串中有效的表达式，并返回结果
# 2、将字符串转成相应的对象（如list、tuple、dict和string之间的转换）

# assign():
# 为变量赋一个新值  Assigns a new value to the variable.