# coding=utf-8
# https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-06-save/
# 莫烦--saver.save保存和 saver.restore读取
#不能用


import tensorflow as tf
import numpy as np

''' 保存'''
# remember to define the same dtype and shape when restore
W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
b = tf.Variable([[1,2,3]],         dtype=tf.float32, name='biases')

init = tf.global_variables_initializer()

# 保存时, 首先要建立一个 tf.train.Saver() 用来保存, 提取变量. 再创建一个名为my_net的文件夹,
#  用这个 saver 来保存变量到这个目录 "my_net/save_net.ckpt".

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "./Saver/save_net.ckpt")
    print("Save to path: ", save_path)

"""    
Save to path:  ./Saver/save_net.ckpt
"""

'''提取'''
# 提取时, 先建立零时的W 和 b容器. 找到文件目录, 并用saver.restore()我们放在这个目录的变量.
# 先建立 W, b 的容器
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

# 这里不需要初始化步骤 init= tf.initialize_all_variables()
tf.reset_default_graph()
# saver = tf.train.Saver()

with tf.Session() as sess:
    # 提取变量
    # tf.reset_default_graph()
    # sess.run(init)
    saver.restore(sess, "./Saver/save_net.ckpt")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))

"""
weights: [[ 1.  2.  3.]
          [ 3.  4.  5.]]
biases: [[ 1.  2.  3.]]
"""