# coding=utf-8
# http://blog.csdn.net/u010223750/article/details/71079036
# tf.dynamic_rnn介绍,来自网上

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 创建输入数据
X = np.random.randn(2, 10, 8)  # 从标准正态分布中返回多个样本值--维度为：2*10*8,2行10列8个一维
print X.shape

# 第二个example长度为6
X[1, 6:] = 0        #等价于 X[1,6:10,:] = 0  第二行，6-10列全为0
# print X
X_lengths = [10, 6]

# cell = tf.contrib.rnn.BasicLSTMCell(num_units=64, state_is_tuple=True)
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=64, forget_bias=1.0)

outputs, last_states = tf.nn.dynamic_rnn( cell=cell,
                                          dtype=tf.float64,
                                          sequence_length=X_lengths,
                                          inputs=X)
print "outputs:", outputs.shape            #array ,(2, 10, 64)
print "last_states:", np.array(last_states).shape    #list  ,(2,)
# outputs: Tensor("rnn/transpose:0", shape=(2, 10, 64), dtype=float64)
# last_states: LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_2:0' shape=(2, 64) dtype=float64>

# 在终端中，进入python--help()--输入tensorflow.contrib.learn.run_n--查找完成
result = tf.contrib.learn.run_n( {"outputs": outputs, "last_states": last_states},
                                 n=1, feed_dict=None)

# print "result:", result   #result是list,含有字典
# result: [{'outputs': array([[[  1.12458271e-01,   1.09040150e-05, ...,
#           -6.19570859e-03,   1.02059934e-01,  -1.22463077e-01],

# assert断言语句，用来判断对错
assert result[0]["outputs"].shape == (2, 10, 64)
print result[0]["last_states"]
# 第二个example中的outputs超过6步(7-10步)的值应该为0
assert (result[0]["outputs"][1,7,:] == np.zeros(cell.output_size)).all()


# assert的异常参数，其实就是在断言表达式后添加字符串信息，用来解释断言并更好的知道是哪里出了问题。格式如下：
# assert expression [, arguments]
# assert 表达式 [, 参数]
# assert len(lists) >=5,'列表元素个数小于5'  #
# assert 2==1,'2不等于1'   #返回错误


# numpy中有一些常用的用来产生随机数的函数，randn()和rand()就属于这其中。
# numpy.random.randn(d0, d1, …, dn)是从标准正态分布中返回一个或多个样本值。
# numpy.random.rand(d0, d1, …, dn)的随机样本位于[0, 1)中。


# 简要介绍tensorflow的RNN:
# 其实在前面多篇都已经提到了TensorFlow的RNN，也在我之前的文章TensorFlow实现文本分类文章中用到了
# BasicLSTM的方法，通常的，使用RNN的时候，我们需要指定num_step，也就是TensorFlow的roll step步数，
# 但是对于变长的文本来说，指定num_step就不可避免的需要进行padding操作，在之前的文章TensorFlow
# 高阶读写教程也使用了dynamic_padding方法实现自动padding，但是这还不够，因为在跑一遍RNN/LSTM之后，
# 还是需要对padding部分的内容进行删除，我称之为“反padding”，无可避免的，我们就需要指定mask矩阵了，
# 这就有点不优雅，但是TensorFlow提供了一个很优雅的解决方法，让mask去见马克思去了，那就是dynamic_rnn
# tf.dynamic_rnn
# tensorflow 的dynamic_rnn方法，我们用一个小例子来说明其用法，假设你的RNN的输入input是[2,20,128]，
# 其中2是batch_size,20是文本最大长度，128是embedding_size，可以看出，有两个example，我们假设
# 第二个文本长度只有13，剩下的7个是使用0-padding方法填充的。dynamic返回的是两个参数：
# outputs,last_states，其中outputs是[2,20,128]，也就是每一个迭代隐状态的输出,
# last_states是由(c,h)组成的tuple，均为[batch,128]。
# 到这里并没有什么不同，但是dynamic有个参数：sequence_length，这个参数用来指定每个example的长度，
# 比如上面的例子中，我们令 sequence_length为[20,13]，表示第一个example有效长度为20，
# 第二个example有效长度为13，当我们传入这个参数的时候，对于第二个example，TensorFlow对于13以后的padding
# 就不计算了，其last_states将重复第13步的last_states直至第20步，而outputs中超过13步的结果将会被置零。

