# coding=utf-8
# http://makaidong.com/max-hu/190280_642815.html
# RNN情感分析:本文中使用一个基于lstm的RNN来预测电影评论的情感方向是“正面”还是“负面”

# 导入库文件
import numpy as np
import tensorflow as tf
import time

# 读取数据
with open('./A_reviews.txt', 'r') as f:
    reviews = f.read()
with open('./A_labels.txt', 'r') as f:
    labels = f.read()

# 数据预处理
from string import punctuation

all_text = ''.join([c for c in reviews if c not in punctuation])  # 去掉标点符号
reviews = all_text.split('\n')

all_text = ' '.join(reviews)
words = all_text.split()

# 编码review和label
# 创建词到数字转换的词典
from collections import Counter

counter = Counter(words)
vocab_sorted = sorted(counter, key=counter.get, reverse=True)
vocab_to_int = {word: num for num, word in enumerate(vocab_sorted, 1)}

# 将评论转化为数字
reviews_ints = []
for review in reviews:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])

# 将'positive' 和'negative'的label分别转换为1和0
labels = labels.split('\n')
labels = np.array([1 if each == 'positive' else 0 for each in labels])

# 删除长度为0的review和对应的label
non_zero_index = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]
reviews_ints = [reviews_ints[ii] for ii in non_zero_index]
labels = np.array([labels[ii] for ii in non_zero_index])
# 至此，我们已将reviews和labels全部转换为更容易处理的整数。

# 现在，我们要创建一个传递给网络的features数组，为方便处理可以将特征向量的长度定义为200。
# 对于短于200个字的评论，左边补全为0。 也就是说，如果review是['best'，'movie'，'ever']
# （对应整数位[117，18，128]），相对应的行将是[0，0，0，...，0，117 ，18, 128]；
# 对于超过200次的评论，使用前200个单词作为特征向量。
seq_len = 200
features = np.array([review[:seq_len] if len(review) > seq_len else [0] *
                    (seq_len - len(review)) + review for review in reviews_ints])

# 创建training validation test数据集
split_frac = 0.8

split_idx = int(len(features) * split_frac)
train_x, val_x = features[:split_idx], features[split_idx:]
train_y, val_y = labels[:split_idx], labels[split_idx:]

val_idx = int(len(val_x) * 0.5)
val_x, test_x = val_x[:val_idx], val_x[val_idx:]
val_y, test_y = val_y[:val_idx], val_y[val_idx:]

# 创建graph
# 首先，定义超参数
lstm_size = 256
lstm_layers = 1
batch_size = 1000
learning_rate = 0.01
# lstm_size：LSTM元胞中隐藏层的单元数量，LSTM元胞中实际有四种不同的网络层, 这是每一层中的单元数
# lstm_layers：LSTM层的数量
# batch_size： 单次训练中传入网络的review数量
# learning_rate：学习率

# 定义变量及嵌入层
n_words = len(vocab_to_int)

graph = tf.Graph()
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, (batch_size, seq_len), name='inputs')
    labels_ = tf.placeholder(tf.int32, (batch_size, 1), name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

embed_size = 300

with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)

# LSTM层
# 在TensorFlow中，使用tf.contrib.rnn.BasicLSTMCell可以很方便的创建LSTM元胞，基本用法可以参考官方文档https: // www.tensorflow.org / api_docs / python / tf / contrib / rnn / BasicLSTMCell
# 使用tf.contrib.rnn.BasicLSTMCell(num_units)
# 就可以创建一个隐藏层单元数量为num_units的元胞
# 接下来，可以使用tf.contrib.rnn.DropoutWrapper来给lstm元胞添加dropout
# 例如 ：drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
# 即给元胞cell添加了dropout
# 通常，多个LSTM层可以是我们的模型获得更好的表现，如果我们想使用多个LSTM层的话，该怎么做呢？TensorFlow也能很方便的实现这个
# 例如：cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
# 就创建了lstm_layers个lstm层，每层的结构和drop类似（drop是一个添加了dropout的基本的lstm层）。
with graph.as_default():
    # lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0)
    # drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    # cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    # initial_state = cell.zero_state(batch_size, tf.float32)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
# 前向传播
# 在数据的前向传播中，我们需要使用tf.nn.dynamic_rnn来运行LSTM层的代码。
# 基本使用方法：outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)，其中cell是上面定义的lstm层。
with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, embed, initial_state=init_state)

# 输出
# 由于我们只关心最终的输出，所以我们需要使用outputs[:, -1]
# 来获取最终输出，并由此计算cost。
with graph.as_default():
    predictions = tf.contrib.layers.fully_connected(outputs[:,:,1], 1, activation_fn=tf.sigmoid)
    # cost = tf.losses.mean_squared_error(labels_, predictions)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predictions, labels))

    # optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(cost)

# 获取精度
# 从validation中获取精度
with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 获取训练batch
# 这里使用生成器generator来获取训练用的batch

def get_batches(x, y, batch_size=100):
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size]
       #yield: return 返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后(下一行)开始

#计时#
start_time = time.time()
print ("构建图耗时", time.time() - start_time)
start_time = time.time()
print ("训练耗时：", time.time() - start_time)

# 训练
# Trainging
epochs = 3

with graph.as_default():
    saver = tf.train.Saver()       #保存变量？

with tf.Session(graph=graph) as sess:
    # 初始化
    init = tf.initialize_all_variables()
    sess.run(init)
    iteration = 1
    for e in range(epochs):
        state = sess.run(init_state)

        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 0.5,
                    init_state: state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

            if iteration % 5 == 0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))

            if iteration % 25 == 0:
                val_acc = []
                val_state = sess.run(lstm_cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keep_prob: 1,
                            init_state: val_state}
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            iteration += 1
    # saver.save(sess, "checkpoints/sentiment.ckpt")      #存储模型



