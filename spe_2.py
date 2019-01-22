# coding=utf-8
# http://blog.csdn.net/liuchonge/article/details/70809288
import tensorflow as tf
import numpy as np


# RNN学习时使用的参数
learning_rate = 0.001        #学习率
training_iters = 400         #训练次数
batch_size =128              #每轮训练数据大小

# 神经网络的参数
num_inputs = 28        # 输入层的n,28步
num_steps = 28        # 28长度,每一步
num_hidden = 128      # 隐藏层神经元个数
num_classes = 10
state_size = 28

# 这部分主要是生成实验数据，并将其按照RNN模型的输入格式进行切分和batch化。
# 1，生成实验数据：

def gen_data(size=100000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        #判断X[i-3]和X[i-8]是否为1，修改阈值
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        #生成随机数，以threshold为阈值给Yi赋值
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)

# 接下来将生成的数据按照模型参数设置进行切分，这里需要用得到的参数主要包括：batch_size和num_steps，
# 分别是批量数据大小和RNN每层rnn_cell循环的次数，也就是下图中Sn中n的大小。

def gen_batch(raw_data, batch_size, num_steps):
    #raw_data是使用gen_data()函数生成的数据，分别是X和Y
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # 首先将数据切分成batch_size份，0-batch_size，batch_size-2*batch_size。。。
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]

    #因为RNN模型一次只处理num_steps个数据，所以将每个batch_size在进行切分成epoch_size份，每份num_steps个数据。注意这里的epoch_size和模型训练过程中的epoch不同。
    epoch_size = batch_partition_length // num_steps

    #x是0-num_steps， batch_partition_length -batch_partition_length +num_steps。。。共batch_size个
    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)

#这里的n就是训练过程中用的epoch，即在样本规模上循环的次数
def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)


# 至于使用TensorFlow构建RNN模型，主要就是定义rnn_cell类型，然后将其复用即可。代码如下所示：

x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
#RNN的初始化状态，全设为零。注意state是与input保持一致，接下来会有concat操作，所以这里要有batch的维度。即每个样本都要有隐层状态
init_state = tf.zeros([batch_size, state_size])

#将输入转化为one-hot编码，两个类别。[batch_size, num_steps, num_classes]
x_one_hot = tf.one_hot(x, num_classes)
#将输入unstack，即在num_steps上解绑，方便给每个循环单元输入。这里可以看出RNN每个cell都处理一个batch的输入（即batch个二进制样本输入）
rnn_inputs = tf.unstack(x_one_hot, axis=1)

#定义rnn_cell的权重参数，
with tf.variable_scope('rnn_cell'):
    W = tf.get_variable('W', [num_classes + state_size, state_size])
    b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
#使之定义为reuse模式，循环使用，保持参数相同
def rnn_cell(rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [num_classes + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    #定义rnn_cell具体的操作，这里使用的是最简单的rnn，不是LSTM
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)

state = init_state
rnn_outputs = []
#循环num_steps次，即将一个序列输入RNN模型
for rnn_input in rnn_inputs:
    state = rnn_cell(rnn_input, state)
    rnn_outputs.append(state)
final_state = rnn_outputs[-1]

#定义softmax层
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
#注意，这里要将num_steps个输出全部分别进行计算其输出，然后使用softmax预测
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]

# Turn our y placeholder into a list of labels
y_as_list = tf.unstack(y, num=num_steps, axis=1)

#losses and train_step
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for \
          logit, label in zip(logits, y_as_list)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)


# 模型训练
# 定义好我们的模型之后，接下来就是将数据传入，然后进行训练

def train_network(num_epochs, num_steps, state_size=4, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        #得到数据，因为num_epochs==1，所以外循环只执行一次
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            training_loss = 0
            #保存每次执行后的最后状态，然后赋给下一次执行
            training_state = np.zeros((batch_size, state_size))
            if verbose:
                print("\nEPOCH", idx)
            #这是具体获得数据的部分，应该会执行1000000//200//5 = 1000次，即每次执行传入的数据是batch_size*num_steps个（1000），共1000000个，所以每个num_epochs需要执行1000次。
            for step, (X, Y) in enumerate(epoch):
                tr_losses, training_loss_, training_state, _ = \
                    sess.run([losses,
                              total_loss,
                              final_state,
                              train_step],
                                  feed_dict={x:X, y:Y, init_state:training_state})
                training_loss += training_loss_
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print("Average loss at step", step,
                              "for last 250 steps:", training_loss/100)
                    training_losses.append(training_loss/100)
                    training_loss = 0

    return training_losses
training_losses = train_network(1,num_steps)
plt.plot(training_losses)
plt.show()
