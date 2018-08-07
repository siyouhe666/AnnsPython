import loaddata
import numpy as np
import tensorflow as tf

"""
此代码仅供学习交流 siyouhe666@gmail.com
"""
# 初始化TensorFlow变量 @n_x 特征维度 @n_y 标签维度
def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape = [n_x, None])
    Y = tf.placeholder(tf.float32, shape = [n_y, None])
    return X, Y


# 初始化参数 @neural_structre 神经网络结构
def initialize_parameters(neural_structre):
    parameters = {}
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
    parameters['para_num'] = len(neural_structre)
    for i in range(0, len(neural_structre)):
        w_name = "w"+str(i+1)
        b_name = "b"+str(i+1)
        w_structre = neural_structre[i]
        w = tf.get_variable(w_name, w_structre, initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b_structre = [w_structre[0], 1]
        b = tf.get_variable(b_name, b_structre, initializer=tf.zeros_initializer())
        parameters[w_name] = w
        parameters[b_name] = b
    return parameters


#定义前向传播图
def forward_propagation(X, parameters):
    para_num = parameters['para_num']
    a = X
    for i in range(1, para_num):
        w_name = 'w'+str(i)
        b_name = 'b'+str(i)
        w = parameters[w_name]
        b = parameters[b_name]
        a = forward_tool(w, b, a)
    w = parameters['w'+str(para_num)]
    b = parameters['b'+str(para_num)]
    z = tf.add(tf.matmul(w, a), b)
    return z


#每一层网络的计算方式，采用Relu激活函数
def forward_tool(w, b, X):
    z = tf.add(tf.matmul(w, X), b)
    a = tf.nn.relu(z)
    return a


#交叉熵损失函数度量损失值
def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    all_para = tf.trainable_variables()
    regular = 0.01*tf.reduce_sum([tf.nn.l2_loss(a) for a in all_para])
    cost = cost + regular
    # cost = tf.reduce_mean(tf.nn.l2_loss(logits, labels))
    return cost


# 训练神经网络 @learning_rate 学习率 @num_epoch 迭代次数 @batch_size batch大小
def model(learning_rate, num_epoch, batch_size):
    batches = loaddata.load_batches("data/space.txt", "data/test.txt", batch_size)
    tf.set_random_seed(1)
    x_temp = batches[0][0]
    y_temp = batches[0][1]
    (x_dim, m) = x_temp.shape
    y_dim = y_temp.shape[0]
    X, Y = create_placeholders(x_dim, y_dim)
    # ↓设计神经网络结构↓ 此处只需填写不同结构的数字即可，但必须保证每一层网络的输出输出shape相对应
    # 如下结构所示为5层神经网络每层 神经元个数与维度分别为 （10,x_dim） (10,10) (15,10) (10,15) (2,10)
    structure = []
    structure.append([10, x_dim])
    structure.append([10, 10])
    structure.append([15, 10])
    structure.append([10, 15])
    structure.append([2, 10])
    # ↑设计神经网络结构↑此处只需填写不同结构的数字即可，但必须保证每一层网络的输出输出shape相对应
    parameters = initialize_parameters(structure)# 初始化参数
    final_z = forward_propagation(X, parameters) # 定义前向传播计算图
    y = tf.nn.softmax(final_z) #定义输出，在下方使用sess.run(y, {X:x_temp}) 即可输出结果
    cost = compute_cost(final_z, Y) # 计算损失
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam优化算法
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epoch):
            epoch_cost = 0.
            for batch in batches:
                x_temp = batch[0]
                y_temp = batch[1]
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: x_temp, Y: y_temp})
                epoch_cost += minibatch_cost / len(batches)
            print("cost:" + str(epoch_cost) + " epoch:" + str(epoch))
        testdata = loaddata.load_batches("data/space.txt", "data/tt.txt", 17000)
        x_temp = testdata[0][0]
        y_temp = testdata[0][1]
        test_cost = minibatch_cost = sess.run([cost], feed_dict={X: x_temp, Y: y_temp})
        print("测试集cost:"+str(test_cost))



model(0.001, 25, 1024)# 调用算法