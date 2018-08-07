import numpy as np
import tensorflow as tf

"""
此代码仅供学习交流 siyouhe666@gmail.com
"""


# 读取文件 @path 文件路径
def readfile(path):
    f = open(path, encoding='utf-8')
    return f


# 读取特征空间 @f 文件流
def readspace(f):
    space = {}
    line = f.readline()
    while line:
        if len(line) == 0:
            continue
        kv = line.split(" ")
        key = kv[0]
        val = kv[1]
        space[key] = val
        line = f.readline()
    f.close()
    return space


# 读取数据 @f 文件流 @dimension 特征维度
def readdata(f, dimension):
    line = f.readline()
    datalist = []
    labellist = []
    while line:
        vec = np.zeros(dimension, int)
        fl = line.split("\t")
        features = fl[0].split(" ")
        label = fl[1]
        line = f.readline()
        for fea in features:
            if fea == '':
                continue
            index = int(fea)
            vec[index] = 1
        datalist.append(vec)
        labellist.append(label)
    f.close()
    return datalist, labellist


# 打乱数据顺序 @datalist 数据列表 @labellist 标签列表
def resort(datalist, labellist):
    np.random.seed(1) # 随机数种子
    if len(datalist) != len(labellist):
        print("resort error")
    permutation = list(np.random.permutation(len(datalist)))
    newdata = []
    newlabel = []
    for i in permutation:
        newdata.append(datalist[i])
        newlabel.append(int(labellist[i]))
    return newdata, newlabel


# mini_batch生成器 @data 数据 @label 标签 @batch_size batch大小 @label_num 标签类别数
def mini_batches(data, label, batch_size, label_num):
    C = tf.constant(value=label_num, name="C")
    one_hot = tf.one_hot(label, C)
    sess = tf.Session()
    label = sess.run(one_hot)
    sess.close()
    mini_batches = []
    m = len(data)
    num_batch = int(m/batch_size)
    for k in range(0, num_batch):
        if k == num_batch-1:
            d = np.array(data[k*batch_size : ]).T
            l = label[k * batch_size: ]
            l = l.T
        else:
            d = np.array(data[k * batch_size: k * batch_size + batch_size]).T
            l = label[k * batch_size: k * batch_size + batch_size]
            l = l.T
        mini_batch = (d, l)
        mini_batches.append(mini_batch)
    return mini_batches


# 获取切分好的batch数据 @space_path 特征空间文件路径 @train_path 训练数据
def load_batches(space_path, train_path, batch_size):
    f = readfile(space_path)
    space = readspace(f)
    dimension = len(space)
    print("维度：" + str(dimension))
    f = readfile(train_path)
    datalist, labellist = readdata(f, dimension)
    datalist, labellist = resort(datalist, labellist)
    minis = mini_batches(datalist, labellist, batch_size, 2)
    # print("batches总数：" + str(len(minis)))
    # print("特征向量："+str(minis[0][0]))
    # print("类标向量："+str(minis[0][1]))
    return minis

