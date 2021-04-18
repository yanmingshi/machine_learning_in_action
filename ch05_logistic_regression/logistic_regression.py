# _*_ coding: utf-8 _*_
# @time : 2021/4/8 22:19
# @author : yanms
# @version：V 0.1
# @file : logistic_regression.py
# @desc :

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data_set():
    """
    加载数据集
    :return:
    """
    data_mat = []
    label_mat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def pd_load_data_set():
    """
    pandas加载数据集
    :return:
    """
    df = pd.read_table("./testSet.txt", header=None)
    df.insert(0, 3, 1.0)
    return df.iloc[:, :-1].values.tolist(), df.iloc[:, -1].values.tolist()


def sigmoid(x):
    """
    sigmoid函数
    :param x:
    :return:
    """
    return 1.0 / (1 + np.exp(-x))


def grad_ascent(data_mat_in, class_labels):
    """
    梯度上升算法
    :param data_mat_in:
    :param class_labels:
    :return:
    """
    data_matrix = np.mat(data_mat_in)
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(data_matrix)

    # 步长
    alpha = 0.001
    # 循环次数
    max_cycles = 500
    weights = np.ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = (label_mat - h)
        weights = weights + alpha * data_matrix.transpose() * error
    return weights


def plot_best_fit(weights):
    """
    画出曲线
    :param weights:
    :return:
    """
    data_mat, label_mat = load_data_set()
    data_arr = np.array(data_mat)
    n = np.shape(data_arr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(label_mat[i]):
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def stoc_grad_ascent0(data_matrix, class_labels, alpha=0.01):
    """
    随机梯度上升算法(简化版)
    :param data_matrix:
    :param class_labels:
    :param alpha: 学习率
    :return:
    """
    m, n = np.shape(data_matrix)
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(data_matrix[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * data_matrix[i]
    return weights


def stoc_grad_ascent(data_matrix, class_labels, epoch=150):
    """
    随机梯度上升算法(改进版)
    :param data_matrix:
    :param class_labels:
    :param epoch: 训练次数
    :return:
    """
    m, n = np.shape(data_matrix)
    weights = np.ones(n)
    for j in range(epoch):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            rand_index = int(np.random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_matrix[rand_index]
            del (data_index[rand_index])
    return weights


def classify_vector(x, weights):
    """
    采用sigmoid函数进行分类
    :param x:
    :param weights:
    :return:
    """
    prob = sigmoid(sum(x * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    """
    对病马死亡率预测
    :return:
    """
    fr_train = open('horseColicTraining.txt')
    fr_test = open('horseColicTest.txt')
    training_set = []
    training_labels = []

    # 训练logistic 分类器
    # 用pandas读取数据
    train_data = pd.read_table("./horseColicTraining.txt", header=None)
    training_set, training_labels = train_data.iloc[:, :-1].values.tolist(), train_data.iloc[:, -1].values.tolist()
    train_weights = stoc_grad_ascent(np.array(training_set), training_labels, 500)

    # 用循环方式读取数据
    # for line in fr_train.readlines():
    #     curr_line = line.strip().split('\t')
    #     line_arr = []
    #     for i in range(21):
    #         line_arr.append(float(curr_line[i]))
    #     training_set.append(line_arr)
    #     training_labels.append(float(curr_line[21]))
    # train_weights = stoc_grad_ascent(np.array(training_set), training_labels, 500)

    # 测试
    error_count = 0
    num_test_vec = 0.0
    # 用pandas读取数据
    test_data = pd.read_table("./horseColicTest.txt", header=None)
    test_set, test_label = test_data.iloc[:, :-1].values.tolist(), test_data.iloc[:, -1].values.tolist()
    for i in range(len(test_label)):
        if int(classify_vector(np.array(test_set[i]), train_weights)) != int(test_label[i]):
            error_count += 1
    error_rate = (float(error_count) / len(test_label))

    # 用循环方式读取数据
    # for line in fr_test.readlines():
    #     num_test_vec += 1.0
    #     curr_line = line.strip().split('\t')
    #     line_arr = []
    #     for i in range(21):
    #         line_arr.append(float(curr_line[i]))
    #     if int(classify_vector(np.array(line_arr), train_weights)) != int(curr_line[21]):
    #         error_count += 1
    # error_rate = (float(error_count) / num_test_vec)
    print('the error rate of this test is : %f' % error_rate)
    return error_rate


def multi_test():
    """
    调用colic_test()函数训练10次求平均值
    :return:
    """
    num_tests = 10
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    print('after % d iterations the average error rate is: %f' % (num_tests, error_sum / float(num_tests)))


if __name__ == '__main__':
    # data_mat, label_mat = load_data_set()
    # print(data_mat)
    # print(label_mat)

    data_mat, label_mat = pd_load_data_set()
    # print(data_mat)
    # print(label_mat)

    # weights = grad_ascent(data_mat, label_mat)
    # print(weights)
    # print(weights.getA())
    # plot_best_fit(weights.getA())

    # weights = stoc_grad_ascent(np.array(data_mat), label_mat, 500)
    # plot_best_fit(weights)

    multi_test()
