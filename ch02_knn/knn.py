# _*_ coding: utf-8 _*_
# @time : 2021/3/20 22:28
# @author : yanms
# @version：V 0.1
# @file : knn.py
# @desc :

import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir


def create_date_set():
    """
    模拟创建数据集
    :return: group 样本
             label 标签
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    label = ['A', 'A', 'B', 'B']
    return group, label


def classify0(in_x, data_set, labels, k):
    """
    基本knn分类算法
    :param in_x:
    :param data_set:
    :param labels:
    :param k:
    :return:


    """
    # 计算距离
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    # axis 代表数组的层数 0为最外层，1为第二层。。。
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()

    # 选择距离最小的k个点
    class_count = {}
    for i in range(k):
        vote_i_label = labels[sorted_dist_indicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    """
    a = [1,2,3] 
    >>> b=operator.itemgetter(1)      //定义函数b，获取对象的第1个域的值
    >>> b(a) 
    2 
    """
    # 排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file2matrix(file_name):
    """
    将文本记录转换为numpy的接续程序
    :param file_name:
    :return:
    """

    dating_label_code = {'didntLike': 1, 'smallDoses': 2, 'largeDoses': 3}

    fr = open(file_name)
    array_o_lines = fr.readlines()
    number_o_f_lines = len(array_o_lines)
    return_mat = np.zeros((number_o_f_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_o_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(int(dating_label_code.get(list_from_line[-1])))
        index += 1
    return return_mat, class_label_vector


def auto_norm(data_set):
    """
    归一化
    :param data_set:
    :return:
    """
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    m = data_set.shape[0]
    norm_data_set = data_set - np.tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


def dating_class_test():
    """
    分类器验证
    :return:
    """
    ho_ratio = 0.10
    dating_data_mat, dating_labels = file2matrix('datingTestSet.txt')
    norm_mat, ranges, min_value = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], 3)
        print('the classifier came back with:%d, the real answer is: %d' % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print('the total error rate is:%f' % (error_count / float(num_test_vecs)))


def classify_person():
    """
    预测
    :return:
    """
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input('percentage of time spent playing video games?'))
    ff_miles = float(input('frequent flier miles earned per year?'))
    ice_cream = float(input('liters of ice cream consumed per year?'))
    dating_data_mat, dating_labels = file2matrix('datingTestSet.txt')
    norm_mat, ranges, min_value = auto_norm(dating_data_mat)
    in_arr = np.array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - min_value) / ranges, norm_mat, dating_labels, 3)
    print('you will probably like this person: ', result_list[classifier_result - 1])


def img2vector(file_name):
    """
    图像转换成向量
    :param file_name:
    :return:
    """
    return_vect = np.zeros((1, 1024))
    fr = open(file_name)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect


def handwriting_class_test():
    """
    测试手写数字识别
    :return:
    """
    hw_labels = []
    training_file_list = listdir('digits/trainingDigits')
    m = len(training_file_list)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_number_str = int(file_str.split('_')[0])
        hw_labels.append(class_number_str)
        training_mat[i, :] = img2vector('digits/trainingDigits/%s' % file_name_str)
    test_file_list = listdir('digits/testDigits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_number_str = int(file_str.split('_')[0])
        vector_under_test = img2vector('digits/testDigits/%s' % file_name_str)
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)
        print('the classifier came back with: %d, the real answer is: %d' % (classifier_result, class_number_str))
        if classifier_result != class_number_str:
            error_count += 1.0
    print('\n the total number of errors is: %d' % error_count)
    print('\n the total error rate is: %f' % (error_count/float(m_test)))


def draw(data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 1], data[:, 0], 15.0 * np.array(labels), 2.0 * np.array(labels))
    plt.show()


if __name__ == '__main__':
    # dating_data_mat, dating_label = file2matrix('datingTestSet.txt')
    # norm_mat, ranges, min_value = auto_norm(dating_data_mat)
    # print(norm_mat)
    # print(ranges)
    # print(min_value)
    # classify_person()
    # img2vector("digits/testDigits/0_0.txt")
    handwriting_class_test()