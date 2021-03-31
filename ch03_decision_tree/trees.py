# _*_ coding: utf-8 _*_
# @time : 2021/3/24 22:37
# @author : yanms
# @version：V 0.1
# @file : trees.py
# @desc :

from math import log
import operator
import pickle


def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def split_data_set(data_set, axis, value):
    """

    :param data_set: 数据集
    :param axis: 要划分的列（划分后的结果中去掉该列）
    :param value: 划分依据（axis值等于value的进行划分，否则略过）
    :return:
    """
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def calc_shannon_ent(data_set):
    """
    计算给定数据集的香农熵
    样本类别出现的概率乘以log_2的样本类别出现的概率 然后求和
    H = -∑p(x_i)log_2p(x_i)
    :param data_set:
    :return:
    """
    num_entries = len(data_set)
    label_counts = {}
    # 获取标签值及其所出现的次数 yes: 2  no: 3
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    # 计算标签的香农熵
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def choose_best_feature_to_split(data_set):
    """
    选择最好的数据集划分方式
    :param data_set:
    :return:
    """
    num_features = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        # 获取样本中特征的集合，并去重
        feat_list = [example[i] for example in data_set]
        unique_values = set(feat_list)
        new_entropy = 0.0
        # 计算每种划分的信息熵
        for value in unique_values:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        # 获取最好的信息增益
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    """
    获取统计出来的数据最大的值
    :param class_list:
    :return:
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """
    创建树
    :param data_set: 数据集
    :param labels: 标签
    :return:
    """
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    best_feat = choose_best_feature_to_split(data_set)
    best_feature_label = labels[best_feat]
    my_tree = {best_feature_label: {}}
    del (labels[best_feat])
    feat_values = [example[best_feat] for example in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


def classify(input_tree: map, feat_labels, test_vec):
    """
    使用决策树进行分类
    :param input_tree: 决策树
    :param feat_labels: 标签
    :param test_vec: 测试样本
    :return:
    """
    class_label = None
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


def store_tree(input_tree, file_name):
    """
    保存模型
    :param input_tree: 决策树模型
    :param file_name: 文件名
    :return:
    """
    fw = open(file_name, 'wb')
    pickle.dump(input_tree, fw)
    fw.close()


def grab_tree(file_name):
    """
    加载模型
    :param file_name: 文件名
    :return:
    """
    fr = open(file_name, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    data_set, labels = create_data_set()
    calc_shannon_ent(data_set)
    # rest = split_data_set(data_set, 1, 0)
    # best_feature = choose_best_feature_to_split(data_set)
    # print(data_set[best_feature])
    tree = create_tree(data_set, labels)
    print(tree)
    print(str(tree))
    store_tree(tree, 'classifierStorage.txt')
    load_tree = grab_tree('classifierStorage.txt')
    print(load_tree)
