# _*_ coding: utf-8 _*_
# @time : 2021/3/31 22:19
# @author : yanms
# @version：V 0.1
# @file : main.py
# @desc :
from ch03_decision_tree import trees
from ch03_decision_tree import treePlotter


def test_predict():
    """
    测试决策树
    :return:
    """
    my_data, labels = trees.create_data_set()
    print(labels)
    my_tree = treePlotter.retrieve_tree(0)
    print(my_tree)
    res = trees.classify(my_tree, labels, [1, 0])
    print(res)
    res = trees.classify(my_tree, labels, [1, 1])
    print(res)


def test_lenses():
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_tree = trees.create_tree(lenses, lenses_labels)
    print(lenses_tree)
    treePlotter.create_plot(lenses_tree)


if __name__ == '__main__':
    test_lenses()
