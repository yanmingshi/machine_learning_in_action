# _*_ coding: utf-8 _*_
# @time : 2021/3/29 22:53
# @author : yanms
# @version：V 0.1
# @file : treePlotter.py
# @desc :

import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plot_node(node_text, center_pt, parent_pt, node_type):
    """
    绘制node节点
    annotate函数是为绘制图上指定的数据点xy添加一个nodeTxt注释
    :param node_text: 给数据点xy添加一个注释，xy为
    :param center_pt: 注释的中间点坐标
    :param parent_pt: 数据点的开始绘制的坐标,位于节点的中间位置
    :param node_type: 注释样式
    :return:
    """
    create_plot.ax1.annotate(node_text,
                             xy=parent_pt,
                             xycoords='axes fraction',
                             xytext=center_pt,
                             textcoords='axes fraction',
                             va="center",
                             ha="center",
                             bbox=node_type,
                             arrowprops=arrow_args)


def create_plot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)
    plot_node('决策节点', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node('叶子节点', (0.8, 0.1), (0.3, 0.5), leaf_node)
    plt.show()


def get_num_leafs(my_tree: map):
    """
    获取叶子节点的数目
    :param my_tree:
    :return:
    """
    num_leafs = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(my_tree: map):
    """
    获取树的深度
    :param my_tree:
    :return:
    """
    max_depth = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def retrieve_tree(i):
    list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                     {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return list_of_trees[i]


def plot_mid_text(cntr_pt, parent_pt, txt_string):
    """
    绘制线中间的文字(0和1)的绘制
    :param cntr_pt:
    :param parent_pt:
    :param txt_string:
    :return:
    """
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string)


def plot_tree(my_tree, parent_pt, node_txt):
    """
    绘制树
    :param my_tree: 要花的树
    :param parent_pt: 初始点位置
    :param node_txt: 节点标签值
    :return:
    """
    num_leafs = get_num_leafs(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = list(my_tree.keys())[0]
    # 计算子节点的坐标
    cntr_pt = (plot_tree.x_offset + (1.0 + float(num_leafs)) / 2.0 / plot_tree.total_width, plot_tree.y_offset)
    # 绘制线上的文字
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    # 绘制节点
    plot_node(first_str, cntr_pt, parent_pt, decision_node)
    # 获取下一个键值
    second_dict = my_tree[first_str]
    # 计算节点y方向上的偏移量，根据树的深度0
    plot_tree.y_offset = plot_tree.y_offset - 1.0 / plot_tree.total_depth
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            # 递归绘制树
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            # 更新x的偏移量,每个叶子结点x轴方向上的距离为 1/plotTree.total_width
            plot_tree.x_offset = plot_tree.x_offset + 1.0 / plot_tree.total_width
            plot_node(second_dict[key], (plot_tree.x_offset, plot_tree.y_offset), cntr_pt, leaf_node)
            # 绘制箭头上的标志
            plot_mid_text((plot_tree.x_offset, plot_tree.y_offset), cntr_pt, str(key))
    plot_tree.y_offset = plot_tree.y_offset + 1.0 / plot_tree.total_depth


def create_plot(in_tree):
    """
    绘制决策树
    :param in_tree: 格式为{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    :return:
    """
    fig = plt.figure(1, facecolor='white')
    # 清除figure
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.total_width = float(get_num_leafs(in_tree))
    plot_tree.total_depth = float(get_tree_depth(in_tree))
    plot_tree.x_offset = -0.5 / plot_tree.total_width
    plot_tree.y_offset = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()


if __name__ == '__main__':
    # create_plot()
    my_tree = retrieve_tree(1)
    # get_num_leafs(my_tree)
    # get_tree_depth(my_tree)
    create_plot(my_tree)
