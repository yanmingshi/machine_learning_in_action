# _*_ coding: utf-8 _*_
# @time : 2021/4/1 22:22
# @author : yanms
# @version：V 0.1
# @file : bayes.py
# @desc :
import operator

import numpy as np
import jieba
import re  # 正则表达式

import feedparser


def load_data_set():
    """
    词表到向量的转换函数
    :return:
    """
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1 代表侮辱性文字, 0 代表正常言论
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


def create_vocab_list(data_set):
    """
    创建一个包含在所有文档中出现的不重复词的列表
    :param data_set:
    :return:
    """
    vocab_set = set([])
    for document in data_set:
        # 取两个集合的并集
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_of_words2vec(vocab_list, input_set):
    """
    获得文档向量, 文档中的词存在词汇表中，则置为1
    :param vocab_list: 词汇表
    :param input_set: 某个文档
    :return:
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("the word %s is not in my vocabulary!" % word)
    return return_vec


def bag_of_words2vec(vocab_list, input_set):
    """
    获得文档向量, 文档中的词存在词汇表中，则加1  文档词袋模型
    :param vocab_list: 词汇表
    :param input_set: 某个文档
    :return:
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
        else:
            print("the word %s is not in my vocabulary!" % word)
    return return_vec


def train(train_matrix, train_category):
    """
    训练朴素贝叶斯分类模型
    :param train_matrix:
    :param train_category:
    :return:
    """
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    # 侮辱类出现的概率 先验
    p_abusive = sum(train_category) / float(num_train_docs)

    # 初始化概率
    # p0_num = np.zeros(num_words)
    # p1_num = np.zeros(num_words)
    # p0_denom = 0.0
    # p1_denom = 0.0
    # 为防止出现0的情况影响分类效果，初始化为做调整
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0

    # 向量相加
    # 统计每个词在不同分类中出现的次数以及词在不同分类中出现的总次数
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])

    # 对每个元素做除法 求出每个词在不同分类中出现的概率，也就是p（w_i|Y）的条件概率
    # p1_vec = p1_num / p1_denom
    # p0_vec = p0_num / p0_denom
    # 改成求对数，防止出现下溢情况（太多个小于1的数相乘，浮点运行结果可能为0）
    p1_vec = np.log(p1_num / p1_denom)
    p0_vec = np.log(p0_num / p0_denom)
    return p0_vec, p1_vec, p_abusive


def classify(vec2classify, p0_vec, p1_vec, p_class1):
    """
    分类
    :param vec2classify:
    :param p0_vec:
    :param p1_vec:
    :param p_class1:
    :return:
    """
    p1 = sum(vec2classify * p1_vec) + np.log(p_class1)
    p0 = sum(vec2classify * p0_vec) + np.log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def testing():
    """
    恶意留言测试代码
    :return:
    """
    list_o_posts, list_classes = load_data_set()
    my_vocab_list = create_vocab_list(list_o_posts)
    train_mat = []
    for postin_doc in list_o_posts:
        train_mat.append(set_of_words2vec(my_vocab_list, postin_doc))
    p0v, p1v, pab = train(np.array(train_mat), np.array(list_classes))
    test_entry = ['love', 'my', 'dalmation']
    this_doc = np.array(set_of_words2vec(my_vocab_list, test_entry))
    print(test_entry, 'classified as: ', classify(this_doc, p0v, p1v, pab))
    test_entry = ['stupid', 'garbage']
    this_doc = np.array(set_of_words2vec(my_vocab_list, test_entry))
    print(test_entry, 'classified as: ', classify(this_doc, p0v, p1v, pab))


def word_cut():
    """
    jieba分词
    :return:
    """
    word1 = jieba.cut("无新增死亡病例；新增疑似病例3例")
    word2 = jieba.cut("无现有疑似病例。累计确诊病例1869例，累计治愈出院病例1787例，无死亡病例")
    word3 = jieba.cut("累计治愈出院病例78413例，累计死亡病例4634例，累计报告确诊病例83378例，现有疑似病例13例")
    # 转换成列表,然后转换成字符串
    con1 = " ".join(list(word1))
    con2 = " ".join(list(word2))
    con3 = " ".join(list(word3))
    print(con1)
    print(con2)
    print(con3)

    return con1, con2, con3


def text_parse(text):
    """
    分词，并过滤掉标点符号和长度小于3的单词
    :param text: 待分词文本
    :return:
    """
    list_of_tokens = re.split(r'\W*', text)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spam_test():
    """
    垃圾邮件测试代码
    :return:
    """
    doc_list = []
    class_list = []
    full_text = []

    # 导入并且解析文本
    for i in range(1, 26):
        word_list = text_parse(open('email/spam/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)  # 垃圾邮件标记
        word_list = text_parse(open('email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)  # 正常邮件标记

    vocab_list = create_vocab_list(doc_list)
    training_set = list(range(50))
    test_set = []

    # 划分测试集和训练集
    for i in range(10):
        rand_index = int(np.random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])

    train_mat = []
    train_classes = []

    # 训练模型
    for doc_index in training_set:
        train_mat.append(set_of_words2vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0v, p1v, p_spam = train(np.array(train_mat), np.array(train_classes))

    # 测试
    error_count = 0
    for doc_index in test_set:
        word_vector = set_of_words2vec(vocab_list, doc_list[doc_index])
        if classify(np.array(word_vector), p0v, p1v, p_spam) != class_list[doc_index]:
            error_count += 1
            print("classification error", doc_list[doc_index])
    print('the error rate is: ', float(error_count)/len(test_set))


def calc_most_freq(vocab_list, full_text):
    """
    统计词汇表中单词在文章中出现的频率，返回出现频率最高的30个
    :param vocab_list: 词汇表
    :param full_text: 待统计文本
    :return:
    """
    freq_dict = {}
    for token in vocab_list:
        freq_dict[token] = full_text.count(token)
        # sorted用法参考出ch02 knn.py
        sorted_freq = sorted(freq_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sorted_freq[:30]


def local_words(feed1, feed0):
    """

    :param feed1:
    :param feed2:
    :return:
    """
    doc_list = []
    class_list = []
    full_text = []
    min_len = min(len(feed1['entries']), len(feed0['entries']))

    # 每次访问一条RSS源
    for i in range(min_len):
        word_list = text_parse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)

        word_list = text_parse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)

    vocab_list = create_vocab_list(doc_list)

    # 去掉那一些出现频率最高的词
    top30_words = calc_most_freq(vocab_list, full_text)
    for pair_w in top30_words:
        if pair_w[0] in vocab_list:
            vocab_list.remove(pair_w[0])

    training_set = range(2*min_len)
    test_set = []

    for i in range(20):
        rand_index = int(np.random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])

    train_mat = []
    train_classes = []

    for doc_index in training_set:
        train_mat.append(bag_of_words2vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0v, p1v, p_spam = train(np.array(train_mat), np.array(train_classes))
    error_count = 0

    for doc_index in test_set:
        word_vector = bag_of_words2vec(vocab_list, doc_list[doc_index])
        if classify(np.array(word_vector), p0v, p1v, p_spam) != class_list[doc_index]:
            error_count += 1

    print('the error rate is: ', float(error_count)/len(test_set))
    return vocab_list, p0v, p1v


if __name__ == '__main__':
    # list_o_posts, list_classes = load_data_set()
    # my_vocab_list = create_vocab_list(list_o_posts)
    # print(my_vocab_list)
    # my_vec = set_of_words2vec(my_vocab_list, list_o_posts[0])
    # print(my_vec)
    # train_mat = []
    # for postin_doc in list_o_posts:
    #     train_mat.append(set_of_words2vec(my_vocab_list, postin_doc))
    #
    # print(train_mat)
    # p0v, p1v, pab = train(train_mat, list_classes)
    # print(p0v, p1v, pab)
    spam_test()