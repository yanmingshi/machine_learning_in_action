# _*_ coding: utf-8 _*_
# @time : 2021/3/20 22:02
# @author : yanms
# @version：V 0.1
# @file : 1.7_numpy.py
# @desc :

import numpy as np

np.random.seed(3)

if __name__ == '__main__':
    m_arr = np.random.rand(4, 4)
    print(m_arr, m_arr.shape, type(m_arr))
    print("*"*30)

    m_mat = np.mat(m_arr)
    print(m_mat, m_mat.shape, type(m_mat))
    print("*" * 30)

    invert_mat = m_mat.I
    print(invert_mat)
    print("*" * 30)

    print(m_mat*invert_mat)
    print("*" * 30)

    print(np.eye(4))
