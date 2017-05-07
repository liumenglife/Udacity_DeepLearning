#!/usr/bin/python
# -*- coding: utf-8 -*-

# TODO: Add import statements
import numpy as np

# scalars 标量
s = np.array(5)
s.shape
x = s + 3
type(x)
x.shape

# Vectors 向量
v = np.array([1, 2, 3])
v.shape

x = v[1]
x
v[1:]

# Matrices 矩阵
m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
m.shape
m[1]
m[1][2]

# Tensors 张量
t = np.array([[[[1], [2]], [[3], [4]], [[5], [6]]], [[[7], [8]], [[9], [10]], [
             [11], [12]]], [[[13], [14]], [[15], [16]], [[17], [17]]]])
t.shape
t[2][1][1][0]

# 更改形状
v = np.array([1, 2, 3, 4])
v.shape
x = v.reshape(1, 4)
x.shape
x = v.reshape(4, 1)
x.shape

y = v[None, :]
y.shape

y = v[:, None]
y.shape


# 元素级运算
# Python中的方式
values = [1, 2, 3, 4, 5]
for i in range(len(values)):
    values[i] += 5
values

# NumPy 中的方式
values = [1, 2, 3, 4, 5]
values = np.array(values) + 10
values
v2 = np.array([1, 2, 3, 4, 5])
z = v2 + 20
z

# Numpy 矩阵乘法
# 元素级乘法
m = np.array([[1, 2, 3], [4, 5, 6]])
m
n = m * 0.25
n
m * n
np.multiply(m, n)

# 矩阵乘积
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
a
a.shape

b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
b
b.shape

c = np.matmul(a, b)
c
c.shape


# NumPy的dot函数
# 有时候，在你以为用matmul函数的地方，可能会看到NumPy的dot函数。事实证明，如果矩阵是二维的，那么dot和matmul函数的结果是相同的*。

a = np.array([[1, 2], [3, 4]])
a

np.dot(a, a)

a.dot(a)

np.matmul(a, a)

# NumPy中的装置
m = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
m
m.T
m_t = m.T
m_t[3][1]
m_t[3][1] = 200
m_t[3][1]
m_t
m
m_t.shape

m_t2 = m_t[:, :]
m_t2
m_t3 = m_t[:]
m_t3
m_t2[3][1] = 300
m_t3
m

m_t4 = m_t.copy()  # 深层拷贝，建立2个不同的副本
m_t4[3][1] = 500
m_t4
m

# 实际用例
inputs = np.array([[-0.27, 0.45, 0.64, 0.31]])
inputs

inputs.shape

weights = np.array([[0.02, 0.001, -0.003, 0.036],
                    [0.04, -0.003, 0.025, 0.009],
                    [0.012, -0.045, 0.28, -0.067]])

weights

weights.shape

np.matmul(inputs, weights)

a=np.matmul(inputs,weights.T)
b=np.matmul(weights,inputs.T)
a.shape
b.shape
