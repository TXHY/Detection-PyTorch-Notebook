
# ---------------------------------------
# @date: 2020-7-8
# @author: xiaozhi
# @project: pytorch语法
# ---------------------------------------

import torch
import numpy as np


def printLog(x, name):
    print("--------------------------------------")
    print(">>>" + str(name) + ": ")
    print(x)
    print("\n")

def tensor():
    # ------------------------------------------------------------
    # 基本张量创建(基本与numpy一样)
    x1 = torch.empty(5, 3)  # 创建一个未初始化的5x3张量
    x2 = torch.rand(5, 3)  # 创建一个随机初始化的5x3张量
    x3 = torch.zeros(5, 3, dtype=torch.long)  # 创建一个5x3的0张量，类型为long
    x4 = torch.tensor([2, 3, 4, 5], dtype=torch.float32)  # 直接从数组创建张量
    x5 = torch.ones(5, 3, dtype=torch.double)  # 创建一个5x3的单位张量，类型为double
    x6 = torch.randn_like(x5, dtype=torch.float)  # 从已有的张量创建相同维度的新张量，并且重新定义类型为float
    x6_2 = torch.arange(1,5)

    # -------------------------------------------------------------
    # 维度信息
    x7 = x5.size()  # 或 x5.shape  打印一个张量的维度
    x8 = x5.view(x5.shape[0] * x5.shape[1])  # 从二维(高维)降到一维
    x9 = x8.view(-1, 5)  # 从一维转换成高维
    # x9 = x8.resize(-1,5)
    # x9 = x8.reshape(-1,5)

    # -------------------------------------------------------------
    # 张量加法
    x10_1 = x2 + x5  # 第一种方法
    x10_2 = torch.add(x2,x5)  # 第二种方法
    x10_3 = torch.add(x2, x5, out=x1)  # 第三种方法
    x10_4 = x2.add_(x5)  # 第四种方法

    # ------------------------------------------------------------
    # 数据选择
    x11 = x5[:2, :]  # 前两行,所有列

    # ------------------------------------------------------------
    # 与numpy的操作
    x12_1 = x5.numpy()  # torch转numpy
    x12_2 = np.ones((5,3))
    x12_3 = torch.from_numpy(x12_2)

    # ------------------------------------------------------------
    # 原地操作副: 这种操作被执行后,本身的tensor也会跟着改变
    x13_1 = x2.add_(x5)  # 执行后,tensor即x2本身也会被改变
    x13_2 = x2.resize_(15)

    # ------------------------------------------------------------
    # 排序与取极值(min等常规函数与numpy差不多的用法)
    x14_1 = x1.sort()[0]  # 默认行内从小到大排序
    x14_2 = x1.sort(descending=True)[0]  # 行内从大到小排序
    x14_3 = x1.sort()[1]  #
    x14_4 = x1.sort(descending=True)[1]  #
    x15_1 = x1.max()
    x15_2 = x1.min()

    logs = [x1,x2,x3,x4,x5,x6,x6_2,x7,x8,x9,x10_1,x10_2,x10_3,x10_4,
            x11,x12_1,x12_2,x12_3,x2,x14_1,x14_2,x14_3,x14_4,x15_1,x15_2,
            ]
    names = ['x1','x2','x3','x4','x5','x6','x6_2','x7','x8','x9','x10_1','x10_2','x10_3','x10_4',
             'x11','x12_1','x12_2','x12_3','x2','x14_1','x14_2','x14_3','x14_4','x15_1','x15_2'
             ]
    for i, log in enumerate(logs):
        printLog(log, names[i])


def createModel():

    pass


if __name__ == "__main__":
    tensor()

# 3 4 5 6 7 8 9

