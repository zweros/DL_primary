import numpy as np
import matplotlib.pyplot as plt

"""

numpy.meshgrid()——生成网格点坐标矩阵。： https://blog.csdn.net/lllxxq141592654/article/details/81532855


语法：X,Y = numpy.meshgrid(x, y)
输入的x，y，就是网格点的横纵坐标列向量（非矩阵）
输出的X，Y，就是坐标矩阵。

"""


def test01():
    x = np.array([[0, 1, 3],
                  [0, 1, 3]])

    y = np.array([[0, 0, 0],
                  [1, 1, 1]])

    plt.plot(x, y,
             color='red',  # 全部点设置为红色
             marker='.',  # 点的形状为圆点
             linestyle='-.')  # 线型为空，也即点与点之间不用线连接
    plt.grid(True)
    plt.show()


def test02():
    x = np.linspace(0, 1000, 20)
    y = np.linspace(0, 500, 20)

    X, Y = np.meshgrid(x, y)

    plt.plot(X, Y,
             color='limegreen',  # 设置颜色为limegreen
             marker='.',  # 设置点类型为圆点
             linestyle='')  # 设置线型为空，也即没有线连接点
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    test02()
