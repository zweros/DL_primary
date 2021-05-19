# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    # 设 置 3 个 参 数。第 1 个参数
    # normalize设置是否将输入图像正规化为0.0～1.0的值。如果将该参数设置
    # 为False，则输入图像的像素会保持原来的0～255。第2个参数flatten设置
    # 是否展开输入图像（变成一维数组）。如果将该参数设置为False，则输入图
    # 像为1 × 28 × 28的三维数组；若设置为True，则输入图像会保存为由784个
    # 元素构成的一维数组。
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))