import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

# 获取房价数据集
housing = fetch_california_housing(data_home='./scikit_learn_data', download_if_missing=True)
# m 行， n 列
m, n = housing.data.shape
feature_names = housing.feature_names
# 20640 8
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(m, n)
print(feature_names)
print(housing.data[0:3], housing.target[0:3], type(housing.target[0:3]))

# np.c_  将两个矩阵拼接在一起
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
print(housing_data_plus_bias.shape)
# 创建两个 Tensorflow 的常量节点 X 和 y ， 去持有数据和标签
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
# X = tf.constant(housing.data, dtype=tf.float32, name='X')
# 行向量转为列向量
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
# 使用 Tensorflow 框架提供的矩阵操作操作求 theta
XT = tf.transpose(X)
# 解析解一步计算出最优解
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
# Tensor("MatMul_2:0", shape=(9, 1), dtype=float32)
print(theta)

with tf.Session() as session:
    theta_value = theta.eval()
    print(theta_value)
