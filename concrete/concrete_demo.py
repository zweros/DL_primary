import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

"""
    水泥案例 - 预测水泥强度
    
技术点： 算法选择，超参数设置， 参数原理
归一化：
1. 将 x 值变小，会导致权重 w 值变大， 使得训练的结果更好
2. 梯度下降更均匀
注意：归一化对训练集的 x 和测试集的 x 都需要归一化， y 值不需要归一化

归一化的方式
1. 最大最小值归一化  MaxMinScaler
2. 均值归一化
3. 标准差归一化
4. 均值和标准差归一化 StandardScaler


参考：
- 解决pandas中打印DataFrame行列显示不全的问题： https://blog.csdn.net/gulie8/article/details/102794117

"""

# 读取源数据
data = pd.read_csv('data/concrete.csv')
# print('data = ', data)
# print('data shape', data.shape)

# 显示所有列
pd.set_option('display.max_columns', None)

# 取出前 8 条数据
data_head = data.head(8)
# 描述,  合计， 均值， 标准差， 最小值， 25%百分位数， 50%百分位数， 最大值
print(data.describe())
# print('data.head(8) = ', data_head, sep='\n')
# print("type = ", type(data))
# f.iloc[1:3, 0:3]

# 训练集
x_train = data.iloc[:800, :-1]
y_train = data.iloc[:800, -1:]
# 测试集
x_test = data.iloc[800:, :-1]
y_test = data.iloc[800:, -1:]

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

# 归一化
# 采用均值和标准方差归一化
scaler = StandardScaler(with_mean=True, with_std=True)
# 采用训练集训练求出均值和方差，应用在 x_train,x_test 训练集上做归一化处理
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# A column-vector y was passed when a 1d array was expected.
# Please change the shape of y to (n_samples, ), for example using ravel().
# 二维列向量 -> 一维数组
y_train = np.array(y_train).ravel()
# print(y_train)

# ConvergenceWarning:
# Stochastic Optimizer: Maximum iterations (200) reached and
# the optimization hasn't converged yet.  % self.max_iter, ConvergenceWarning)

# 神经网络回归
reg = MLPRegressor(activation='relu', hidden_layer_sizes=(5,), solver='sgd', random_state=1, max_iter=300)
# 训练
reg.fit(x_train, y_train)
# 验证
y_predicted = reg.predict(x_test)
# 损失函数
# 这里采用 sqrt 开根号，使得误差更接近于真实
# mean_squared_error MES
res = np.sqrt(mean_squared_error(y_pred=y_predicted, y_true=y_test))
print('res = ', res)
