from sklearn.neural_network import MLPClassifier

"""

sk learn 官网分类案例：https://scikit-learn.org/stable/modules/neural_networks_supervised.html

jupyter notebook
"""

X = [[0., 0.], [1., 1.]]
y = [0, 1]

"""
    solver  权限优化器 
    alpha   收敛值
    hidden_layer_sizes 隐藏层数量， （5,2）表示有二个隐藏层， 第一个有5 神经元， 第二个有2 神经元 
    random_state 随机数种子， 确保随机数生成都是唯一的
"""

clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=1000)
res1 = clf.fit(X, y)
print('res1 = ', res1)

# 预测值
predict_result = clf.predict([[2., 2.], [-1., -2.], [-2, -2], [3, 3]])
print('predict_result = ', predict_result)
# 预测值概率
predict_probability_result = clf.predict_proba([[2., 2.], [1., 2.]])
# [[0.15882012 0.84117988][0.15882012 0.84117988]]
print('predict_probability_result = ', predict_probability_result)

# 权重矩阵
coef_weight = [coef.shape for coef in clf.coefs_]
# [(2, 5), (5, 2), (2, 1)]
# (2, 5)  输入层2个神经元 -> 中间层5个神经元
# (5, 2) 中间层5个神经元 -> 中间层 2个神经元
# (2, 1) 中间层2个神经元 -> 输出层 1个神经元
print('coef_weight = ', coef_weight)

# 截取  bias
intercepts_shape = [intercept.shape for intercept in clf.intercepts_]
#  [(5,), (2,), (1,)]
print('intercepts_shape = ', intercepts_shape)
