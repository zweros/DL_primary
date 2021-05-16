import numpy as np

"""
 flatten 降维： https://blog.csdn.net/lilong117194/article/details/78288795
"""

x = np.array([
    [1, 2],
    [3, 4]
])
print('二维：', x)

x = x.flatten()
print('一维： ', x)
