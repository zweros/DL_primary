from __future__ import print_function
import torch

"""
    https://pytorch123.com/SecondSection/what_is_pytorch/#tensors
    print("tensors)
    一维向量  
    二维矩阵
    三维及以上张量
"""

# 构造一个5x3矩阵，不初始化。
print(" 构造一个5x3矩阵，不初始化。")
x = torch.empty(2, 3)
print(x)

# 构造一个随机初始化的矩阵
print(" 构造一个随机初始化的矩阵")
random_num = torch.rand(2, 3)
print(random_num)

# 构造一个矩阵全为 0，而且数据类型是 long.
print(" 构造一个矩阵全为 0，而且数据类型是 long.")
torch_zeros = torch.zeros(2, 3, dtype=torch.float32)
print(torch_zeros)

# 构造一个张量，直接使用数据
print(" 构造一个张量，直接使用数据")
torch_tensor = torch.tensor([
    [5.5, 3, 2.0],
    [5.5, 3, 1.0]
])
print(torch_tensor)

# 创建一个 tensor 基于已经存在的 tensor。
print("创建一个 tensor 基于已经存在的 tensor。")
x = x.new_ones(2, 3, dtype=torch.double)
# new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)
# override dtype!
print(x)
# result has the same size
# 获取它的维度信息:
print(x.size())
