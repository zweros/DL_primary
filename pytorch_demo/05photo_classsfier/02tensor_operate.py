from __future__ import print_function
import torch

x = torch.randn(2, 3)
y = torch.rand(2, 3)

print('x = ', x)
print('y = ', y)
# 方式一
print(x + y)
# 方式二
print(torch.add(x, y))

# 加法: 提供一个输出 tensor 作为参数
result = torch.empty(2, 3)
torch.add(x, y, out=result)
print(result)

# 原地操作
# 加法: in-place
y.add_(x)
print(y)
# 你可以使用标准的  NumPy 类似的索引操作
print(y[1, :])
print('====' * 10)

# 改变大小：如果你想改变一个 tensor 的大小或者形状，你可以使用 torch.view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
z1 = x.view(2, -1)
print(x.size(), y.size(), z.size(), z1.size())

# 如果你有一个元素 tensor ，使用 .item() 来获得这个 value 。
x = torch.randn(1)
print(x)
print(x.item())



