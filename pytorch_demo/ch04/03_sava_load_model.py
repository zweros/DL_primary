import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""
当保存和加载模型时，需要熟悉三个核心功能：

torch.save：将序列化对象保存到磁盘。此函数使用Python的pickle模块进行序列化。使用此函数可以保存如模型、tensor、字典等各种对象。
torch.load：使用pickle的unpickling功能将pickle对象文件反序列化到内存。此功能还可以有助于设备加载数据。
torch.nn.Module.load_state_dict：使用反序列化函数 state_dict 来加载模型的参数字典。
"""


# 定义模型
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = TheModelClass()
# model = model.to(device)

loss = nn.CrossEntropyLoss()

# 初始化优化器
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



# 打印模型的状态字典
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# 打印优化器的状态字典
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

PATH = './model/model.pl'
# 保存模型
torch.save(model.state_dict(), PATH)
# 加载模型
model = TheModelClass()
model.load_state_dict(torch.load(PATH))
model.eval()
