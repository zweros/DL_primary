import torch

# 输入大小， 中间层， 输出分类
INPUT_SIZE, HIDDEN_SIZE, OUT_SIZE = 10, 100, 4

model = torch.nn.Sequential(
    torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
    torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN_SIZE, OUT_SIZE)
)

model.load_state_dict(torch.load('model/test.pl'))

x = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float)
print(x.shape)
print(model.parameters())

with torch.no_grad():
    pre_y = model(x)
    print(pre_y)
