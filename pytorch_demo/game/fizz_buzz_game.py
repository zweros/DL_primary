import numpy as np
import torch

"""
 游戏：fuzzbuzz
 数字从 1 开始报数，
 若数字是 3 的倍数则报 fizz， 若数字是 5 的倍数则报 buzz，
 若数字是 15 的倍数则报 fizzbuzz 若数字是其他正整数则原样输出
"""


# 编码
def fizz_buzz_encode(i):
    if i % 15 == 0: return 3
    if i % 5 == 0: return 2
    if i % 3 == 0: return 1
    return 0


# 解码
def fizz_buzz_decode(i, prediction):
    game_dict = {1: "fizz", 2: "buzz", 3: "fizzbuzz"}
    # return game_dict.get(prediction) or str(i)
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


def helper(i):
    return fizz_buzz_decode(i, fizz_buzz_encode(i))


# for i in range(1, 17):
#     result = helper(i)
#     print(result)


def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)][::-1])


NUM_DIGITS = 10
print(binary_encode(15, NUM_DIGITS))

# 训练数据
train_x = torch.tensor([binary_encode(d, NUM_DIGITS) for d in range(101, 2 ** NUM_DIGITS)], dtype=torch.float)
# 测试数据
test_x = torch.tensor([binary_encode(d, NUM_DIGITS) for d in range(1, 101)], dtype=torch.float)
# 标签
train_y = torch.tensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])
test_y = torch.tensor([fizz_buzz_encode(i) for i in range(1, 101)])

# print(train_x.shape)
# print(train_x[0:3])
# print(train_y.shape)
# print(train_y[0:3])


# 构建深度学习网络模型
# 训练模型 forward
# 计算 loss
# 反向传播
# 权重更新

# 输入大小， 中间层， 输出分类
INPUT_SIZE, HIDDEN_SIZE, OUT_SIZE = NUM_DIGITS, 100, 4

model = torch.nn.Sequential(
    torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
    torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN_SIZE, OUT_SIZE)
)

if torch.cuda.is_available():
    model = model.cuda()

# 定义损失函数
loss_crl = torch.nn.CrossEntropyLoss()
# 定义优化器
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 批处理大小
BATCH_SIZE = 128

for epoch in range(10000):
    run_loss = None
    for start in range(0, len(train_x), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = train_x[start: end]
        batchY = train_y[start: end]

        if torch.cuda.is_available():
            batchX = batchX.cuda()
        batchY = batchY.cuda()
        # forward
        y_pred = model(batchX)
        # 计算损失
        run_loss = loss_crl(y_pred, batchY)

        # 清空优化
        optimizer.zero_grad()
        run_loss.backward()  # backward pass
        optimizer.step()  # gradient descent
    print('Epoch', epoch, run_loss.item())
print('Finished Trans')

if torch.cuda.is_available():
    test_x = test_x.cuda()

with torch.no_grad():
    test_y_pred = model(test_x)

pre_list = test_y_pred.max(1)[1].cpu().data.tolist()

predictions = zip(range(1, 101), pre_list)
print([fizz_buzz_decode(i, x) for i, x in predictions])

pre_list = np.array(pre_list)
test_y = np.array(test_y)

print('acc = ', (pre_list == test_y).sum() / 100, '%')

# 保存模型参数
# torch.save(model.state_dict(), 'model/test.pl')
