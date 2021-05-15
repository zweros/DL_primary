import paddle
from paddle.vision.transforms import ToTensor

"""
 手写数字识别

学习文档地址：https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/02_paddle2.0_develop/01_quick_start_cn.html

深度学习任务一般分为几个核心步骤：
 1.数据集的准备和加载；
 2.模型构建；
 3.模型训练；
 4.模型评估。


"""

train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
val_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())

print(train_dataset.mode)
print(train_dataset.transform)


mnist = paddle.nn.Sequential(
    paddle.nn.Flatten(),
    paddle.nn.Linear(784, 512),
    paddle.nn.ReLU(),
    paddle.nn.Dropout(0.2),
    paddle.nn.Linear(512, 10)
)


# 预计模型结构生成模型对象，便于进行后续的配置、训练和验证
model = paddle.Model(mnist)

# 模型训练相关配置，准备损失计算方法，优化器和精度计算方法
model.prepare(paddle.optimizer.Adam(parameters=model.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())

# 开始模型训练
model.fit(train_dataset,
          epochs=5,
          batch_size=64,
          verbose=1)

model.evaluate(val_dataset, verbose=0)



