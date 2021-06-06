# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

"""
PyTorch之迁移学习

实际中，基本没有人会从零开始（随机初始化）训练一个完整的卷积网络，
因为相对于网络，很难得到一个足够大的数据集[网络很深, 需要足够大数据集]。
通常的做法是在一个很大的数据集上进行预训练得到卷积网络ConvNet, 
然后将这个ConvNet的参数作为目标任务的初始化参数或者固定这些参数。

迁移学习的两个主要场景：
- 微调Convnet：使用预训练的网络(如在imagenet 1000上训练而来的网络)来初始化自己的网络，而不是随机初始化。其他的训练步骤不变。
- 将Convnet看成固定的特征提取器:首先固定ConvNet除了最后的全连接层外的其他所有层。
最后的全连接层被替换成一个新的随机 初始化的层，只有这个新的层会被训练[只有这层参数会在反向传播时更新]


1、一是内存不足，重启一下pycharm
2、把num_works设置为0(这个设置的地方，是你的代码中一般位于数据导入部分，是自己设置)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=1)

"""


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """
    训练模型
    :param model: 网络模型
    :param criterion:  损失函数
    :param optimizer: 优化器
    :param scheduler: 一个来自 torch.optim.lr_scheduler的学习速率调整类的对象(LR scheduler object)。
    :param num_epochs:  训练整个数据集的次数，默认 25次
    :return:
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 零参数梯度
                optimizer.zero_grad()

                # 前向
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 后向+仅在训练阶段进行优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # scheduler.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 深度复制mo
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model


# 一个通用的展示少量预测图片的函数
def visualize_model(model, num_images=6):
    """
    一个通用的展示少量预测图片的函数
    :param model:
    :param num_images:
    :return:
    """
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # batch_size =  inputs.size()[0]
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def imshow_test(dataloaders):
    """
      从数据集中获取图片
    :param dataloaders:
    :return:
    """
    # 获取一批训练数据
    inputs, classes = next(iter(dataloaders['train']))
    print('inputs = ', inputs.size(), 'batch_size = ', inputs.size()[0])
    print('classes = ', classes)
    # 批量制作网格
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])


def mini_conVnet():
    # ==========.场景1：微调ConvNet ==================
    # 1. 加载预训练模型并重置最终完全连接的图层。
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # 观察所有参数都正在优化
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # 每7个epochs衰减LR通过设置gamma=0.1
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.0005)

    # 训练和评估模型
    # （1）训练模型该过程在CPU上需要大约15 - 25分钟，但是在GPU上，它只需不到一分钟。
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
    # （2）模型评估效果可视化
    visualize_model(model_ft)

    plt.ioff()
    plt.show()


# 场景2：ConvNet作为固定特征提取器
def feature_get_tool():
    # =========场景2：ConvNet作为固定特征提取器 ========
    # 在这里需要冻结除最后一层之外的所有网络。通过设置requires_grad == Falsebackward()来冻结参数，
    # 这样在反向传播backward()的时候他们的梯度就不会被计算。
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    print(type(num_ftrs), 'num_ftrs = ', num_ftrs)
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    # （1）训练模型 在CPU上，与前一个场景相比，这将花费大约一半的时间，因为不需要为大多数网络计算梯度。但需要计算转发。
    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)
    # （2）模型评估效果可视化
    visualize_model(model_conv)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    plt.ion()  # interactive mode

    # 训练集数据扩充和归一化
    # 在验证集上仅需要归一化
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),  # 随机裁剪一个area然后再resize
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data/hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print('dataset_sizes: ', dataset_sizes)
    print('分类标签： ', class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 场景1：微调ConvNet  0.6666
    # mini_conVnet()

    # 场景2：ConvNet作为固定特征提取器  0.686275  0.947712
    # feature_get_tool()
