import numpy as np
import torch
import math

import TrainDataLoader


x_train, y_train, x_valid, y_valid = TrainDataLoader.load_train_data()
print("x_train ", x_train.shape)
print("x_valid ", x_valid.shape)
# print(x_train, y_train)


# 随机产生权重和变差值
weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)


# 激活函数
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

# 定义模型
def model(xb):
    return log_softmax(xb @ weights + bias)

# 定义丢失函数
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()
loss_func = nll

# 计算精度函数
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

bs = 64  # batch size
lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

# n是行数，c是列数
n, c = x_train.shape

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb) # 通过模型预测
        loss = loss_func(pred, yb) # 通过与实际结果比对，计算丢失值

        loss.backward() # 反向传播
        with torch.no_grad():
           # print("grad: ", weights.grad)
            weights -= weights.grad * lr  # 调整权重值
            bias -= bias.grad * lr # 调整偏差值
            weights.grad.zero_()  # 清空梯度值（为什么要清空？）
            bias.grad.zero_()


xb = x_train[0:bs]  # a mini-batch from x
yb = y_train[0:bs]
pred = model(xb)
print("loss: ", loss_func(pred, yb))
print("accuracy: ", accuracy(pred, yb))


