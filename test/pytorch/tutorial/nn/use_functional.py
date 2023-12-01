import numpy as np
import torch
import math

import TrainDataLoader


x_train, y_train, x_valid, y_valid = TrainDataLoader.load_train_data()
print("x_train ", x_train.shape)
print("x_valid ", x_valid.shape)
# print(x_train, y_train)


from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias

model = Mnist_Logistic()

# 定义丢失函数
loss_func = torch.nn.functional.cross_entropy

# 计算精度函数
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

bs = 64  # batch size
lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

# n是行数，c是列数
n, c = x_train.shape

def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

fit()


xb = x_train[0:bs]  # a mini-batch from x
yb = y_train[0:bs]
pred = model(xb)
print("loss: ", loss_func(pred, yb))
print("accuracy: ", accuracy(pred, yb))


