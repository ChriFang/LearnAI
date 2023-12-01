import numpy as np
import torch
import math

import TrainDataLoader


x_train, y_train, x_valid, y_valid = TrainDataLoader.load_train_data()
print("x_train ", x_train.shape)
print("x_valid ", x_valid.shape)
# print(x_train, y_train)


bs = 64  # batch size
lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

# n是行数，c是列数
n, c = x_train.shape


from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()

# 定义丢失函数
loss_func = torch.nn.functional.cross_entropy


# 计算精度函数
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        # replace our previous manually coded optimization
        opt.step()
        opt.zero_grad()


xb = x_train[0:bs]  # a mini-batch from x
yb = y_train[0:bs]
pred = model(xb)
print("loss: ", loss_func(pred, yb))
print("accuracy: ", accuracy(pred, yb))

