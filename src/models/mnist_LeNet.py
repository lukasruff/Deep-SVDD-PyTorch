import torch.nn as nn
import torch.nn.functional as F

from base.base_model import BaseModel


class MNIST_LeNet(BaseModel):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4)

        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
