import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class CIFAR10_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 64
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 8, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 16, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(16 * 8 * 8, 256, bias=False)
        self.bn1d1 = nn.BatchNorm1d(256, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(256, 128, bias=False)
        self.bn1d2 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc3 = nn.Linear(128, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.leaky_relu(self.bn1d1(x))
        x = self.fc2(x)
        x = F.leaky_relu(self.bn1d2(x))
        x = self.fc3(x)
        return x


class CIFAR10_LeNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 64
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(3, 8, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 16, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(16 * 8 * 8, 256, bias=False)
        self.bn1d1 = nn.BatchNorm1d(256, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(256, 128, bias=False)
        self.bn1d2 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc3 = nn.Linear(128, self.rep_dim, bias=False)

        # Decoder
        self.fc4 = nn.Linear(self.rep_dim, 128, bias=False)
        self.bn1d4 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc5 = nn.Linear(128, 256, bias=False)
        self.bn1d5 = nn.BatchNorm1d(256, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(16, 16, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(16, 8, 5, bias=False, padding=2)
        self.bn2d4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(8, 3, 5, bias=False, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.leaky_relu(self.bn1d1(x))
        x = self.fc2(x)
        x = F.leaky_relu(self.bn1d2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        x = F.leaky_relu(self.bn1d4(x))
        x = self.fc5(x)
        x = F.leaky_relu(self.bn1d5(x))
        x = x.view(x.size(0), int(256 / (4 * 4)), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.conv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d3(x)), scale_factor=2)
        x = self.conv4(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.conv5(x)
        x = torch.sigmoid(x)
        return x
