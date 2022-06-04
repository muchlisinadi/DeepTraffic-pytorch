import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LitBase


class CNN(nn.Module):
    def __init__(self, input_size=28, hidden_size=50, num_classes=10, **kwargs):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, self.input_size, self.input_size)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class LitCNN(LitBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = CNN(**kwargs)

    def forward(self, x):
        out = self.net(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def loss_function(self, pred, labels):
        return F.cross_entropy(pred, labels)
