import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LitBase


class SVM_Loss(nn.Module):
    def __init__(self):
        super(SVM_Loss, self).__init__()

    def forward(self, outputs, labels):
        return torch.sum(torch.clamp(1 - outputs.t() * labels, min=0)) / labels.shape[0]


class SVM(nn.Module):
    def __init__(self, input_size=28, num_classes=10, **kwargs):
        super(SVM, self).__init__()
        self.input_size = input_size
        self.fc = nn.Linear(input_size * input_size, num_classes)

    def forward(self, x):
        x = x.reshape(-1, self.input_size * self.input_size)
        x = self.fc(x)
        return x


class LitSVM(LitBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = SVM(**kwargs)
        self.svm_criterion = SVM_Loss()

    def forward(self, x):
        out = self.net(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.0001, momentum=0)
        return optimizer

    def loss_function(self, pred, labels):
        return self.svm_criterion(pred, labels)
