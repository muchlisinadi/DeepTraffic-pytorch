import torch
import torch.nn as nn
import torch.nn.functional as F


class SVM_Loss(nn.modules.Module):
    def __init__(self):
        super(SVM_Loss, self).__init__()

    def forward(self, outputs, labels):
        return torch.sum(torch.clamp(1 - outputs.t() * labels, min=0)) / labels.shape[0]


class SVM(nn.Module):
    def __init__(self, input_size=28, num_classes=10, **kwargs):
        self.input_size = input_size
        self.fc = nn.Linear(input_size * input_size, num_classes)

    def forward(self, x):
        x = x.reshape(-1, self.input_size * self.input_size)
        x = self.fc(x)
        return x
