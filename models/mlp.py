import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LitBase


## Define the NN architecture
class MLP(nn.Module):
    def __init__(self, input_size=28, hidden_size=512, num_classes=10, **kwargs):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size * input_size, hidden_size)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, self.input_size * self.input_size)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)


class LitMLP(LitBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = MLP(**kwargs)

    def forward(self, x):
        out = self.net(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def loss_function(self, pred, labels):
        return F.cross_entropy(pred, labels)
