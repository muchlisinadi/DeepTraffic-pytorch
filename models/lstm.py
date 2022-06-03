import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LitBase


class RNN(nn.Module):
    def __init__(
        self, input_size=28, hidden_size=128, num_layers=2, num_classes=10, **kwargs
    ):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        x = x.reshape(-1, self.input_size, self.input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Passing in the input and hidden state into the model and  obtaining outputs
        out, hidden = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out[:, -1, :])
        return out


class LitRNN(LitBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = RNN(**kwargs)

    def forward(self, x):
        out = self.net(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def loss_function(self, pred, labels):
        return F.cross_entropy(pred, labels)
