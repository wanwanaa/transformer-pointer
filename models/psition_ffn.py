import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_size = config.model_size
        self.d_ff = config.d_ff

        self.w1 = nn.Linear(self.model_size, self.d_ff)
        self.w2 = nn.Linear(self.d_ff, self.model_size)
        self.layer_norm = nn.LayerNorm(self.model_size, eps=1e-6)
        self.dropout1 = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        :param x: (batch, len, model_size)
        :return: (batch, len, model_size)
        """
        inter = self.dropout1(self.relu(self.w1(self.layer_norm(x))))
        output = self.dropout2(self.w2(inter))
        return output + x