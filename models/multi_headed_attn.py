import torch.nn as nn
import math
import torch


class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_size = config.model_size
        self.n_head = config.n_head
        self.dim_per_head = config.model_size // config.n_head

        self.linear_k = nn.Linear(self.model_size, self.model_size)
        self.linear_v = nn.Linear(self.model_size, self.model_size)
        self.linear_q = nn.Linear(self.model_size, self.model_size)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.dropout)
        self.final_linear = nn.Linear(self.model_size, self.model_size)

    def forward(self, key, value, query, mask=None):
        """
        :param key: (batch, len. model_size)
        :param value: (batch, len. model_size)
        :param query: (batch, len. model_size)
        :param mask: (batch, 1, len)
        :return: output(batch, len, model_size)
        """
        batch_size = key.size(0)

        def shape(x):
            # (batch, n_head, len, dim_per_head)
            return x.view(batch_size, -1, self.n_head, self.dim_per_head)\
                .transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous()\
                .view(batch_size, -1, self.n_head * self.dim_per_head)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)
        key = shape(key)
        value = shape(value)
        query = shape(query)

        # query (batch, n_head, len, dim_per_head)
        query = query / math.sqrt(self.dim_per_head)
        # scorce (batch, num_head, len, len)
        scores = torch.matmul(query, key.transpose(2, 3)).float()
        if mask is not None:
            mask = mask.unsqueeze(1) # (batch, 1, src_len, tgt_len)
            scores = scores.masked_fill(mask, -1e18)
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)

        # context (batch, n_head, len, dim_per_head)
        context = torch.matmul(drop_attn, value)
        # context (batch, len, dim_per_head)
        context = unshape(context)
        output = self.final_linear(context)
        return output