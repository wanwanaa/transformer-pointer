import torch.nn as nn
from models.multi_headed_attn import MultiHeadedAttention
from models.psition_ffn import PositionwiseFeedForward
from models.embedding import Embeds


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadedAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.model_size, eps=1e-6)

    def forward(self, x, mask):
        """
        :param x:(batch, src_len, model_size)
        :param mask: (batch, src_len, src_len)
        :return: (batch, src_len, model_size)
        """
        inputs = self.layer_norm(x)
        context = self.self_attn(inputs, inputs, inputs, mask=mask)
        out = self.dropout(context) + x
        return self.feed_forward(out)


class Encoder(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.pad = config.pad
        self.embedding = Embeds(config.model_size, vocab_size, config)
        self.encoder = nn.ModuleList(
            [EncoderLayer(config)
             for i in range(config.n_layer)])
        self.layer_norm = nn.LayerNorm(config.model_size, eps=1e-6)

    def forward(self, x):
        """
        :param x: (batch, len)
        :return:
        """
        out = self.embedding(x)

        mask = x.eq(self.pad).unsqueeze(1) # (batch, 1, src_len)
        mask = mask.repeat(1, x.size(1), 1) # (batch, src_len, src_len)
        for layer in self.encoder:
            out = layer(out, mask)
        out = self.layer_norm(out)
        return out