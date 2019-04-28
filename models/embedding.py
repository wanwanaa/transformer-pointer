import torch.nn as nn
import torch
import math


def PositionalEncoding(model_size, max_len=500):
    pe = torch.zeros(max_len, model_size)
    # (0, 1, 2, ...) (max_len, 1)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, model_size, 2, dtype=torch.float) *
                          -(math.log(10000.0) / model_size)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    # pe = pe.unsqueeze(1) # pe (max_len, 1, model_size)
    return pe


class Embeds(nn.Module):
    def __init__(self, model_size, vocab_size, config):
        super().__init__()
        self.model_size = model_size
        self.pe = nn.Embedding.from_pretrained(
            PositionalEncoding(model_size), freeze=True)
        self.embeds = nn.Embedding(vocab_size, model_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, step=None):
        emb = self.embeds(x) # (batch, len, model_size)
        emb = emb * math.sqrt(self.model_size)
        pos = torch.arange(0, x.size(1)).repeat(x.size(0), 1) # (batch, len)
        if torch.cuda.is_available():
            pos = pos.cuda()
        emb = emb + self.pe(pos)
        # if step is None:
        #     emb = emb + self.pe(pos)
        # else:
        #     emb = emb + self.pe(pos)
        emb = self.dropout(emb)
        return emb
