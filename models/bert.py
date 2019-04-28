import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertForMaskedLM, BertTokenizer


class Bert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fine_tune = config.fine_tune
        self.model = BertModel.from_pretrained('bert-base-chinese')
        self.cls = config.cls
        self.sep = config.sep

    def input_Norm(self, x):
        if torch.cuda.is_available():
            start = (torch.ones(x.size(0), 1) * self.cls).type(torch.cuda.LongTensor)
        else:
            start = (torch.ones(x.size(0), 1) * self.cls).type(torch.LongTensor)
        x = torch.cat((start, x), dim=1)
        return x

    def forward(self, x):
        """
        :param x: (batch, len)
        :return: (batch, len, model_size)
        """
        x = self.input_Norm(x)
        segments_tensors = torch.zeros_like(x)
        if self.fine_tune:
            h, _ = self.model(x, segments_tensors)
        else:
            with torch.no_grad():
                h, _ = self.model(x, segments_tensors)
        return h[-1][:, 1:, :]
