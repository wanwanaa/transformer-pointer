import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder
from models.bert import Bert


class AE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.bert = Bert(config)
        self.decoder_ae = Decoder(config)
        self.t_len = config.t_len
        self.s_len = config.s_len
        self.pad = config.pad
        self.bos = config.bos
        self.model_size = config.model_size
        self.linear_bert = nn.Linear(768, config.model_size)
        self.linear_out = nn.Linear(config.model_size, config.tgt_vocab_size)
        self.linear_ae = nn.Linear(config.model_size, config.tgt_vocab_size)

    # add <bos> to sentence
    def convert(self, x):
        """
        :param x:(batch, s_len) (word_1, word_2, ... , word_n)
        :return:(batch, s_len) (<bos>, word_1, ... , word_n-1)
        """
        if torch.cuda.is_available():
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.cuda.LongTensor)
        else:
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.LongTensor)
        x = torch.cat((start, x), dim=1)
        return x[:, :-1]

    def forward(self, x, y):
        y_s = self.convert(y)
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(x, y_s, encoder_out)
        decoder_out = self.linear_out(decoder_out)
        out = torch.nn.functional.softmax(decoder_out)
        out = torch.argmax(out, dim=-1)

        h = self.bert(out)
        h = self.linear_bert(h)
        output = self.decoder_ae(out, y_s, h)
        final_out = self.linear_ae(output)
        return decoder_out, final_out

    def sample(self, x):
        enc_output = self.encoder(x)
        # <start> connect to decoding input at each step
        start = torch.ones(x.size(0)) * self.bos
        start = start.unsqueeze(1)
        if torch.cuda.is_available():
            start = start.type(torch.cuda.LongTensor)
        else:
            start = start.type(torch.LongTensor)
        # the first <start>
        out = torch.ones(x.size(0)) * self.bos
        out = out.unsqueeze(1)
        dec_output = None
        for i in range(self.s_len):
            if torch.cuda.is_available():
                out = out.type(torch.cuda.LongTensor)
            else:
                out = out.type(torch.LongTensor)
            dec_output = self.decoder(x, out, enc_output)
            # print('dec:', dec_output.size())
            dec_output = self.linear_out(dec_output)  # (batch, len, vocab_size)
            # print(dec_output.size())
            gen = torch.nn.functional.softmax(dec_output, dim=-1)
            gen = torch.argmax(gen, dim=-1)  # (batch, len) eg. 1, 2, 3
            # print("start:", start.size())
            # print('gen:', gen.size())
            out = torch.cat((start, gen), dim=1)  # (batch, len+1) eg. <start>, 1, 2, 3

        outs = out[:, 1:]
        # bert
        h = self.bert(outs)
        h = self.linear_bert(h)

        # the first <start>
        out = torch.ones(x.size(0)) * self.bos
        out = out.unsqueeze(1)
        # decoder
        final_output = None
        for i in range(self.s_len):
            if torch.cuda.is_available():
                out = out.type(torch.cuda.LongTensor)
            else:
                out = out.type(torch.LongTensor)
            final_output = self.decoder(outs, out, h)
            final_output = self.linear_ae(final_output)  # (batch, len, vocab_size)
            gen = torch.nn.functional.softmax(final_output, -1)
            gen = torch.argmax(gen, dim=-1)  # (batch, len) eg. 1, 2, 3
            out = torch.cat((start, gen), dim=1)  # (batch, len+1) eg. <start>, 1, 2, 3
        return dec_output, final_output, out[:, 1:]






