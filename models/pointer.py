import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder


class Pointer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(config.model_size*3, config.model_size),
            nn.SELU(),
            nn.Linear(config.model_size, config.model_size))
        self.linear_prob = nn.Linear(config.model_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, emb, hidden, context):
        """
        :param emb:(batch, 1, model_size)
        :param hidden: (batch, 1, model_size)
        :param context: (batch, 1, model_size)
        :return:(batch, c_len)
        """
        context = self.linear(torch.cat((emb, hidden, context), dim=-1))
        # -> (batch, 1, model_size) -> (batch, 1, 1)
        prob = self.sigmoid(self.linear_prob(context)).squeeze()
        return prob


class Luong_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_size = config.model_size

        self.linear_in = nn.Sequential(
            nn.Linear(config.model_size, config.model_size),
            nn.SELU(),
            nn.Linear(config.model_size, config.model_size)
        )
        self.linear_out = nn.Sequential(
            nn.Linear(config.model_size, config.model_size),
            nn.SELU(),
            nn.Linear(config.model_size, config.model_size)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, output, encoder_out):
        """
        :param output: (batch, 1, hidden_size) decoder output
        :param encoder_out: (batch, t_len, hidden_size) encoder hidden state
        :return: attn_weight (batch, time_step)
        """
        out = self.linear_in(output) # (batch, 1, hidden_size)
        out = out.transpose(1, 2) # (batch, hidden_size, 1)
        attn_weights = torch.bmm(encoder_out, out) # (batch, t_len, 1)
        attn_weights = self.softmax(attn_weights.transpose(1, 2)) # (batch, 1, t_len)
        context = torch.bmm(attn_weights, encoder_out)
        context = self.linear_out(context) # (batch, 1, model_size)

        return attn_weights.squeeze(), context


class Transformer_Pointer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_word = Encoder(config, config.src_vocab_size)
        self.encoder_char = Encoder(config, config.tgt_vocab_size)
        self.pointer = Pointer(config)
        self.attention = Luong_Attention(config)
        self.decoder = Decoder(config)
        self.linear_out = nn.Linear(config.model_size, config.tgt_vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.s_len = config.s_len
        self.bos = config.bos

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

    def forward(self, x_w, x_c, y):
        """
        :param x_w:
        :param x_c:
        :param y:
        :return: (batch, s_len, vocab_size)
        """
        y_s = self.convert(y)
        encoder_out = self.encoder_word(x_w)
        encoder_attn = self.encoder_char(x_c)
        final = []
        for i in range(self.s_len):
            dec_output = self.decoder(x_w, y_s[:, :i+1], encoder_out)
            emb = self.decoder.embedding(y_s[:, i].unsqueeze(1))
            output = self.linear_out(dec_output[:, -1, :])
            # gen (batch, vocab_size)
            gen = self.softmax(output)
            # pointer
            # ptr (batch, c_len)
            # context (batch, 1, model_size)
            ptr, context = self.attention(dec_output[:, -1, :].unsqueeze(1), encoder_attn)
            # prob (batch, )
            prob = self.pointer(emb, dec_output[:, -1, :].unsqueeze(1), context).unsqueeze(1)
            final_out = (1-prob) * gen
            final_out = final_out.scatter_add_(1, x_c, prob*ptr)
            final.append(final_out)
        return torch.stack(final)

    def sample(self, x_w, x_c):
        encoder_out = self.encoder_word(x_w)
        encoder_attn = self.encoder_char(x_c)

        start = torch.ones(x_w.size(0)) * self.bos
        start = start.unsqueeze(1)
        if torch.cuda.is_available():
            start = start.type(torch.cuda.LongTensor)
        else:
            start = start.type(torch.LongTensor)
        # the first <start>
        out = torch.ones(x_w.size(0)) * self.bos
        out = out.unsqueeze(1)
        final = []
        for i in range(self.s_len):
            if torch.cuda.is_available():
                out = out.type(torch.cuda.LongTensor)
            else:
                out = out.type(torch.LongTensor)
            dec_output = self.decoder(x_w, out, encoder_out)
            emb = self.decoder.embedding(out[:, -1].unsqueeze(1))
            output = self.linear_out(dec_output[:, -1, :])
            gen = self.softmax(output)
            ptr, context = self.attention(dec_output[:, -1, :].unsqueeze(1), encoder_attn)
            # prob (batch, )
            prob = self.pointer(emb, dec_output[:, -1, :].unsqueeze(1), context).unsqueeze(1)
            final_out = (1 - prob) * gen
            final_out = final_out.scatter_add_(1, x_c, prob * ptr)
            final.append(final_out)
            gen = torch.argmax(gen, dim=-1).unsqueeze(1)
            out = torch.cat((out, gen), dim=1)
        return torch.stack(final), out