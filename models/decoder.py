import torch
import torch.nn as nn
from models.multi_headed_attn import MultiHeadedAttention
from models.psition_ffn import PositionwiseFeedForward
from models.embedding import Embeds


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadedAttention(config)
        self.enc_dec_attn = MultiHeadedAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.layer_norm1 = nn.LayerNorm(config.model_size, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(config.model_size, eps=1e-6)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, y, encoder_outs, src_pad_mask, tgt_pad_mask):
        """
        :param y: ( batch, tgt_len, model_size)
        :param encoder_outs: (batch, src_len, model_size)
        :param src_pad_mask: (batch, src_len, src_len)
        :param tgt_pad_mask: (batch, tgt_len, tgt_len)
        :return:
        """
        tgt_len = tgt_pad_mask.size(-1) # (batch, tgt_len, tgt_len)
        mask = torch.triu(
            torch.ones((tgt_len, tgt_len), dtype=torch.uint8), diagonal=1
        )
        if torch.cuda.is_available():
            mask = mask.type(torch.cuda.ByteTensor)
        mask = mask.unsqueeze(0).repeat(y.size(0), 1, 1)
        dec_mask = (tgt_pad_mask + mask).gt(0)
        inputs = self.layer_norm1(y)
        attn = self.self_attn(inputs, inputs, inputs, mask=dec_mask)
        attn = self.drop(attn) + y
        attn_norm = self.layer_norm2(attn)
        decoder_outs = self.enc_dec_attn(encoder_outs, encoder_outs, attn_norm, mask=src_pad_mask)
        output = self.feed_forward(self.drop(decoder_outs) + attn)
        return output


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pad = config.pad
        self.embedding = Embeds(config.model_size, config.tgt_vocab_size, config)
        self.decoder = nn.ModuleList(
            [DecoderLayer(config)
             for i in range(config.n_layer)])
        self.layer_morm = nn.LayerNorm(config.model_size, eps=1e-6)

    def forward(self, x, y, encoder_outs):
        """
        :param y:(batch,  tgt_len)
        :param encoder_outs: (batch, src_len, model_size)
        :return: (batch, src_len, model_size)
        """
        output = self.embedding(y)
        src_pad_mask = x.eq(self.pad).unsqueeze(1).repeat(1, y.size(1), 1)
        tgt_pad_mask = y.eq(self.pad).unsqueeze(1).repeat(1, y.size(1), 1)

        for layer in self.decoder:
            output = layer(output, encoder_outs, src_pad_mask, tgt_pad_mask)
        output = self.layer_morm(output)
        return output