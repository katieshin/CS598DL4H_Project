#  -*- coding: utf-8 -*-
# @Time : 2019/11/14 下午8:30
# @Author : Xianli Zhang
# @Email : xlbryant@stu.xjtu.edu.cn

import torch.nn as nn
from .subLayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, capture_uncertainty=False, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.capture_uncertainty = capture_uncertainty
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, capture_uncertainty, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, capture_uncertainty, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
