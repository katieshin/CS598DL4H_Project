import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, capture_uncertainty=False, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.capture_uncertainty = capture_uncertainty
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.mcDrop = attn_dropout

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)

        if self.capture_uncertainty:
            attn = MC_dropout(attn, p=self.mcDrop, train=True)
        else:
            attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


def MC_dropout(input, p=0.5, train=True):
    return F.dropout(input, p=p, training=train, inplace=False)
