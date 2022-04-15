#  -*- coding: utf-8 -*-
# @Time : 2019/11/15 下午10:42
# @Author : Xianli Zhang
# @Email : xlbryant@stu.xjtu.edu.cn
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from .layers import EncoderLayer
from .modules import MC_dropout
import numpy as np
import torchsparseattn
from doctor.sparsemax import Sparsemax

def get_attn_key_pad_mask(seq_q, mask):
    mb_size, len_q, _ = seq_q.size()
    pad_attn_mask = mask.data.eq(0).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_q) # bxsqxsk
    return pad_attn_mask


def get_non_pad_mask(mask):
    return mask.ne(0).type(torch.float).unsqueeze(-1)


def mask_softmax(x, mask):
    exp = torch.exp(x)
    masked_exp = exp * mask
    sum_masked_exp = torch.sum(masked_exp, dim=1, keepdim=True)
    return masked_exp / sum_masked_exp


def sparsemax(x, mask):
    x = x.cpu()
    fusedmax = torchsparseattn.Fusedmax(alpha=.2)
    mask = torch.sum(mask.squeeze(2), 1).long()
    return fusedmax(x.squeeze(2), mask.unsqueeze(1)).unsqueeze(2).cuda()


class Encoder(nn.Module):

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, capture_uncertainty=False, dropout=0.1):
        super(Encoder, self).__init__()
        self.capture_uncertainty = capture_uncertainty
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, capture_uncertainty, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, seq, mask, return_attns=False):
        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq, mask)
        non_pad_mask = get_non_pad_mask(mask)

        # -- Forward
        # enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        enc_output = None
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                seq,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class Inprem(nn.Module):
    def __init__(self, task, input_dim, out_dim, emb_dim, max_visit, n_layers, n_head, d_k, d_v, d_inner,
                 capture_uncertainty=False, mcDrop=0.5, analysis=False, delete_p=False, delete_vp=False, delete_fused=False):
        super(Inprem, self).__init__()
        self.capture_uncertainty = capture_uncertainty
        self.mc_Drop = mcDrop
        self.analysis = analysis
        self.delete_p = delete_p
        self.delete_vp = delete_vp
        self.delete_fused = delete_fused

        self.embedding = nn.Linear(input_dim, emb_dim, bias=False)
        init.xavier_normal_(self.embedding.weight)

        self.position_embedding = nn.Embedding(max_visit+1, emb_dim)

        self.encoder = Encoder(n_layers, n_head, d_k, d_v, emb_dim, d_inner, capture_uncertainty, mcDrop)

        self.w_alpha_1 = nn.Linear(emb_dim, 1, bias=True)
        self.w_alpha_2 = nn.Linear(emb_dim, 1, bias=True)

        init.xavier_normal_(self.w_alpha_1.weight)
        self.w_alpha_1.bias.data.zero_()
        init.xavier_normal_(self.w_alpha_2.weight)
        self.w_alpha_2.bias.data.zero_()


        self.w_beta = nn.Linear(emb_dim, emb_dim, bias=True)
        init.xavier_normal_(self.w_beta.weight)
        self.w_beta.bias.data.zero_()

        if self.capture_uncertainty:
            variance_dim = 1
            if task == 'diagnoses':
                variance_dim = out_dim
                out_dim = out_dim * 2
            self.variance = nn.Linear(emb_dim, variance_dim)
            init.xavier_normal_(self.variance.weight)

        self.predict = nn.Linear(emb_dim, out_dim)
        init.xavier_normal_(self.predict.weight)
        self.predict.bias.data.zero_()

        self.dropout = nn.Dropout(p=mcDrop)
        self.sparsemax = Sparsemax(dim=1)

    def forward(self, seq, mask):
        mask = Variable(mask, requires_grad=False)

        if self.capture_uncertainty:
            seq = MC_dropout(seq, p=self.mc_Drop, train=True)
        else:
            seq = self.dropout(seq)

        emb = self.embedding(seq)
        if not self.delete_p:
            length = torch.sum(mask, dim=1).cpu().data.numpy()
            position = []
            for len in length:
                posi = [len - i for i in range(int(len))]
                for i in range(mask.shape[1] - int(len)):
                    posi.append(0)
                position.append(posi)
            position = np.array(position)
            position = Variable(torch.from_numpy(position)).long()
            if mask.is_cuda:
                position = position.cuda()
            position_emb = self.position_embedding(position)
            emb = emb + position_emb

        hidden = self.encoder(emb, mask)
        alpha = self.w_alpha_1(hidden)

        if self.delete_fused:
            alpha = mask_softmax(alpha, mask.unsqueeze(2))
        else:
            length = torch.sum(mask, dim=1).cpu().data.numpy()
            alpha_ = []
            for i, item in enumerate(alpha):
                if int(length[i]) < mask.shape[1]:
                    alpha_.append(torch.cat(
                        (self.sparsemax(item[:int(length[i])].squeeze(1).unsqueeze(0)),
                         torch.zeros(1, mask.shape[1]-int(length[i])).float().cuda()), dim=1)
                    )
                else:
                    alpha_.append(self.sparsemax(item[:int(length[i])].squeeze(1).unsqueeze(0)))
            alpha = (torch.cat(alpha_, dim=0).unsqueeze(2))*0.5 + \
                    mask_softmax(alpha, mask.unsqueeze(2))*0.5

        beta = self.w_beta(hidden)

        if self.capture_uncertainty:
            beta = MC_dropout(beta, p=self.mc_Drop, train=True)
        else:
            beta = self.dropout(beta)

        beta = torch.tanh(beta * mask.unsqueeze(2))

        context = torch.bmm(torch.transpose(alpha, 1, 2), beta * emb).squeeze(1)
        logit = self.predict(context)

        if self.analysis:
            variance = self.variance(context)
            variance = variance * variance
            logit = torch.cat((logit, variance), 1)
            return seq.cpu().data.numpy(), position.cpu().data.numpy(), logit.cpu().data.numpy(), \
                   alpha.cpu().data.numpy(), beta.cpu().data.numpy(), emb.cpu().data.numpy(), \
                   position_emb.cpu().data.numpy(), mask.cpu().data.numpy()

        if self.capture_uncertainty:
            variance = F.softplus(self.variance(context))
            variance = variance * variance
            return torch.cat((logit, variance), 1)
        return logit