# -*- coding: utf-8 -*-
"""

"""
import random
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class OuterProductMean(nn.Module):
    """
    Implements Algorithm 10.
    """

    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):
        """
        Args:
            c_m:
                Residue level embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super(OuterProductMean, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = nn.Linear(c_m, c_hidden)
        self.linear_2 = nn.Linear(c_m, c_hidden)
        self.linear_out = nn.Linear(c_hidden ** 2, c_z)

    def _opm(self, a, b):
        outer = torch.einsum("...bac,...dae->...bdce", a, b)
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        outer = self.linear_out(outer)
        return outer


    def forward(self,
                m: torch.Tensor,
                mask = None,
                chunk_size = None,
                ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            mask:
                [*, N_seq, N_res] MSA mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        m = m.unsqueeze(1)
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        ln = self.layer_norm(m)
        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln)
        a = a * mask

        b = self.linear_2(ln)
        b = b * mask

        del ln
        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)
        outer = self._opm(a, b)

        # [*, N_res, N_res, 1]
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        # [*, N_res, N_res, C_z]

        outer = outer / norm

        return outer

class dot_attention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(dot_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2) # for rv

    def forward(self, gate,q, k, v,pair, scale=None, attn_mask=None):

        #q : [*, src, head, h_dim]
        q = q.permute(0,2,1,3) #[*, head,src,h_dim ]
        k = k.permute(0,2,3,1) #[*, head,h_dim, src]
        attention = torch.matmul(q,k).permute(0,2,3,1) #[*, src,src, head]
        if scale:
            attention = attention * scale  # scared or not
        if attn_mask is not None:
            attention = attention.masked_fill(attn_mask, -np.inf)     # mask
        # add pair bias to attention
        attention = attention + pair
        # softmax
        attention = self.softmax(attention)
        # dropout
        attention = self.dropout(attention)
        # multipy with V
        context = torch.matmul(attention.permute(0,3,1,2), v.permute(0,2,1,3)).permute(0,2,1,3)
        # dot point with gate
        context = context * gate
        return context, attention


class RowAttentionWithPairBias(nn.Module):
    def __init__(self, pair_dim=39, model_dim=34,num_heads=2, dropout=0.0):
        super(RowAttentionWithPairBias, self).__init__()
        self.dim_per_head = model_dim//num_heads   # 每个头的维度
        self.num_heads = num_heads
        self.linear = nn.Linear(model_dim, model_dim)
        self.dot_product_attention = dot_attention(dropout)
        self.linearNoBias = nn.Linear(model_dim, model_dim,bias=False)
        self.linear_q = nn.Linear(model_dim, model_dim,bias=False)
        self.linear_k = nn.Linear(model_dim, model_dim,bias=False)
        self.linear_v = nn.Linear(model_dim, model_dim,bias=False)
        self.linearPair = nn.Linear(pair_dim,num_heads,bias = False)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)         # LayerNorm
        self.layer_norm_pair = nn.LayerNorm(pair_dim)
        self.sigmoid = nn.Sigmoid()
        self.transition_layer = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )
    def forward(self, value, pair, attn_mask=None):
        # residual
        value = self.layer_norm(value)
        residual = value
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = value.size(0)
        src = value.size(1)
        # linear
        gate = self.sigmoid(self.linear(value))
        key = self.linear_k(value)
        value = self.linear_v(value)
        query = self.linear_q(value)
        pair_bais = self.linearPair(self.layer_norm_pair(pair)) # src, scr, Nhead
        # divide by heads
        gate = gate.view(batch_size, src, num_heads, dim_per_head)
        key = key.view(batch_size, src,num_heads, dim_per_head) # batch_size * Nhead, src,head_dim
        value = value.view(batch_size, src, num_heads, dim_per_head)
        query = query.view(batch_size, src,num_heads, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(1, 1, num_heads).reshape(batch_size,src,src,num_heads) #  batch_size * Nhead, src, 1

        # scaled multi-heads
        scale = (self.dim_per_head) ** -0.5
        context, attention = self.dot_product_attention(gate, query, key, value, pair_bais,scale, attn_mask)

        # concat heads
        context = context.reshape(batch_size, src , dim_per_head * num_heads)
        # linear
        output = self.linear_final(context)
        # dropout
        output = self.dropout(output)


        output = self.layer_norm(residual + output)
        output = self.transition_layer(output)
        return output, attention


