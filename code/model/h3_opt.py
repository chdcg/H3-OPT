# -*- coding: utf-8 -*-
"""

"""
import os
import pandas as pd
import torch,esm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.AF2_models import RowAttentionWithPairBias,OuterProductMean
from model.triangular_multiplicative_update_init import *

def get_sequence(feature_dir,pdbid):
    # load sequence
    data = []
    feats = pd.read_pickle(feature_dir + '%s_res.pkl'%pdbid)
    H_seq = feats['H_seq']
    L_seq = feats['L_seq']
    seq = (pdbid, H_seq + L_seq)
    data.append(seq)
    return data

class AFNet_ft(torch.nn.Module):
    def __init__(self, input_dim):
        super(AFNet_ft, self).__init__()
        self.AFNet = AFNet(input_dim) 
        self.esm_model,self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    
    def forward(self,x,pair,pdbid,feature_dir):
        ft_embedding = self.embed_seq(feature_dir,pdbid,x.device)
        out = self.AFNet(x,pair,ft_embedding)
        return out

    def embed_seq(self,feature_dir,pdbid,device):
        # get embedding representations of ESM2
        batch_converter = self.alphabet.get_batch_converter(truncation_seq_length=160)
        data = get_sequence(feature_dir,pdbid)
        batch_label, batch_str, batch_token = batch_converter(data)
        results = self.esm_model(batch_token.to(device), repr_layers=[33])
        token_representations = results["representations"][33][:,0,:]
        return token_representations
     


class AFNet(torch.nn.Module):
    def __init__(self, input_dim):
        super(AFNet, self).__init__()
        self.hidden_size = 64
        self.drop_out_rate = 0
        self.input_dim = 35
        self.outsize = 132
        self.encoder_layer = RowAttentionWithPairBias(pair_dim=60,dropout=0)
        self.pretrain_linear = nn.Linear(1280,160)
        self.outproduct = OuterProductMean(34,c_z=60,c_hidden=16)
        self.incomtriangle = TriangleMultiplicationIncoming(60, 32)
        self.outcomtriangle = TriangleMultiplicationOutgoing(60,32)
        self.relpos_linear = nn.Linear(65, 60)

        self.out  = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.drop_out_rate),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.drop_out_rate),
            nn.Linear(self.hidden_size, self.outsize),
            nn.ReLU(),
        )

    def forward(self, x,pair,pretrained_seq):
        residue_idx = torch.arange(0,x.shape[1]).unsqueeze(0).repeat(x.shape[0],1)
        a = relpos(residue_idx).float()
        pair = pair + self.relpos_linear(a.to(x.device))#add position encoding to pair
        del a, residue_idx

        encoder_out,attn = self.encoder_layer(x,pair) # same dims with x
        for i in range(3):
            encoder_out,attn = self.encoder_layer(encoder_out,pair)
            pair = self.outproduct(encoder_out)
            pair = self.outcomtriangle(pair)
            pair = self.incomtriangle(pair)

        pre_seq = self.pretrain_linear(pretrained_seq.squeeze(1)).view(x.shape[0],160,1)
        out = torch.cat((encoder_out,pre_seq),dim=2)
        ''' Incorporating sequence embeddings from ESM2 into residue-level representations ''' 
        output = self.out(out).sum(dim = 1).unsqueeze(1)
        return output


